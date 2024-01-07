import base64
import io
import types
import imageio
import matplotlib.pylab as pl
import numpy as np
import PIL.Image
import PIL.ImageDraw
import requests
import torch
import torch.nn.functional as F
from IPython.display import HTML, Image, clear_output
from torchvision.transforms.functional_tensor import gaussian_blur
from AgentTorch.helpers import *

def nca_initialize_state(shape, params):
    device = torch.device(params['device'])
    batch_size = params['batch_size']
    n_channels = int(params['n_channels'].item())
    processed_shape = shape #[process_shape_omega(s) for s in shape]
    grid_shape = [np.sqrt(processed_shape[0]).astype(int), np.sqrt(processed_shape[0]).astype(int), processed_shape[1]]
    seed_x = np.zeros(grid_shape, np.float32)
    seed_x[grid_shape[0]//2, grid_shape[1]//2, 3:] = 1.0
    x0 = np.repeat(seed_x[None, ...], batch_size, 0)
    x0 = torch.from_numpy(x0.astype(np.float32)).to(device)
    return x0

def nca_initialize_state_pool(shape, params):
        x = torch.zeros((int(params["pool_size"].item()), int(params["chn"].item()), 48,48))
        if params["scalar_chn"] != params["chn"]:
            x[:,-1] = torch.rand(params["pool_size"], 48, 48)*np.pi*2.0
        r, s = (params["w"]//2).int(),( params["seed_size"]).int()
        scalar_chn = int(params["scalar_chn"].item())
        x[:,3:scalar_chn,r:r+s, r:r+s] = 1.0
        if params["angle"].item() != 0.0:
            x[:,-1,r:r+s, r:r+s] = params["angle"]
        x.to(params["device"])
        return x
    

def assign_method(runner, method_name, method):
        setattr(runner, method_name, types.MethodType(method, runner))

class IsoNcaOps():
    def __init__(self, cfg = None):
        
        self.ident = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.lap = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]])
        self.lap6 = torch.tensor(
            [[0.0, 2.0, 2.0], [2.0, -12.0, 2.0], [2.0, 2.0, 0.0]])
        self.gauss = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])/16.0

    def get_xy_grid(self,W):
        s = np.sqrt(3)/2.0
        hex2xy = np.float32([[1.0, 0.0],
                                    [0.5, s]])
        xy2hex = torch.tensor(np.linalg.inv(hex2xy))
        x = torch.linspace(-1, 1, W)
        y, x = torch.meshgrid(x,x)
        xy_grid = torch.stack([x, y], -1)
        # This grid will be needed later on, in the step functions.
        xy_grid = (xy_grid@xy2hex+1.0) % 2.0-1.
        return xy_grid
        
    def imread(self, url, max_size=None, mode=None):
        if url.startswith(('http:', 'https:')):
            # wikimedia requires a user agent
            headers = {
                "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
            }
            r = requests.get(url, headers=headers,timeout=3)
            f = io.BytesIO(r.content)
        else:
            f = url
        img = PIL.Image.open(f)
        if max_size is not None:
            img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
        if mode is not None:
            img = img.convert(mode)
        img = np.float32(img)/255.0
        return img
    
    def load_emoji(self,index, path="/Users/shashankkumar/Documents/AgentTorch/models/nca/data/emoji.png"):
        im = imageio.imread(path)
        emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
        emoji /= 255.0
        return emoji

    def np2pil(self, a):
        if a.dtype in [np.float32, np.float64]:
            a = np.uint8(np.clip(a, 0, 1)*255)
        return PIL.Image.fromarray(a)

    def imwrite(self, f, a, fmt=None):
        a = np.asarray(a)
        if isinstance(f, str):
            fmt = f.rsplit('.', 1)[-1].lower()
            if fmt == 'jpg':
                fmt = 'jpeg'
            f = open(f, 'wb')
        self.np2pil(a).save(f, fmt, quality=95)

    def imencode(self, a, fmt='jpeg'):
        a = np.asarray(a)
        if len(a.shape) == 3 and a.shape[-1] == 4:
            fmt = 'png'
        f = io.BytesIO()
        self.imwrite(f, a, fmt)
        return f.getvalue()

    def im2url(self, a, fmt='jpeg'):
        encoded = self.imencode(a, fmt)
        base64_byte_string = base64.b64encode(encoded).decode('ascii')
        return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

    def imshow(self, a, fmt='jpeg'):
        try:
            display(Image(data=self.imencode(a, fmt)))
        except:
            pass
#             cv2.imshow("image",a)
            # img = Image(data=self.imencode(a, fmt))
            # img.show()
            
            
        

    def tile2d(self, a, w=None):
        a = np.asarray(a)
        if w is None:
            w = int(np.ceil(np.sqrt(len(a))))
        th, tw = a.shape[1:3]
        pad = (w-len(a)) % w
        a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
        h = len(a)//w
        a = a.reshape([h, w]+list(a.shape[1:]))
        a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
        return a

    def zoom(self, img, scale=4):
        img = np.repeat(img, scale, 0)
        img = np.repeat(img, scale, 1)
        return img

    def to_rgb(self, x):
        rgb, a = x[:, :3], x[:, 3:4]
        return 1.0-a+rgb

    def perchannel_conv(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        y = x.reshape(b*ch, 1, h, w)
        y = F.pad(y, [1, 1, 1, 1], 'circular')
        y = F.conv2d(y, filters[:, None])
        return y.reshape(b, -1, h, w)

    def make_concentric_discrete(self, h, w, n):
        # reference https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
        x = np.linspace(-1.0, 1.0, w)[None, :]
        y = np.linspace(-1.0, 1.0, h)[:, None]
        center = np.zeros([2, 1, 1])
        # r = np.array([0.8])[:,None]
        x, y = (x-center[0]), (y-center[1])
        act = np.sin if n % 2 == 0 else np.cos
        mask = np.sign(act(np.sqrt(x*x+y*y)*n*np.pi))
        return mask.astype(np.float32) * 0.5

    def make_concentric(self, h, w, n):
        # version with blurriness
        # reference https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
        x = np.linspace(-1.0, 1.0, w)[None, :]
        y = np.linspace(-1.0, 1.0, h)[:, None]
        center = np.zeros([2, 1, 1])
        # r = np.array([0.8])[:,None]
        x, y = (x-center[0]), (y-center[1])
        grad_start = 0.2
        act = np.sin if n % 2 == 0 else np.cos
        period = act(np.sqrt(x*x+y*y)*n*np.pi)
        grad = (np.abs(period) < grad_start).astype(np.float32)
        mask = np.sign(period) * (1-grad) + period * grad / grad_start
        return mask.astype(np.float32) * 0.5

    def make_stripes(self, h, w, n):
        x = np.linspace(-1.0, 1.0, w)[None, :]
        y = np.linspace(-1.0, 1.0, h)[:, None]
        center = np.zeros([2, 1, 1])
        # r = np.array([0.8])[:,None]
        x, y = (x-center[0])*np.ones_like(y), (y-center[1])*np.ones_like(x)
        act = np.sin if n % 2 == 0 else np.cos
        grad_start = 0.2
        x_period = act(x*n*np.pi)
        x_grad = (np.abs(x_period) < grad_start).astype(np.float32)
        x_mask = np.sign(x_period) * (1-x_grad) + \
            x_period * x_grad / grad_start
        y_period = act(y*n*np.pi)
        y_grad = (np.abs(y_period) < grad_start).astype(np.float32)
        y_mask = np.sign(y_period) * (1-y_grad) + \
            y_period * y_grad / grad_start
        return x_mask.astype(np.float32) * 0.5, y_mask.astype(np.float32) * 0.5

    def get_perception(self, model_type):
        if model_type == 'steerable':
            ANGLE_CHN = 1  # last state channel is angle and should be treated
            # differently

            def perception(state):
                state, angle = state[:, :-1], state[:, -1:]
                c, s = angle.cos(), angle.sin()

                # cells can also feel the average direction of their neightbours
                # alpha = state[:,3:4].clip(0.0, 1.0)
                # dir = torch.cat([c, s], 1)*alpha  # only
                # avg_dir = perchannel_conv(dir, gauss[None,:])
                grad = self.perchannel_conv(state, torch.stack(
                    [self.sobel_x, self.sobel_x.T]))
                # grad = torch.cat([grad, avg_dir], 1)
                # transform percieved vectors into local coords
                gx, gy = grad[:, ::2], grad[:, 1::2]
                rot_grad = torch.cat([gx*c+gy*s, gy*c-gx*s], 1)
                state_lap = self.perchannel_conv(state, self.lap[None, :])
                return torch.cat([state, rot_grad, state_lap], 1)

        elif model_type == 'steerable_nolap':
            ANGLE_CHN = 1  # last state channel is angle and should be treated
            # differently

            def perception(state):
                state, angle = state[:, :-1], state[:, -1:]
                c, s = angle.cos(), angle.sin()

                # cells can also feel the average direction of their neightbours
                # alpha = state[:,3:4].clip(0.0, 1.0)
                # dir = torch.cat([c, s], 1)*alpha  # only
                # avg_dir = perchannel_conv(dir, gauss[None,:])
                grad = self.perchannel_conv(state, torch.stack(
                    [self.sobel_x, self.sobel_x.T]))
                # grad = torch.cat([grad, avg_dir], 1)
                # transform percieved vectors into local coords
                gx, gy = grad[:, ::2], grad[:, 1::2]
                rot_grad = torch.cat([gx*c+gy*s, gy*c-gx*s], 1)
                return torch.cat([state, rot_grad], 1)

        elif model_type == 'gradient':
            def perception(state):
                grad = self.perchannel_conv(state, torch.stack(
                    [self.sobel_x, self.sobel_x.T]))
                # gradient of the last channel determines the cell direction
                grad, dir = grad[:, :-2], grad[:, -2:]
                dir = dir/dir.norm(dim=1, keepdim=True).clip(1.0)
                c, s = dir[:, :1], dir[:, 1:2]
                # transform percieved vectors into local coords
                gx, gy = grad[:, ::2], grad[:, 1::2]
                rot_grad = torch.cat([gx*c+gy*s, gy*c-gx*s], 1)
                state_lap = self.perchannel_conv(state, self.lap[None, :])
                return torch.cat([state, state_lap, rot_grad], 1)

        elif model_type == 'lap_gradnorm':
            def perception(state):
                grad = self.perchannel_conv(state, torch.stack(
                    [self.sobel_x, self.sobel_x.T]))
                gx, gy = grad[:, ::2], grad[:, 1::2]
                state_lap = self.perchannel_conv(state, self.lap[None, :])
                return torch.cat([state, state_lap, (gx*gx+gy*gy+1e-8).sqrt()], 1)

        elif model_type == 'laplacian':
            def perception(state):
                state_lap = self.perchannel_conv(state, self.lap[None, :])
                return torch.cat([state, state_lap], 1)

        # add norm of gradients

        elif model_type == 'lap6':
            nhood_kernel = (self.lap6 != 0.0).to(torch.float32)

            def perception(state):
                state_lap = self.perchannel_conv(state, self.lap6[None, :])
                return torch.cat([state, state_lap], 1)

        else:
            assert False, "unknown model_type"

        return perception


class InvariantLoss:
    def __init__(self, target, mirror=False, sharpen=True, hex_grid=False, device = "cpu"):
        self.ops = IsoNcaOps()
        self.device = device
        self.sharpen = sharpen
        self.mirror = mirror
        self.channel_n = target.shape[0]
        self.hex_grid = hex_grid
        self.W = target.shape[-1]
        self.r = r = torch.linspace(0.5/self.W, 1, self.W//2,device=self.device)[:, None]
        self.angle = a = torch.range(0, self.W*np.pi,device=self.device)/(self.W/2)
        self.polar_xy = torch.stack([r*a.cos(), r*a.sin()], -1)[None, :]
        
        # also make an x
        target = target[None, :]
        if self.sharpen:
            target = self.sharpen_filter(target)
        self.polar_target = F.grid_sample(target, self.polar_xy)
        self.fft_target = torch.fft.rfft(self.polar_target).conj()
        self.polar_target_sqnorm = self.polar_target.square().sum(-1, keepdim=True)

    def calc_losses(self, batch, extra_outputs=False):
        batch = batch[:, :self.channel_n]
        if self.sharpen:
            batch = self.sharpen_filter(batch)
        polar_batch = F.grid_sample(
            batch, self.polar_xy.repeat(len(batch), 1, 1, 1))
        X = torch.fft.rfft(polar_batch)
        n = polar_batch.shape[-1]
        xy = torch.fft.irfft(X*self.fft_target, n)
        if self.mirror:
            xy = torch.cat([xy, torch.fft.irfft(
                X*self.fft_target.conj(), n)], -1)
        xx = polar_batch.square().sum(-1, keepdim=True)
        yy = self.polar_target_sqnorm
        sqdiff = (xx+yy-2.0*xy)
        losses = sqdiff.mean([1, 2])
        if extra_outputs:
            return losses, batch, polar_batch
        else:
            return losses

    def __call__(self, batch):
        return self.calc_losses(batch).min(-1)[0].mean()

    def plot_losses(self, x):
        losses = self.calc_losses(x[None, :])[0].cpu()
        fig = pl.figure(figsize=(10, 10))
        ax0 = fig.add_subplot(111)
        vis = self.ops.to_rgb(x[None, :4])[0].permute(1, 2, 0).cpu().clip(0, 1)
        ax0.imshow(vis, alpha=0.5)
        ax0.axis("off")
        ax = fig.add_subplot(111, polar=True, label='polar')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_facecolor("None")
        ang = self.angle.cpu()
        if not self.mirror:
            ax.plot(ang, losses, linewidth=3.0)
        else:
            ax.plot(ang, losses[:len(ang)], linewidth=3.0)
            ax.plot(ang, losses[len(ang):], linewidth=3.0)
        min_i = losses.argmin()
        pl.plot(ang[min_i % len(ang)], losses[min_i], 'or', markersize=12)

    def sharpen_filter(self, img):
        blured = gaussian_blur(img, [5, 5], [1, 1])
        return img + (img-blured)*2.0


class AddAuxilaryChannel():
    def __init__(self, target_p, aux_l_type, h, w, model_type, device):
        self.TARGET_P = target_p
        self.AUX_L_TYPE = aux_l_type
        self.H = h
        self.model_type = model_type
        self.W = w
        self.mask = self.make_circle_masks(self.H, self.W)
        self.ops = IsoNcaOps()
        self.device = device
        self.model_suffix = self.model_type + "_" + \
            self.TARGET_P + "_" + self.AUX_L_TYPE

    def make_circle_masks(self, h, w):
        x = np.linspace(-1.0, 1.0, w)[None, :]
        y = np.linspace(-1.0, 1.0, h)[:, None]
        center = np.zeros([2, 1, 1])
        r = np.array([0.9])[:, None]
        x, y = (x-center[0])/r, (y-center[1])/r
        mask = (x*x+y*y < 1.0).astype(np.float32)
        return mask

    def get_targets(self):
        if self.TARGET_P == 'circle':
            mask = self.make_circle_masks(self.H, self.W)
            IS_COLORED = False
            if IS_COLORED:
                r = np.linspace(0, 1, self.H)[:, None]*mask
                g = np.linspace(0, 1, self.W)[None, :]*mask
            else:
                r = g = np.zeros(mask.shape)
            target_colors = np.stack([r, g, np.zeros(mask.shape)], -1)
            # target = np.zeros([H, W, 4], dtype=np.float32)

            # target[..., 3] = mask

            target = np.concatenate(
                [target_colors, mask[..., None]], -1).astype(np.float32)
            # imshow(target)
            target[:, :, :3] *= target[:, :, 3:]
        else:
            emoji = {'lizard': '🦎',
                        'heart': '❤️',
                        'smiley': '😁',
                        'lollipop': '🍭',
                        'unicorn': '🦄',  # overfits to the grid
                        'spiderweb': '🕸️'
                        }[self.TARGET_P][0]

            code = hex(ord(emoji))[2:].lower()
            url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true' % code
            try:
                target = self.ops.imread(url, 48)
            except:
                target = self.ops.load_emoji(index=0)
            self.ops.imshow(target)
            target[:, :, :3] *= target[:, :, 3:]

        p = 12
        # target = F.pad(torch.tensor(target).permute(2, 0, 1), [p, p, p, p, 0, 2])
        target = F.pad(torch.tensor(target).permute(
            2, 0, 1), [p, p, p, p, 0, 0])
        W = target.shape[1]

        Wp = Hp = target.shape[1]
        if self.AUX_L_TYPE != 'noaux':
            aux_target_l = []
            # x_mask, y_mask = make_stripes(Hp, Wp, 2)
            # aux_target_l += [torch.tensor(x_mask), torch.tensor(y_mask)]
            # x_mask, y_mask = make_stripes(Hp, Wp, 2)
            # aux_target_l += [torch.tensor(y_mask)]

            print(target[3:].shape)
            y_mask = torch.linspace(-1, 1, W)[:, None].sign()*target[3]*0.5
            aux_target_l += [y_mask]

            if self.AUX_L_TYPE == "extended":
                x_mask = torch.linspace(-1, 1, W)[None, :].sign()*target[3]*0.5
                aux_target_l += [torch.tensor(x_mask)]
                aux_target_l += [torch.tensor(self.ops.make_concentric(Hp, Wp, 2)),
                                    torch.tensor(
                                        self.ops.make_concentric(Hp, Wp, 3)),
                                    torch.tensor(self.ops.make_concentric(Hp, Wp, 4))]
            if self.AUX_L_TYPE == "minimal":
                aux_target_l += [torch.tensor(
                    self.ops.make_concentric(Hp, Wp, 4))]
            aux_target = torch.stack(aux_target_l)*target[3:4]
            target = target.to(self.device)
            aux_target = target.to(self.device)
            return target, aux_target
        return target, None

class IsoNcaConfig():
    def __init__(self,config):
        self.ident = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.lap = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]])
        self.lap6 = torch.tensor(
            [[0.0, 2.0, 2.0], [2.0, -12.0, 2.0], [2.0, 2.0, 0.0]])
        self.gauss = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])/16.0

        self.ANGLE_CHN = config['simulation_metadata']['angle_chn']
        self.nhood_kernel = (self.lap != 0.0).to(torch.float32)
        self.CHN = config['simulation_metadata']['chn']
        self.SCALAR_CHN = self.CHN-self.ANGLE_CHN
        self.DEFAULT_UPDATE_RATE = config['simulation_metadata']['update_rate']

        # @param ['circle','lizard', 'heart', 'smiley', 'lollipop', 'unicorn', 'spiderweb']
        self.TARGET_P = config['simulation_metadata']['target']
        # @param ['noaux', 'binary', 'minimal', 'extended']
        self.AUX_L_TYPE = config['simulation_metadata']['aux_l_type']
        self.H = self.W = config['simulation_metadata']['w']
        # @param ['laplacian', 'lap6', 'lap_gradnorm', 'steerable', 'gradient', 'steerable_nolap']
        self.model_type = config['simulation_metadata']['model_type']
        self.sharpen = config['simulation_metadata']['n_channels']
        self.mirror = config['simulation_metadata']['mirror']
        self.hex_grid = self.model_type == "lap6"
        self.model_suffix = "model"
        self.hidden_n = config['simulation_metadata']['hidden_n']

def make_circle_masks(n, h, w):
        x = np.linspace(-1.0, 1.0, w)[None, None, :]
        y = np.linspace(-1.0, 1.0, h)[None, :, None]
        center = np.random.uniform(-0.5, 0.5, [2, n, 1, 1])
        r = np.random.uniform(0.1, 0.4, [n, 1, 1])
        x, y = (x-center[0])/r, (y-center[1])/r
        mask = (x*x+y*y < 1.0).astype(np.float32)
        return mask
    
def get_parameter_count(self):
        param_n = sum(p.numel() for p in self.ca().parameters())
        return param_n