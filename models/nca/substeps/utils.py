import numpy as np
import torch


from AgentTorch.helpers import *

import os
import io
import PIL.Image
import PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import matplotlib.pylab as pl
import glob

from IPython.display import Image, HTML, clear_output
from tqdm import tqdm_notebook, tnrange
import torch
import torch.nn.functional as F
import torchvision.models as models
from functools import partial

from einops import rearrange
from torchvision.transforms.functional_tensor import gaussian_blur


def nca_initialize_state(shape, params):
    device = torch.device(params['device'])
    batch_size = params['batch_size']
    n_channels = int(params['n_channels'].item())
    processed_shape = shape  # [process_shape_omega(s) for s in shape]
    grid_shape = [np.sqrt(processed_shape[0]).astype(int), np.sqrt(
        processed_shape[0]).astype(int), processed_shape[1]]
    seed_x = np.zeros(grid_shape, np.float32)
    seed_x[grid_shape[0]//2, grid_shape[1]//2, 3:] = 1.0
    x0 = np.repeat(seed_x[None, ...], batch_size, 0)
    x0 = torch.from_numpy(x0.astype(np.float32)).to(device)
    return x0


class IsoNcaOps():
    def __init__(self, cfg):
        self.cfg = cfg

    def imread(self, url, max_size=None, mode=None):
        if url.startswith(('http:', 'https:')):
            # wikimedia requires a user agent
            headers = {
                "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
            }
            r = requests.get(url, headers=headers)
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
        display(Image(data=self.imencode(a, fmt)))

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
                    [self.cfg.sobel_x, self.cfg.sobel_x.T]))
                # grad = torch.cat([grad, avg_dir], 1)
                # transform percieved vectors into local coords
                gx, gy = grad[:, ::2], grad[:, 1::2]
                rot_grad = torch.cat([gx*c+gy*s, gy*c-gx*s], 1)
                state_lap = self.perchannel_conv(state, self.cfg.lap[None, :])
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
                    [self.cfg.sobel_x, self.cfg.sobel_x.T]))
                # grad = torch.cat([grad, avg_dir], 1)
                # transform percieved vectors into local coords
                gx, gy = grad[:, ::2], grad[:, 1::2]
                rot_grad = torch.cat([gx*c+gy*s, gy*c-gx*s], 1)
                return torch.cat([state, rot_grad], 1)

        elif model_type == 'gradient':
            def perception(state):
                grad = self.perchannel_conv(state, torch.stack(
                    [self.cfg.sobel_x, self.cfg.sobel_x.T]))
                # gradient of the last channel determines the cell direction
                grad, dir = grad[:, :-2], grad[:, -2:]
                dir = dir/dir.norm(dim=1, keepdim=True).clip(1.0)
                c, s = dir[:, :1], dir[:, 1:2]
                # transform percieved vectors into local coords
                gx, gy = grad[:, ::2], grad[:, 1::2]
                rot_grad = torch.cat([gx*c+gy*s, gy*c-gx*s], 1)
                state_lap = self.perchannel_conv(state, self.cfg.lap[None, :])
                return torch.cat([state, state_lap, rot_grad], 1)

        elif model_type == 'lap_gradnorm':
            def perception(state):
                grad = self.perchannel_conv(state, torch.stack(
                    [self.cfg.sobel_x, self.cfg.sobel_x.T]))
                gx, gy = grad[:, ::2], grad[:, 1::2]
                state_lap = self.perchannel_conv(state, self.cfg.lap[None, :])
                return torch.cat([state, state_lap, (gx*gx+gy*gy+1e-8).sqrt()], 1)

        elif model_type == 'laplacian':
            def perception(state):
                state_lap = self.perchannel_conv(state, self.cfg.lap[None, :])
                return torch.cat([state, state_lap], 1)

        # add norm of gradients

        elif model_type == 'lap6':
            nhood_kernel = (self.cfg.lap6 != 0.0).to(torch.float32)

            def perception(state):
                state_lap = self.perchannel_conv(state, self.cfg.lap6[None, :])
                return torch.cat([state, state_lap], 1)

        else:
            assert False, "unknown model_type"

        return perception


class InvariantLoss:
    def __init__(self, cfg, target, mirror=False, sharpen=True, hex_grid=False):
        self.ops = IsoNcaOps(cfg)
        self.sharpen = sharpen
        self.mirror = mirror
        self.channel_n = target.shape[0]
        self.hex_grid = hex_grid
        self.W = target.shape[-1]
        self.r = r = torch.linspace(0.5/self.W, 1, self.W//2)[:, None]
        self.angle = a = torch.range(0, self.W*np.pi)/(self.W/2)
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
    def __init__(self, cfg):
        self.cfg = (cfg)

        self.TARGET_P = self.cfg.TARGET_P
        self.AUX_L_TYPE = self.cfg.AUX_L_TYPE
        self.H = self.cfg.H
        self.model_type = self.cfg.model_type
        self.W = self.cfg.W
        self.mask = self.make_circle_masks(self.H, self.W)
        self.ops = IsoNcaOps(cfg)
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
            target = self.ops.imread(url, 48)
            # imshow(target)
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
            return target, aux_target
        return target, -1


class IsoNcaConfig():
    def __init__(self):
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

        self.ANGLE_CHN = 0
        self.nhood_kernel = (self.lap != 0.0).to(torch.float32)
        self.CHN = 16
        self.SCALAR_CHN = self.CHN-self.ANGLE_CHN
        self.DEFAULT_UPDATE_RATE = 0.5

        # @param ['circle','lizard', 'heart', 'smiley', 'lollipop', 'unicorn', 'spiderweb']
        self.TARGET_P = "lizard"
        # @param ['noaux', 'binary', 'minimal', 'extended']
        self.AUX_L_TYPE = "binary"
        self.H = self.W = 48
        # @param ['laplacian', 'lap6', 'lap_gradnorm', 'steerable', 'gradient', 'steerable_nolap']
        self.model_type = "steerable"
        self.sharpen = "sharpen"
        self.mirror = "False"
        self.hex_grid = self.model_type == "lap6"
        self.model_suffix = "model"
        self.hidden_n = 128


class CA(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.isoOps = IsoNcaOps(cfg)
        self.chn = self.cfg.CHN
        self.ANGLE_CHN = self.cfg.ANGLE_CHN
        self.SCALAR_CHN = self.cfg.CHN-self.cfg.ANGLE_CHN
        # self.model_type = self.cfg.model_type
        self.hidden_n = self.cfg.hidden_n

        # self.perception = perception

        self.perception = self.isoOps.get_perception(self.cfg.model_type)
        self.nhood_kernel = self.cfg.nhood_kernel
        self.perchannel_conv = self.isoOps.perchannel_conv
        # determene the number of perceived channels
        perc_n = self.perception(torch.zeros([1, self.chn, 8, 8])).shape[1]
        # approximately equalize the param number btw model variants
        hidden_n = 8*1024//(perc_n+self.chn)
        hidden_n = (hidden_n+31)//32*32
        print('perc_n:', perc_n, 'hidden_n:', hidden_n)
        self.w1 = torch.nn.Conv2d(perc_n, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, self.chn, 1, bias=False)
        self.w2.weight.data.zero_()

    def forward(self, x, update_rate=0.5):
        alive = self.get_alive_mask(x)
        y = self.perception(x)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
        x = x + y*update_mask
        if self.SCALAR_CHN == self.chn:
            x = x*alive
        else:
            x = torch.cat([x[:, :self.SCALAR_CHN]*alive,
                            x[:, self.SCALAR_CHN:] % (np.pi*2.0)], 1)
        return x

    def seed(self, n, sz=128, angle=None, seed_size=1):
        x = torch.zeros(n, self.chn, sz, sz)
        if self.SCALAR_CHN != self.chn:
            x[:, -1] = torch.rand(n, sz, sz)*np.pi*2.0
        r, s = sz//2, seed_size
        x[:, 3:self.SCALAR_CHN, r:r+s, r:r+s] = 1.0
        if angle is not None:
            x[:, -1, r:r+s, r:r+s] = angle
        return x

    def get_alive_mask(self, x):
        mature = (x[:, 3:4] > 0.1).to(torch.float32)
        return self.perchannel_conv(mature, self.nhood_kernel[None, :]) > 0.5

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