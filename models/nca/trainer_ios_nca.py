import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as pl

from IPython.display import clear_output
from tqdm import tqdm_notebook, tnrange
import torchvision.models as models
from functools import partial

from einops import rearrange
from torchvision.transforms.functional_tensor import gaussian_blur


from simulator import NCARunner, get_registry
from AgentTorch.helpers import read_config
from substeps.utils import AddAuxilaryChannel, InvariantLoss, IsoNcaOps, IsoNcaConfig, make_circle_masks



class TrainIsoNca:
    def __init__(self,config,runner):
        self.cfg = IsoNcaConfig()
        #fill values in cfg
        self.ops = IsoNcaOps(self.cfg)
        
        self.runner = runner
        self.device = torch.device(runner.config['simulation_metadata']['device'])        
        
        self.chn = self.cfg.CHN
        self.ANGLE_CHN = self.cfg.ANGLE_CHN
        self.SCALAR_CHN = self.cfg.CHN-self.cfg.ANGLE_CHN

        self.AUX_L_TYPE = self.cfg.AUX_L_TYPE
        self.TARGET_P = self.cfg.TARGET_P

        # self.model_type = self.cfg.model_type
        # self.model_suffix = self.model_type + "_" + \
        #     self.TARGET_P + "_" + self.AUX_L_TYPE

        self.loss_log = []

        self.mirror = self.cfg.model_type in ['gradnorm', 'laplacian', 'lap6'] #can make it boolean

        self.hex_grid = self.cfg.hex_grid  #add cond for this
        
        self.H = self.cfg.H
        self.W = self.cfg.W
        
        #add cond for hex
        self.s = np.sqrt(3)/2.0
        self.hex2xy = np.float32([[1.0, 0.0],
                                [0.5, self.s]])
        self.xy2hex = torch.tensor(np.linalg.inv(self.hex2xy))
        self.x = torch.linspace(-1, 1, self.W)
        self.y, self.x = torch.meshgrid(self.x, self.x)
        self.xy_grid = torch.stack([self.x, self.y], -1)
        # This grid will be needed later on, in the step functions.
        self.xy_grid = (self.xy_grid@self.xy2hex+1.0) % 2.0-1.
        
        
        
        self.aux = AddAuxilaryChannel(self.cfg)
        self.target, self.aux_target = self.aux.get_targets()
        self.target_loss_f = InvariantLoss(self.cfg,
            self.target, mirror=self.mirror, hex_grid=self.hex_grid)

        self.opt = optim.Adam(runner.parameters(), 
                lr=runner.config['simulation_metadata']['learning_params']['lr'], 
                betas=runner.config['simulation_metadata']['learning_params']['betas'])
        self.lr_sched = optim.lr_scheduler.ExponentialLR(self.opt, 
                        runner.config['simulation_metadata']['learning_params']['lr_gamma'])
        self.num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]
        # with torch.no_grad():
        #     self.pool = self.ca.seed(256, self.W)
        # self.opt = torch.optim.Adam(self.ca.parameters(), 1e-3)
        # # lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [1000, 3000, 20000], 0.3)
        # # for the experiment with auxiliary loss
        # # lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [3000, 10000], 0.3)
        # self.lr_sched = torch.optim.lr_scheduler.CyclicLR(
        #     self.opt, 1e-5, 1e-3, step_size_up=2000, mode='triangular2', cycle_momentum=False)

    def train(self):

        for i in range(self.runner.config['simulation_metadata']['num_episodes']):
            print(i)
            # with torch.no_grad():
            #     # batch_idx = np.random.choice(len(self.pool), 8, replace=False)
            #     # x = self.pool[batch_idx]
            #     # if len(self.loss_log) < 4000:
            #     #     seed_rate = 1
            #     # else:
            #     #     # exp because of decrease of step_n
            #     #     # seed_rate = 3
            #     #     seed_rate = 6
            #     # if i % seed_rate == 0:
            #     #     x[:1] = self.ca.seed(1, self.W)

            #         # damage_rate = 3 # for spiderweb and heart
            #     damage_rate = 6  # for lizard? #add parameter in config
            #     if i % damage_rate == 0:
            #         mask = torch.from_numpy(make_circle_masks(
            #             1, self.W, self.W)[:, None]).to(self.device)
            #         if self.hex_grid:
            #             mask = F.grid_sample(mask, self.xy_grid[None, :].repeat(
            #                 [len(mask), 1, 1, 1]), mode='bicubic')
            #         x[-1:] *= (1.0 - mask)

            #     # EXTRA:
            #     # if all the cells have died, reset the sample.
            #     if len(self.loss_log) % 10 == 0:
            #         all_cells_dead_mask = (
            #             torch.sum(x[1:, 3:4], (1, 2, 3)) < 1e-6).float()[:, None, None, None]
            #         if all_cells_dead_mask.sum() > 1e-6:
            #             print("got here.")
            #             x[1:] = all_cells_dead_mask * \
            #                 self.ca.seed(7, self.W) + \
            #                 (1. - all_cells_dead_mask) * x[1:]

            # step_n = np.random.randint(32, 128)
            # new!
            # everything worked but the unicorn pattern was constantly imploding with this.
            # step_n = np.random.randint(96, 128)
            step_n = np.random.randint(64, 96)
            overflow_loss = 0.0
            diff_loss = 0.0
            target_loss = 0.0
            self.runner.reset()
            self.opt.zero_grad()
            
            # px = x
            # x = self.ca(x)
            self.runner.step(step_n)  # its is sampled randomly right now
            output = self.runner.state_trajectory[-1][-1]
            x = output['agents']['automata']['cell_state']
            
            #diff_loss += (x-px).abs().mean()
            # scalar chn
            overflow_loss += (x-x.clamp(-2.0, 2.0)
                            )[:, :self.SCALAR_CHN].square().sum()

            # experimenting to address implosions:
            # if k == 0:
            #     # target
            #     target_loss += self.target_loss_f(
            #         x[:, :self.target.shape[0]])
            
            # if self.AUX_L_TYPE != "noaux":
            #     aux_target_loss += aux_target_loss_f(
            #         x[:,4:4+aux_target.shape[-3]]) * 2e-1
            

            target_loss += self.target_loss_f(x[:, :self.target.shape[0]])
            
            # if self.AUX_L_TYPE != "noaux":
            #     aux_target_loss += aux_target_loss_f(
            #         x[:,4:4+aux_target.shape[-3]]) * 2e-1
            
            target_loss /= 2.
            # aux_target_loss /= 2.
            diff_loss = diff_loss*10.0
            loss = target_loss+overflow_loss+diff_loss  # + aux_target_loss

            with torch.no_grad():
                try:
                    loss.backward()
                except:
                    import ipdb; ipdb.set_trace()
                # for p in self.ca.parameters():
                #     p.grad /= (p.grad.norm()+1e-8)   # normalize gradients
                self.opt.step()
                self.lr_sched.step()

                
                # self.pool[batch_idx] = x                # update pool

                self.loss_log.append(loss.item())
                if i % 32 == 0:
                    clear_output(True)
                    pl.plot(self.loss_log, '.', alpha=0.1)
                    pl.yscale('log')
                    pl.ylim(np.min(self.loss_log), self.loss_log[0])
                    pl.show()
                    imgs = self.ops.to_rgb(x)
                    if self.hex_grid:
                        imgs = F.grid_sample(imgs, self.xy_grid[None, :].repeat(
                            [len(imgs), 1, 1, 1]), mode='bicubic')
                    imgs = imgs.permute([0, 2, 3, 1]).cpu()

                    self.ops.imshow(self.ops.zoom(
                        self.ops.tile2d(imgs, 4), 2))  # zoom

                    if self.AUX_L_TYPE != "noaux":
                        alphas = x[:, 3].cpu()
                        for extra_i in range(self.aux_target.shape[-3]):
                            imgs = 1. - alphas + alphas * \
                                (x[:, 4+extra_i].cpu() + 0.5)

                        if self.hex_grid:
                            imgs = F.grid_sample(
                                imgs[:, None], self.xy_grid[None, :].repeat(
                                    [len(imgs), 1, 1, 1]).cpu(),
                                mode='bicubic')[:, 0]
                        self.ops.imshow(self.ops.zoom(
                            self.ops.tile2d(imgs, 8), 1))

                if i % 10 == 0:
                    print('\rstep_n:', len(self.loss_log),
                        ' loss:', loss.item(),
                        ' lr:', self.lr_sched.get_lr()[0], end='')
                if len(self.loss_log) % 500 == 0:
                    model_name = self.model_suffix + \
                        "_{:07d}.pt".format(len(self.loss_log))
                    print(model_name)
                    torch.save(self.ca.state_dict(), model_name)
        torch.save(self.runner.state_dict(), self.runner.config['simulation_metadata']['learning_params']['model_path'])
    


# *************************************************************************
# Parsing command line arguments

# parser = argparse.ArgumentParser(
#     description="AgentTorch: design, simulate and optimize agent-based models"
# )
# parser.add_argument(
#     "-c", "--config", help="Name of the yaml config file with the parameters."
# )
# # *************************************************************************
# args = parser.parse_args()
# config_file = args.config
if __name__ == "__main__":
    config_file = "/Users/shashankkumar/Documents/GitHub/NCA/AgentTorch/models/nca/config.yaml"
    config = read_config(config_file)
    registry = get_registry()
    runner = NCARunner(config, registry)
    runner.init()
    trainer = TrainIsoNca(config,runner)
    trainer.train()

