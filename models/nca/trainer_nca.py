import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as pl
import sys
from IPython.display import clear_output
from tqdm import tqdm_notebook, tnrange
import torchvision.models as models
from functools import partial
import cv2
from einops import rearrange
from torchvision.transforms.functional_tensor import gaussian_blur

from simulator import NCARunner, configure_nca, NCARunnerWithPool
from AgentTorch.helpers import read_config
from substeps.utils import AddAuxilaryChannel, InvariantLoss, IsoNcaOps, make_circle_masks
import torcheck
import wandb

def log_preds_table(images,loss):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image","loss"])
    for img, loss in zip(images.to("cpu"),loss.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255),loss )
    wandb.log({"predictions_table":table}, commit=False)
    

class TrainIsoNca:
    def __init__(self,runner):
        # fill values in cfg
        self.ops = IsoNcaOps()

        self.runner = runner
        self.device = torch.device(
            runner.config['simulation_metadata']['device'])
        
        self.CHN = runner.config['simulation_metadata']['chn']
        self.ANGLE_CHN = runner.config['simulation_metadata']['angle_chn']
        self.SCALAR_CHN = self.CHN-self.ANGLE_CHN
        self.AUX_L_TYPE = runner.config['simulation_metadata']['aux_l_type']
        self.TARGET_P = runner.config['simulation_metadata']['target']
        self.MODEL_TYPE = runner.config['simulation_metadata']['model_type']
        
        self.H = runner.config['simulation_metadata']['h']
        self.W = runner.config['simulation_metadata']['w']
        self.mirror = self.MODEL_TYPE in ['gradnorm', 'laplacian', 'lap6']
        self.hex_grid = self.MODEL_TYPE == "lap6"
        
        self.aux = AddAuxilaryChannel(
            self.TARGET_P, self.AUX_L_TYPE, self.H, self.W, self.MODEL_TYPE)
        
        self.loss_log = []
        
        try:
            self.model_suffix = self.MODEL_TYPE + "_" + \
            self.TARGET_P + "_" + self.AUX_L_TYPE + "_" + runner.config['simulation_metadata']['exp_no']
        except:
            self.model_suffix = self.MODEL_TYPE + "_" + \
            self.TARGET_P + "_" + self.AUX_L_TYPE
        
        if self.hex_grid:
            self.xy_grid = self.ops.get_xy_grid(self.W)
        
        
        self.target, self.aux_target = self.aux.get_targets() #Can Define your own target here
        self.target_loss_f = InvariantLoss(
            self.target, mirror=self.mirror, sharpen=True,hex_grid=self.hex_grid) #Can define your own loss function here

        self.opt = optim.Adam(self.runner.parameters(),
                                lr=self.runner.config['simulation_metadata']['learning_params']['lr'],
                                betas=self.runner.config['simulation_metadata']['learning_params']['betas'])
        self.lr_sched = optim.lr_scheduler.ExponentialLR(self.opt,
                                                            self.runner.config['simulation_metadata']['learning_params']['lr_gamma'])
        self.num_steps_per_episode = self.runner.config["simulation_metadata"]["num_steps_per_episode"]
        self.normalize_gradient =  False
        self.pool_size = self.runner.config["simulation_metadata"]["pool_size"]
        wandb.init(
        entity="blankpoint",
        project="NCA",         
        name=f"{self.model_suffix}", 
        config={
        "learning_rate" : self.runner.config['simulation_metadata']['learning_params']['lr'],
        "architecture": "CNN",
        "target": self.TARGET_P,
        "epochs": self.runner.config['simulation_metadata']['num_episodes'],
        })
        
    def train(self):
        print("Starting training...")
        print(f"Target is: {self.runner.config['simulation_metadata']['target']}",)
        for i in range(self.runner.config['simulation_metadata']['num_episodes']):
            with torch.no_grad():
                batch_idx = np.random.choice(len(self.pool_size), 8, replace=False)
                print(batch_idx)
                x = self.pool[batch_idx]
                print(x.shape)
                if len(self.loss_log) < 4000:
                    seed_rate = 1
                else:
                    # exp because of decrease of step_n
                    #seed_rate = 3
                    seed_rate = 6
                if i%seed_rate==0:
                    x[:1] = self.ca.seed(1, self.W)

                    #damage_rate = 3 # for spiderweb and heart
                damage_rate = 6  # for lizard?
                if i%damage_rate==0:
                    mask = torch.from_numpy(self.make_circle_masks(1, self.W, self.W)[:,None]).to("cuda")
                    if self.hex_grid:
                        mask = F.grid_sample(mask, self.xy_grid[None,:].repeat([len(mask), 1, 1, 1]), mode='bicubic')
                    x[-1:] *= (1.0 - mask)

                # EXTRA:
                # if all the cells have died, reset the sample.
                if len(self.loss_log) % 10 == 0:
                    all_cells_dead_mask = (torch.sum(x[1:, 3:4],(1,2,3)) < 1e-6).float()[:,None,None,None]
                    if all_cells_dead_mask.sum() > 1e-6:
                        
                        x[1:] = all_cells_dead_mask * self.ca.seed(7, self.W) + (1. - all_cells_dead_mask) * x[1:]

            # step_n = np.random.randint(64, 96)
            step_n = 5
            overflow_loss = 0.0
            diff_loss = 0.0
            target_loss = 0.0
            aux_target_loss = 0.0
            self.runner.reset()
            self.opt.zero_grad()

            self.runner.step(step_n)  # its is sampled randomly right now
            
            outputs = self.runner.state_trajectory[-1][-step_n:]
            list_outputs = [outputs[i]['agents']['automata']['cell_state'] for i in range(step_n)]
            x_intermediate_steps = torch.stack(list_outputs,dim=0)
            # x_intermediate_steps = x_intermediate_steps.permute([1,0,4,2,3])
            
            overflow_loss = (x_intermediate_steps-x_intermediate_steps.clamp(-2.0, 2.0)
                                )[:,:,:self.SCALAR_CHN].square().sum()

            final_step_output = outputs[-1]
            
            x_final_step = final_step_output['agents']['automata']['cell_state']
            # x_final_step = x_final_step.permute([0,3,1,2])
            target_loss = self.target_loss_f(x_final_step[:,:self.target.shape[0]])

            target_loss /= 2.
            aux_target_loss /= 2.
            diff_loss = diff_loss*10.0
            loss = target_loss + overflow_loss+diff_loss + aux_target_loss
            wandb.log({"loss": loss})
            wandb.log({' lr:', self.lr_sched.get_lr()[0]})
            with torch.no_grad():
                try:
                    loss.backward()
                except Exception as e:
                    print(e)
                    import ipdb
                    ipdb.set_trace()
                
                if self.normalize_gradients:
                    for p in self.runner.parameters():
                        p.grad /= (p.grad.norm()+1e-8)   # normalize gradients
                self.opt.step()
                self.lr_sched.step()

                # self.pool[batch_idx] = x                # update pool

                self.loss_log.append(loss.item())
                if i % 32 == 0:
                    clear_output(True)
                    pl.plot(self.loss_log, '.', alpha=0.1)
                    pl.yscale('log')
                    pl.ylim(np.min(self.loss_log), self.loss_log[0])
                    # pl.show()
                    pl.savefig('loss_curve.png')
                    imgs = self.ops.to_rgb(x_final_step)
                    if self.hex_grid:
                        imgs = F.grid_sample(imgs, self.xy_grid[None, :].repeat(
                            [len(imgs), 1, 1, 1]), mode='bicubic')
                    imgs = imgs.permute([0, 2, 3, 1]).cpu()
                    # cv2.imshow(self.ops.zoom(
                    #     self.ops.tile2d(imgs, 4), 2))
                    # self.ops.imshow(self.ops.zoom(
                    #     self.ops.tile2d(imgs, 4), 2))  # zoom
                    wandb.log({"loss curve":[wandb.Image('loss_curve.png',caption=f"loss for the episode {i}")]})
                    wandb.log({"output_result_image": [wandb.Image(im) for im in imgs]})
                    # log_preds_table(imgs,loss)

                    if self.AUX_L_TYPE != "noaux":
                        alphas = x_final_step[:, 3].cpu()
                        for extra_i in range(self.aux_target.shape[-3]):
                            imgs = 1. - alphas + alphas * \
                                (x_final_step[:, 4+extra_i].cpu() + 0.5)

                        if self.hex_grid:
                            imgs = F.grid_sample(
                                imgs[:, None], self.xy_grid[None, :].repeat(
                                    [len(imgs), 1, 1, 1]).cpu(),
                                mode='bicubic')[:, 0]
                        # self.ops.imshow(self.ops.zoom(
                        #     self.ops.tile2d(imgs, 8), 1))
                        # wandb.log({"aux layers": [wandb.Image(im) for im in self.ops.zoom(
                        #     self.ops.tile2d(imgs, 8), 1)]})
                if i % 10 == 0:
                    print('\rstep_n:', len(self.loss_log),
                        ' loss:', loss.item(),
                        ' lr:', self.lr_sched.get_lr()[0], end='')
                if len(self.loss_log) % 500 == 0:
                    model_name = self.model_suffix + \
                        "_{:07d}.pt".format(len(self.loss_log))
                    print(model_name)
                    torch.save(self.ca.state_dict(), model_name)
        torch.save(self.runner.state_dict(
        ), self.runner.config['simulation_metadata']['learning_params']['model_path'])


# *************************************************************************

if __name__ == "__main__":
    wandb.login()
    # Parsing command line arguments
    parser = argparse.ArgumentParser(
        description="AgentTorch: design, simulate and optimize agent-based models"
    )
    parser.add_argument(
        "-c", "--config", help="Name of the yaml config file with the parameters."
    )
    # *************************************************************************
    args = parser.parse_args()
    if not args:
        config_file = args.config
    
    else:
        config_file = "/Users/shashankkumar/Documents/AgentTorch/models/nca/config.yaml"
    
    config, registry = configure_nca(config_file)
    runner = NCARunnerWithPool(read_config('/Users/shashankkumar/Documents/AgentTorch-original/AgentTorch/models/nca/config_nca.yaml'), registry)
    runner.init()
    trainer = TrainIsoNca(runner)
    trainer.train()
