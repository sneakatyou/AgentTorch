import sys

sys.path.insert(0, '/Users/shashankkumar/Documents/AgentTorchLatest/AgentTorch')
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
from torchvision.transforms.functional_tensor import gaussian_blur
from AgentTorch.utils import initialise_wandb

from simulator import configure_nca, NCARunnerWithPool
from AgentTorch.helpers import read_config
from substeps.utils import AddAuxilaryChannel, InvariantLoss, IsoNcaOps, assign_method, make_circle_masks
import wandb
# torch.cuda.set_device(3)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def log_preds_table(images,loss):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image","loss"])
    for img, loss in zip(images.to("cpu"),loss.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255),loss )
    wandb.log({"predictions_table":table}, commit=False)
    

class TrainNca:
    def __init__(self,runner,custom_loss_fn = None, target = None, aux_target = None):
        # fill values in cfg
        self.ops = IsoNcaOps()

        self.runner = runner
        self.device = torch.device(
            runner.config['simulation_metadata']['device'])
        self.runner.to(self.device)
        self.custom_loss_fn = custom_loss_fn
        self.CHN = runner.config['simulation_metadata']['chn']
        self.ANGLE_CHN = runner.config['simulation_metadata']['angle_chn']
        self.SCALAR_CHN = self.CHN-self.ANGLE_CHN
        self.AUX_L_TYPE = runner.config['simulation_metadata']['aux_l_type']
        self.TARGET_P = runner.config['simulation_metadata']['target']
        self.MODEL_TYPE = runner.config['simulation_metadata']['model_type']
        self.normalize_gradients = True
        self.H = runner.config['simulation_metadata']['h']
        self.W = runner.config['simulation_metadata']['w']
        self.mirror = self.MODEL_TYPE in ['gradnorm', 'laplacian', 'lap6']
        self.hex_grid = self.MODEL_TYPE == "lap6"
        
        self.aux = AddAuxilaryChannel(
            self.TARGET_P, self.AUX_L_TYPE, self.H, self.W, self.MODEL_TYPE,self.device)
        
        self.loss_log = []
        
        try:
            self.model_suffix = self.MODEL_TYPE + "_" + \
            self.TARGET_P + "_" + self.AUX_L_TYPE + "_" + runner.config['simulation_metadata']['exp_no']
        except:
            self.model_suffix = self.MODEL_TYPE + "_" + \
            self.TARGET_P + "_" + self.AUX_L_TYPE
        
        if self.hex_grid:
            self.xy_grid = self.ops.get_xy_grid(self.W)
        
        if target is None:
            self.target, self.aux_target = self.aux.get_targets() #Can Define your own target here
        else:
            self.target = target
            if aux_target is not None:
                self.aux_target = aux_target
        
        if custom_loss_fn is None:
            self.target_loss_f = InvariantLoss(
            self.target, mirror=self.mirror, sharpen=True,hex_grid=self.hex_grid,device = self.device) #Can define your own loss function here

        self.opt = optim.Adam(self.runner.parameters(),
                                lr=self.runner.config['simulation_metadata']['learning_params']['lr'],
                                betas=self.runner.config['simulation_metadata']['learning_params']['betas'])
        self.lr_sched = optim.lr_scheduler.ExponentialLR(self.opt,
                                                            self.runner.config['simulation_metadata']['learning_params']['lr_gamma'])
        
        self.num_steps_per_episode = self.runner.config["simulation_metadata"]["num_steps_per_episode"]
        self.normalize_gradient =  False
        self.pool_size = self.runner.config["simulation_metadata"]["pool_size"]
        
        initialise_wandb(entity="blankpoint",
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
            # step_n = np.random.randint(64, 96)
            step_n = 50
            self.train_step(i, step_n)
        
        torch.save(self.runner.state_dict(
        ), self.runner.config['simulation_metadata']['learning_params']['model_path'])

    def train_step(self, i, step_n):
        self.runner.reset(1,i,len(self.loss_log))
        self.opt.zero_grad()
        self.runner.step(step_n) 
        x_final_step, loss = self.calculate_loss(step_n)
        self.wandb_log("loss",loss.item())
        self.wandb_log("lr",self.lr_sched.get_lr()[0])
            
        with torch.no_grad():
            try:
                loss.backward()
            except Exception as e:
                print(e)
                import ipdb
                ipdb.set_trace()
                
            if self.normalize_gradients:
                for p in self.runner.parameters():
                    p.grad = p.grad/(p.grad.norm()+1e-8) if p.grad is not None else p.grad   # normalize gradients
            
            self.opt.step()
            self.lr_sched.step()
            self.runner.update_pool(x_final_step)
                
            self.loss_log.append(loss.item())
            self.save_output(i, x_final_step)

    def wandb_log(self, name, value):
        wandb.log({name: value})

    def calculate_loss(self, step_n):
        if self.custom_loss_fn is not None:
            return self.custom_loss_fn(self.runner.state_trajectory[-1][-step_n:])
        else:
            outputs = self.runner.state_trajectory[-1][-step_n:]
            list_outputs = [outputs[i]['agents']['automata']['cell_state'] for i in range(step_n)]
            x_intermediate_steps = torch.stack(list_outputs,dim=0)
            
            overflow_loss = 0.0
            diff_loss = 0.0
            target_loss = 0.0
            aux_target_loss = 0.0
            
            overflow_loss = (x_intermediate_steps-x_intermediate_steps.clamp(-2.0, 2.0)
                                    )[:,:,:self.SCALAR_CHN].square().sum()

            final_step_output = outputs[-1]
                
            x_final_step = final_step_output['agents']['automata']['cell_state']
            target_loss = self.target_loss_f(x_final_step[:,:self.target.shape[0]])

            target_loss /= 2.
            aux_target_loss /= 2.
            diff_loss = diff_loss*10.0
            loss = target_loss + overflow_loss+diff_loss + aux_target_loss
            return x_final_step,loss

    def save_output(self, i, x_final_step):
            if i % 500 == 0:
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
                imgs = imgs.cpu()
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
                    # wandb.log({"aux layers": [wandb.Image(im) for im in self.ops.zoom(
                    #     self.ops.tile2d(imgs, 8), 1)]})
            if len(self.loss_log) % 1000 == 0:
                model_name = self.model_suffix + \
                            "_{:07d}.pt".format(len(self.loss_log))
                print(model_name)
                torch.save(self.runner.state_dict(), 
                            self.runner.config['simulation_metadata']['learning_params']['model_path'])
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
        config_file = "/Users/shashankkumar/Documents/AgentTorchLatest/AgentTorch/models/nca/config.yaml"
    
    params = {
    "num_episodes": 150000,
    "num_steps_per_episode": 5,
    "num_substeps_per_step": 1,
    "w": 48,
    "h": 48,
    "n_channels": 16,
    "batch_size": 8,
    "hidden_size": 128,
    "device": "cpu",
    "fire_rate": 0.5,
    "angle": 0.0,
    "learning_params": {
        "lr": 2e-3,
        "betas": [0.5, 0.5],
        "lr_gamma": 0.9999,
        "model_path": "saved_model.pth"
    },
    "angle_chn": 0,
    "chn": 16,
    "scalar_chn": 16,
    "update_rate": 0.5,
    "seed_size": 1,
    "pool_size": 128,
    "target": "heart",
    "aux_l_type": "binary",
    "model_type": "laplacian",
    "mirror": False,
    "alive_threshold_value": 0.1,
    "pool_size": 128,
    "hex_grid": False
    }
    
    config, registry = configure_nca(config_file,params)
    runner = NCARunnerWithPool(read_config(config_file), registry)
    
    runner.init()
    trainer = TrainNca(runner)
    trainer.train()
