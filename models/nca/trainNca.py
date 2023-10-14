import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as pl
from IPython.display import clear_output
from simulator import NCARunner, configure_nca
from AgentTorch.helpers import read_config
from substeps.utils import AddAuxilaryChannel, InvariantLoss, IsoNcaOps


class TrainNca:
    def __init__(self, runner):
        
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
        self.loss_log = []
        self.model_suffix = self.MODEL_TYPE + "_" + \
            self.TARGET_P + "_" + self.AUX_L_TYPE
        if self.hex_grid:
            self.xy_grid = self.ops.get_xy_grid(self.W)

        self.aux = AddAuxilaryChannel(
            self.TARGET_P, self.AUX_L_TYPE, self.H, self.W, self.MODEL_TYPE)
        self.target, self.aux_target = self.aux.get_targets()
        self.target_loss_f = InvariantLoss(
            self.target, mirror=self.mirror, sharpen=True, hex_grid=self.hex_grid)
        self.opt = optim.Adam(self.runner.parameters(),
                                lr=self.runner.config['simulation_metadata']['learning_params']['lr'],
                                betas=self.runner.config['simulation_metadata']['learning_params']['betas'])
        self.lr_sched = optim.lr_scheduler.ExponentialLR(self.opt,
                                                            self.runner.config['simulation_metadata']['learning_params']['lr_gamma'])

    def train(self):
        print("Starting training...")
        print(
            f"Target is: {self.runner.config['simulation_metadata']['target']}",)

        for i in range(self.runner.config['simulation_metadata']['num_episodes']):
            step_n = np.random.randint(64, 96)
            #step_n = 5
            overflow_loss = 0.0
            diff_loss = 0.0
            target_loss = 0.0
            aux_target_loss = 0.0
            self.runner.reset()
            self.opt.zero_grad()

            self.runner.step(step_n)  # its is sampled randomly right now
            outputs = self.runner.state_trajectory[-1][-step_n:]
            
            list_outputs = [outputs[i]['agents']['automata']
                            ['cell_state'] for i in range(step_n)]
            x_intermediate_steps = torch.stack(list_outputs, dim=0)
            x_intermediate_steps = x_intermediate_steps.permute(
                [1, 0, 4, 2, 3])
            overflow_loss = (x_intermediate_steps-x_intermediate_steps.clamp(-2.0, 2.0)
                                )[:,:,:self.SCALAR_CHN].square().sum()

            final_step_output = outputs[-1]
            x_final_step = final_step_output['agents']['automata']['cell_state']
            x_final_step = x_final_step.permute([0, 3, 1, 2])
            target_loss = self.target_loss_f(
                x_final_step[:, :self.target.shape[0]])

            target_loss /= 2.
            aux_target_loss /= 2.
            diff_loss = diff_loss*10.0
            loss = target_loss+overflow_loss+diff_loss + aux_target_loss

            with torch.no_grad():
                # loss.backward()
                try:
                    loss.backward()
                except Exception as e:
                    print(e)
                self.opt.step()
                self.lr_sched.step()
                self.loss_log.append(loss.item())
                
                if i % 32 == 0:
                    clear_output(True)
                    pl.plot(self.loss_log, '.', alpha=0.1)
                    pl.yscale('log')
                    pl.ylim(np.min(self.loss_log), self.loss_log[0])
                    pl.show()
                    imgs = self.ops.to_rgb(x_final_step)
                    if self.hex_grid:
                        imgs = F.grid_sample(imgs, self.xy_grid[None, :].repeat(
                            [len(imgs), 1, 1, 1]), mode='bicubic')
                    imgs = imgs.permute([0, 2, 3, 1]).cpu()
                    self.ops.imshow(self.ops.zoom(
                        self.ops.tile2d(imgs, 4), 2))  # zoom
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
        
        print("Training complete")
        torch.save(self.runner.state_dict(
        ), self.runner.config['simulation_metadata']['learning_params']['model_path'])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="AgentTorch: design, simulate and optimize agent-based models"
    )
    parser.add_argument(
        "-c", "--config", help="Name of the yaml config file with the parameters."
    )
    
    args = parser.parse_args()
    if not args:
        config_file = args.config
    else:
        config_file = "/Users/shashankkumar/Documents/AgentTorch/models/nca/config.yaml"

    config, registry = configure_nca(config_file)
    runner = NCARunner(read_config(
        '/Users/shashankkumar/Documents/AgentTorch-original/AgentTorch/models/nca/config_nca.yaml'), registry)
    runner.init()
    trainer = TrainNca(runner)
    trainer.train()
