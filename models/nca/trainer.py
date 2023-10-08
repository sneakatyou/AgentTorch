from trainer_nca import TrainIsoNca
from simulator import NCARunner, configure_nca, configure_nca_with_multiple_experiments, get_registry
from AgentTorch.helpers import read_config
import argparse
import torch
import wandb

# class NcaConfig():
#     def __init__(self):
#         self.num_experiments = 1
#         self.ANGLE_CHN = [0]
#         self.CHN = [16]
#         self.DEFAULT_UPDATE_RATE = [0.5]

#         self.TARGET_P = ["lizard"] #@param ['circle','lizard', 'heart', 'smiley', 'lollipop', 'unicorn', 'spiderweb']
#         self.AUX_L_TYPE = ["binary"] #@param ['noaux', 'binary', 'minimal', 'extended']
#         self.H = self.W = [48]
#         self.model_type = ["laplacian"] #@param ['laplacian', 'lap6', 'lap_gradnorm', 'steerable', 'gradient', 'steerable_nolap']
#         self.sharpen = ["sharpen"]
#         self.mirror = ["False"]
#         self.hidden_n=[128]

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
    runner = NCARunner(read_config('/Users/shashankkumar/Documents/AgentTorch-original/AgentTorch/models/nca/config_nca.yaml'), registry)
    runner.init()
    trainer = TrainIsoNca(config, runner)
    trainer.train()
    
    #changes for running multiple experiments
    # exp_configs = NcaConfig()
    # if exp_configs.num_experiments > 1:
    #     for exp_no in range(exp_configs.num_experiments):
    #         config, registry = configure_nca_with_multiple_experiments(config_file,exp_configs,exp_no)
    #         runner = NCARunner(read_config(config, registry))
    #         runner.init()
    #         trainer = TrainIsoNca(config, runner)
    #         trainer.train()
    wandb.finish()