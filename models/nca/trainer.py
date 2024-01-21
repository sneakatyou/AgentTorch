from AgentTorch.models.nca.substeps.utils import assign_method
from trainer_nca import TrainIsoNca, TrainNca
from simulator import NCARunner, configure_nca
from AgentTorch.helpers import read_config
import argparse
import torch
import wandb

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
    "hex_grid": False,
    "trainable_transition_network" : True
    }
    
    if params['device'] != 'cpu':
        torch.cuda.set_device(params.device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    config, registry = configure_nca(config_file,params)
    runner = NCARunner(read_config(config_file), registry)
    
    def seed(self, pool_size, seed_size):
    # Generate a tensor with random values between 0 and 1
        x = torch.randint(0, 256, (pool_size, self.config['simulation_metadata']['chn'], self.config['simulation_metadata']['w'], self.config['simulation_metadata']['h'])).float()
        if self.config['simulation_metadata']['scalar_chn'] != self.config['simulation_metadata']['chn']:
            x[:,-1] = torch.rand(pool_size, self.config['simulation_metadata']['w'], self.config['simulation_metadata']['h'])*np.pi*2.0
        # The rest of your code...
        r, s = self.config['simulation_metadata']['w']//2, seed_size
        x[:,3:self.config['simulation_metadata']['scalar_chn'],r:r+s, r:r+s] = 1.0
        if self.config['simulation_metadata']['angle'] is not None:
            x[:,-1,r:r+s, r:r+s] = self.config['simulation_metadata']['angle']
        x = x.to(self.config['simulation_metadata']['device'])
    
        return x
    
    assign_method(runner, 'seed', seed)
    runner.init()  
    trainer = TrainNca(runner)
    trainer.train()
    wandb.finish()