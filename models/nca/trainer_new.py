from trainer_nca import TrainIsoNca
from simulator import NCARunner, get_registry
from AgentTorch.helpers import read_config
import argparse

if __name__ == "__main__":
    try:
        # Parsing command line arguments
        parser = argparse.ArgumentParser(
            description="AgentTorch: design, simulate and optimize agent-based models"
        )
        parser.add_argument(
            "-c", "--config", help="Name of the yaml config file with the parameters."
        )
        # *************************************************************************
        args = parser.parse_args()
        config_file = args.config
    
    except:
        config_file = "/Users/shashankkumar/Documents/GitHub/NCA/AgentTorch/models/nca/config.yaml"
    
    config = read_config(config_file)
    registry = get_registry()
    runner = NCARunner(config, registry)
    runner.init()
    trainer = TrainIsoNca(config, runner)
    trainer.train()