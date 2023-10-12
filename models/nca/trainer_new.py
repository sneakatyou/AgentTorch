from trainer_nca import TrainIsoNca
from simulator import NCARunner, configure_nca, get_registry
from AgentTorch.helpers import read_config
import argparse

if __name__ == "__main__":
    
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