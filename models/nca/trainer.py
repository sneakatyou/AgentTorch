import argparse
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import imageio

from simulator import NCARunner, configure_nca
from AgentTorch.helpers import read_config
from substeps.utils import AddAuxilaryChannel, InvariantLoss, IsoNcaOps, IsoNcaConfig, make_circle_masks
import torcheck
# *************************************************************************
# Parsing command line arguments
parser = argparse.ArgumentParser(
    description="AgentTorch: design, simulate and optimize agent-based models"
)
parser.add_argument(
    "-c", "--config", help="Name of the yaml config file with the parameters."
)
# # *************************************************************************
args = parser.parse_args()
config_path = args.config

config, registry = configure_nca('/Users/shashankkumar/Documents/AgentTorch-original/AgentTorch/models/nca/config.yaml')

runner = NCARunner(read_config('/Users/shashankkumar/Documents/AgentTorch-original/AgentTorch/models/nca/config_nca.yaml'), registry)
runner.init()

device = torch.device(runner.config['simulation_metadata']['device'])

# *************************************************************************
# Generating target
def load_emoji(index, path="/Users/shashankkumar/Documents/AgentTorch-original/AgentTorch/models/nca/data/emoji.png"):
    im = imageio.imread(path)
    emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
    emoji /= 255.0
    return emoji
TARGET_EMOJI = 0
TARGET_PADDING = 16
target_img = load_emoji(TARGET_EMOJI)
p = TARGET_PADDING
pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)

# *************************************************************************

optimizer = optim.Adam(runner.parameters(), 
                lr=runner.config['simulation_metadata']['learning_params']['lr'], 
                betas=runner.config['simulation_metadata']['learning_params']['betas'])
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 
                runner.config['simulation_metadata']['learning_params']['lr_gamma'])
loss_log = []

num_steps_per_episode = runner.config["simulation_metadata"]["num_steps_per_episode"]
# try:
#     torcheck.register(optimizer)
#     torcheck.add_module_changing_check(runner, module_name="runner")
#     # torcheck.add_module_unchanging_check(runner.initializer, module_name="runner_initializer")
#     torcheck.add_module_nan_check(runner)
#     torcheck.add_module_inf_check(runner)
#     torcheck.verbose_on()
# except torcheck.TorcheckError as e:
#     print(e)
ops = IsoNcaOps()
for ix in range(runner.config['simulation_metadata']['num_episodes']):
    runner.reset()
    optimizer.zero_grad()
    runner.step(num_steps_per_episode)
    output = runner.state_trajectory[-1][-1]
    x = output['agents']['automata']['cell_state']
    
    loss = F.mse_loss(x[:, :, :, :4], pad_target)
    try:
        loss.backward()
    except:
        import ipdb; ipdb.set_trace()
    optimizer.step()
    scheduler.step()
    # x = x.permute([0,3,1,2])
    # imgs = ops.to_rgb(x.detach().cpu())
    # imgs = imgs.permute([0, 2, 3, 1]).cpu()
    # cv2.imshow(ops.zoom(
    #     ops.tile2d(imgs, 4), 2))
    loss_log.append(loss.item())
    print(f"Episode {ix} | Loss: {loss.item()}")

torch.save(runner.state_dict(), runner.config['simulation_metadata']['learning_params']['model_path'])

print("Execution complete")
