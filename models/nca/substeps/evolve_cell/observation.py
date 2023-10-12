from AgentTorch.substep import SubstepObservation
import torch.nn.functional as F
from substeps.utils import IsoNcaOps, IsoNcaConfig

class ObserveAliveState(SubstepObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state):
        x = state['agents']['automata']['cell_state']
        x = x.transpose(1,3)
        observation_grid = F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1)       
        return {self.output_variables[0] : observation_grid}

class ObserveNeighborsState(SubstepObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ops = IsoNcaOps()
        self.perception = ops.get_perception(self.config['simulation_metadata']['model_type'])
    
    def forward(self, state):       
        x = state['agents']['automata']['cell_state']
        x = x.transpose(1,3)
        observed_neigbors_state = self.perception(x)     
        return {self.output_variables[0] : observed_neigbors_state}
