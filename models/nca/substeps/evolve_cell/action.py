from AgentTorch.substep import SubstepAction
import torch
class GenerateAliveMask(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, observation):
        
        observation_grid = observation['AliveState']
        threshold = torch.tensor(self.config['simulation_metadata']['alive_threshold_value'])
        alive_mask = torch.gt(observation_grid, threshold)
        # alive_mask = observation_grid > 0.1 #convert 0.1 to config variable
        
        return {self.output_variables[0] : alive_mask}


class GenerateStateVector(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, observation):
        
        observation_grid = observation['NeighborsState']
        state_vector = observation_grid 
        
        return {self.output_variables[0] : state_vector}
        