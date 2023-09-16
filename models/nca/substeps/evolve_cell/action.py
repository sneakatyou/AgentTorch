from AgentTorch.substep import SubstepAction

class GenerateAliveMask(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, observation):
        try:
            observation_grid = observation['AliveState']
            alive_mask = observation_grid > 0.1 #convert 0.1 to config variable
        except Exception as e:
            print(e)
            print(observation)
            raise e
        return {self.output_variables[0] : alive_mask}


class GenerateStateVector(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, observation):
        try:
            observation_grid = observation['NeighborsState']
            state_vector = observation_grid #need not do anything right now
        except Exception as e:
            print(e)
            print(observation)
            raise e
        return {self.output_variables[0] : state_vector}
        