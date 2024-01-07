import wandb


def set_custom_transition_network_factory(custom_transition_network):
    def set_custom_transition_network(cls):
        class CustomTransition(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.custom_transition_network = custom_transition_network

        return CustomTransition
    return set_custom_transition_network

def set_custom_observation_network_factory(custom_observation_network):
    def set_custom_observation_network(cls):
        class CustomObservation(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.custom_observation_network = custom_observation_network

        return CustomObservation
    return set_custom_observation_network

def set_custom_action_network_factory(custom_action_network):    
    def set_custom_action_network(cls):
        class CustomAction(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.custom_action_network = custom_action_network

        return CustomAction
    return set_custom_action_network

def initialise_wandb(entity, project, name, config):
        wandb.init(
            entity=entity,
            project=project,         
            name=name, 
            config=config
            )  