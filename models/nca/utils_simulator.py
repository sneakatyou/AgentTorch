def create_variables(conf, params):
    arguments_list = []
    for key, value in params.items():
        dtype = type(value).__name__
        variable = conf.create_variable(key=key, name=key, learnable=False, shape=(1,), initialization_function=None, value=value, dtype=dtype)
        arguments_list.append(variable)
    return arguments_list


def add_environment_network(conf, config_values):
    from AgentTorch.helpers import grid_network
    conf.add_network('evolution_network', grid_network, arguments={'shape': [config_values['w'], config_values['h']]})

def add_substep(conf, active_agents ,observation_fn, policy_fn, transition_fn):
    conf.add_substep(name="Evolution", active_agents=active_agents, observation_fn=observation_fn, policy_fn=policy_fn, transition_fn=transition_fn)

def add_agent_properties(conf, config_values, automata_number):
    from substeps.utils import nca_initialize_state_pool
    params = {
        'n_channels': config_values['n_channels'],
        'batch_size': config_values['batch_size'],
        'angle': config_values['angle'],
        'seed_size': config_values['seed_size'],
        'pool_size': config_values['pool_size'],
        'scalar_chn': config_values['scalar_chn'],
        'chn': config_values['chn'],
        'device': config_values['device'],
        'w': config_values['w']
    }
    arguments_list = create_variables(conf, params)
    cell_state_initializer = conf.create_initializer(generator=nca_initialize_state_pool, arguments=arguments_list)
    conf.add_property(root='state.agents.automata', key='cell_state', name="cell_state", learnable=True, shape=(config_values['n_channels'], automata_number), initialization_function=cell_state_initializer, dtype="float")
