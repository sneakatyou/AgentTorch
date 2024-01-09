import sys

sys.path.insert(0, '/Users/shashankkumar/Documents/AgentTorchLatest/AgentTorch')
import numpy as np
import torch
import torch.nn.functional as F
from models.nca.utils_simulator import add_agent_properties, add_configuration, add_environment_network, add_substep, create_variables, get_config_values, set_custom_transition_network_factory
from models.nca.substeps.utils import make_circle_masks
from AgentTorch import Configurator, Runner
from AgentTorch.helpers import read_config

def configure_nca(config_path, params, custom_transition_network=None,custom_observation_network=None,custom_action_network=None):
    conf = set_config(params)   
    #define active agent
    active_agent = "automata" 
    
    #add transition
    perc_n = 32
    hidden_n = 192
    custom_transition_network = torch.nn.Sequential(
            torch.nn.Conv2d(perc_n, hidden_n, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_n, 16, 1, bias=False)
        )
    custom_transition_network[0].weight.data.zero_()
    from substeps.evolve_cell.transition import IsoNCAEvolve
    if custom_transition_network is None:
        evolve_transition = conf.create_function(IsoNCAEvolve, input_variables={'cell_state': f'agents/{active_agent}/cell_state'}, output_variables=['cell_state'], fn_type="transition")
    else:
        @set_custom_transition_network_factory(custom_transition_network)
        class CustomNCAEvolve(IsoNCAEvolve):
            pass
        
        evolve_transition = conf.create_function(CustomNCAEvolve, input_variables={'cell_state': f'agents/{active_agent}/cell_state'}, output_variables=['cell_state'], fn_type="transition")
    
    #add policy
    from substeps.evolve_cell.action import GenerateStateVector
    generate_state_vector =  conf.create_function(GenerateStateVector, input_variables={'cell_state': f'agents/{active_agent}/cell_state'}, output_variables=['StateVector'], fn_type="policy")

    from substeps.evolve_cell.action import GenerateAliveMask
    generate_alive_mask = conf.create_function(GenerateAliveMask, input_variables={'cell_state': f'agents/{active_agent}/cell_state'}, output_variables=['AliveMask'], fn_type="policy")
    
    #add observation
    from substeps.evolve_cell.observation import ObserveAliveState
    alive_state_observation = conf.create_function(ObserveAliveState, input_variables={'cell_state': f'agents/{active_agent}/cell_state'}, output_variables=['AliveState'], fn_type="observation")
    
    from substeps.evolve_cell.observation import ObserveNeighborsState
    neighbors_state_observation = conf.create_function(ObserveNeighborsState, input_variables={'cell_state': f'agents/{active_agent}/cell_state'}, output_variables=['NeighborsState'], fn_type="observation")
    
    # add substep
    add_substep(conf, active_agents=[active_agent], observation_fn={alive_state_observation, neighbors_state_observation}, policy_fn={generate_state_vector, generate_alive_mask}, transition_fn={evolve_transition})
    
    conf.render(config_path)
    
    return read_config(config_path), conf.reg

def set_config(params):
    conf = Configurator()
    
    #add config metadata
    add_configuration(conf, params)
    
    #retrieve config values
    config_values = get_config_values(conf, ['angle', 'seed_size', 'n_channels', 'batch_size', 'scalar_chn', 'chn', 'device', 'w', 'pool_size','h'])   
    automata_number = config_values['w'] * config_values['h']
    conf.add_agents(key="automata", number=automata_number)
    
    #add agent and network
    add_agent_properties(conf, config_values, automata_number)
    add_environment_network(conf, config_values)
    
    
    return conf


class NCARunnerWithPool(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _nca_initialize_state(self,seed_size,i,len_loss):
        if (self.pool is None):            
            x = self.seed(self.config['simulation_metadata']['pool_size'],seed_size)
            self.pool = x
            
        
        self.batch_idx = np.random.choice(self.config['simulation_metadata']['pool_size'], self.config['simulation_metadata']['batch_size'], replace=False)
        x0 = self.pool[self.batch_idx] 
        
        x0 = self.augment_input(i, len_loss, x0)
        return x0

    def augment_input(self, i, len_loss, x0):
        if len_loss < 4000:
                    seed_rate = 1
        else:
            # exp because of decrease of step_n
            #seed_rate = 3
            seed_rate = 6
        if i%seed_rate==0:
            x0[:1] = self.seed(1,1)
        #damage_rate = 3 # for spiderweb and heart
        damage_rate = 6  # for lizard?
        device = self.config['simulation_metadata']['device']
        if i%damage_rate==0:
            mask = torch.from_numpy(make_circle_masks(1, self.config['simulation_metadata']['w'], self.config['simulation_metadata']['w'])[:,None]).to(device)
            if self.config['simulation_metadata']['hex_grid']:
                mask = F.grid_sample(mask, self.xy_grid[None,:].repeat([len(mask), 1, 1, 1]), mode='bicubic').to(device)
            x0[-1:] *= (1.0 - mask)
        return x0

    def seed(self, pool_size, seed_size):
        x = torch.zeros(pool_size, self.config['simulation_metadata']['chn'], self.config['simulation_metadata']['w'], self.config['simulation_metadata']['h'])
        if self.config['simulation_metadata']['scalar_chn'] != self.config['simulation_metadata']['chn']:
            x[:,-1] = torch.rand(pool_size, self.config['simulation_metadata']['w'], self.config['simulation_metadata']['h'])*np.pi*2.0
        r, s = self.config['simulation_metadata']['w']//2, seed_size
        x[:,3:self.config['simulation_metadata']['scalar_chn'],r:r+s, r:r+s] = 1.0
        if self.config['simulation_metadata']['angle'] is not None:
            x[:,-1,r:r+s, r:r+s] = self.config['simulation_metadata']['angle']
        x = x.to(self.config['simulation_metadata']['device'])
        
        return x

    def reset(self,seed_size=1,i=0,len_loss=0):
        x0 = self._nca_initialize_state(seed_size,i,len_loss)
        self.state = self.initializer.state
        self.state['agents']['automata']['cell_state'] = x0

    def update_pool(self,x):
        self.pool[self.batch_idx] = x
        
