import torch
import numpy as np
import sys
import torch.nn.functional as F
sys.path.insert(0, '/u/ayushc/projects/COLLAB/nca_collab/NCA/AT_gpu/AgentTorch/')
from models.nca.substeps.utils import make_circle_masks

from AgentTorch import Configurator, Runner
from AgentTorch.helpers import read_config

def configure_nca_with_multiple_experiments(config_path,args,exp_no):
    conf = Configurator()

    # add metadata
    conf.add_metadata('exp_no', exp_no)
    conf.add_metadata('num_episodes', args.num_episodes[exp_no])
    conf.add_metadata('num_steps_per_episode', args.num_steps_per_episode[exp_no])
    conf.add_metadata('num_substeps_per_step', args.num_substeps_per_step[exp_no])
    conf.add_metadata('h', args.h[exp_no])
    conf.add_metadata('w', args.w[exp_no])
    conf.add_metadata('n_channels', args.n_channels)[exp_no]
    conf.add_metadata('batch_size', args.batch_size[exp_no])
    conf.add_metadata('device', args.device[exp_no])
    conf.add_metadata('hidden_size', args.hidden_size[exp_no])
    conf.add_metadata('fire_rate', args.fire_rate[exp_no])
    conf.add_metadata('angle', args.angle[exp_no])
    conf.add_metadata('learning_params', args.learning_params[exp_no])
    conf.add_metadata('seed_size', args.seed_size[exp_no])
    
    # create agent
    w, h = conf.get('simulation_metadata.w'), conf.get('simulation_metadata.h')    
    automata_number = h*w
    automata = conf.add_agents(key="automata", number=automata_number)

    # add agent properties
    n_channels = conf.get('simulation_metadata.n_channels')
    batch_size = conf.get('simulation_metadata.batch_size')
    device = conf.get('simulation_metadata.device')
    angle = conf.get('simulation_metadata.angle')
    seed_size = conf.get('simulation_metadata.seed_size')

    from substeps.utils import nca_initialize_state, nca_initialize_state_pool

    arguments_list = [conf.create_variable(key='n_channels', name="n_channels", learnable=False, shape=(1,), initialization_function=None, value=n_channels, dtype="int"),
                    conf.create_variable(key='batch_size', name="batch_size", learnable=False, shape=(1,), initialization_function=None, value=batch_size, dtype="int"),
                    conf.create_variable(key='angle', name="angle", learnable=False, shape=(1,), initialization_function=None, value=angle, dtype="float"),
                    conf.create_variable(key='seed_size', name="seed_size", learnable=False, shape=(1,), initialization_function=None, value=seed_size, dtype="int")
                    ]

    cell_state_initializer = conf.create_initializer(generator = nca_initialize_state, arguments=arguments_list)
    conf.add_property(root='state.agents.automata', key='cell_state', name="cell_state", learnable=True, shape=(n_channels,5184), initialization_function=cell_state_initializer, dtype="float")

    # add environment network
    from AgentTorch.helpers import grid_network
    conf.add_network('evolution_network', grid_network, arguments={'shape': [w, h]})

    # add substep
    # from substeps.evolve_cell.transition import NCAEvolve
    # evolve_transition = conf.create_function(NCAEvolve, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['cell_state'], fn_type="transition")
    
    from substeps.evolve_cell.transition import IsoNCAEvolve
    evolve_transition = conf.create_function(IsoNCAEvolve, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['cell_state'], fn_type="transition")
    
    from substeps.evolve_cell.action import GenerateStateVector, GenerateAliveMask
    generate_state_vector = conf.create_function(GenerateStateVector, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['StateVector'], fn_type="policy")   
    generate_alive_mask = conf.create_function(GenerateAliveMask, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['AliveMask'], fn_type="policy")

    from substeps.evolve_cell.observation import ObserveAliveState, ObserveNeighborsState
    alive_state_observation = conf.create_function(ObserveAliveState, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['AliveState'], fn_type="observation")     
    neighbors_state_observation = conf.create_function(ObserveNeighborsState, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['NeighborsState'], fn_type="observation")
    
    conf.add_substep(name="Evolution", active_agents=["automata"], observation_fn=[alive_state_observation,neighbors_state_observation],policy_fn=[generate_state_vector,generate_alive_mask],transition_fn=[evolve_transition])    
    conf.render(config_path)
    return read_config(config_path), conf.reg

def configure_nca(config_path):
    conf = Configurator()
    # add metadata
    conf.add_metadata('num_episodes', 3)
    conf.add_metadata('num_steps_per_episode', 20)
    conf.add_metadata('num_substeps_per_step', 1)
    conf.add_metadata('h', 72)
    conf.add_metadata('w', 72)
    conf.add_metadata('n_channels', 16)
    conf.add_metadata('batch_size', 8)
    conf.add_metadata('device', 'mps')
    conf.add_metadata('hidden_size', 128)
    conf.add_metadata('fire_rate', 0.5)
    conf.add_metadata('angle', 0.0)
    conf.add_metadata('learning_params', {'lr': 2e-3, 'betas': [0.5, 0.5], 'lr_gamma': 0.9999, 'model_path': 'saved_model.pth'})
    conf.add_metadata('seed_size', 1)
    conf.add_metadata('pool_size', 128)
    conf.add_metadata('scalar_chn', 128)
    conf.add_metadata('chn', 128)
    conf.add_metadata('angle', 128)
    # create agent
    w, h = conf.get('simulation_metadata.w'), conf.get('simulation_metadata.h')    
    automata_number = h*w
    automata = conf.add_agents(key="automata", number=automata_number)

    # add agent properties
    angle = conf.get('simulation_metadata.angle')
    seed_size = conf.get('simulation_metadata.seed_size')
    n_channels = conf.get('simulation_metadata.n_channels')
    batch_size = conf.get('simulation_metadata.batch_size')
    scalar_chn = conf.get('simulation_metadata.scalar_chn')
    chn = conf.get('simulation_metadata.chn')
    angle = conf.get('simulation_metadata.angle')
    device = conf.get('simulation_metadata.device')
    pool_size = conf.get('simulation_metadata.pool_size')
    from substeps.utils import nca_initialize_state,nca_initialize_state_pool

    arguments_list = [conf.create_variable(key='n_channels', name="n_channels", learnable=False, shape=(1,), initialization_function=None, value=n_channels, dtype="int"),
                    conf.create_variable(key='batch_size', name="batch_size", learnable=False, shape=(1,), initialization_function=None, value=batch_size, dtype="int"),
                    conf.create_variable(key='angle', name="angle", learnable=False, shape=(1,), initialization_function=None, value=angle, dtype="float"),
                    conf.create_variable(key='seed_size', name="seed_size", learnable=False, shape=(1,), initialization_function=None, value=seed_size, dtype="int"),
                    conf.create_variable(key='pool_size', name="pool_size", learnable=False, shape=(1,), initialization_function=None, value=128, dtype="int"),
                    conf.create_variable(key='scalar_chn', name="scalar_chn", learnable=False, shape=(1,), initialization_function=None, value=scalar_chn, dtype="int"),
                    conf.create_variable(key='chn', name="chn", learnable=False, shape=(1,), initialization_function=None, value=chn, dtype="int"), 
                    conf.create_variable(key='device', name="device", learnable=False, shape=(1,), initialization_function=None, value=device, dtype="str") 
                    ]
    # cell_state_initializer = conf.create_initializer(generator = nca_initialize_state, arguments=arguments_list)
    cell_state_initializer = conf.create_initializer(generator = nca_initialize_state_pool, arguments=arguments_list)
    conf.add_property(root='state.agents.automata', key='cell_state', name="cell_state", learnable=True, shape=(n_channels,5184), initialization_function=cell_state_initializer, dtype="float")

    # add environment network
    from AgentTorch.helpers import grid_network
    conf.add_network('evolution_network', grid_network, arguments={'shape': [w, h]})

    # add substep
    # from substeps.evolve_cell.transition import NCAEvolve
    # evolve_transition = conf.create_function(NCAEvolve, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['cell_state'], fn_type="transition")
    
    from substeps.evolve_cell.transition import IsoNCAEvolve
    evolve_transition = conf.create_function(IsoNCAEvolve, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['cell_state'], fn_type="transition")
    
    from substeps.evolve_cell.action import GenerateStateVector, GenerateAliveMask
    generate_state_vector = conf.create_function(GenerateStateVector, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['StateVector'], fn_type="policy")
    # conf.add_substep(name="Evolution", active_agents=["automata"], policy_fn=[generate_state_vector])
    
    generate_alive_mask = conf.create_function(GenerateAliveMask, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['AliveMask'], fn_type="policy")
    # conf.add_substep(name="Evolution", active_agents=["automata"], policy_fn=[generate_alive_mask])

    from substeps.evolve_cell.observation import ObserveAliveState, ObserveNeighborsState
    alive_state_observation = conf.create_function(ObserveAliveState, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['AliveState'], fn_type="observation") 
    # conf.add_substep(name="Evolution", active_agents=["automata"], observation_fn=[alive_state_observation])
    
    neighbors_state_observation = conf.create_function(ObserveNeighborsState, input_variables={'cell_state':'agents/automata/cell_state'}, output_variables=['NeighborsState'], fn_type="observation")
    # conf.add_substep(name="Evolution", active_agents=["automata"], observation_fn=[neighbors_state_observation])
    conf.add_substep(name="Evolution", active_agents=["automata"], observation_fn=[alive_state_observation,neighbors_state_observation],policy_fn=[generate_state_vector,generate_alive_mask],transition_fn=[evolve_transition])
    
    conf.render(config_path)

    return read_config(config_path), conf.reg

class NCARunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _nca_initialize_state(self, shape, params):
        device = torch.device(params['device'])
        batch_size = params['batch_size']
        n_channels = int(params['n_channels'].item())
        processed_shape = shape #[process_shape_omega(s) for s in shape]
        grid_shape = [np.sqrt(processed_shape[0]).astype(int), np.sqrt(processed_shape[0]).astype(int), processed_shape[1]]
        seed_x = np.zeros(grid_shape, np.float32)
        seed_x[grid_shape[0]//2, grid_shape[1]//2, 3:] = 1.0
        x0 = np.repeat(seed_x[None, ...], batch_size, 0)
        x0 = torch.from_numpy(x0.astype(np.float32)).to(device)
        return x0
    

    def reset(self,seed_size=1):
        x0 = self._nca_initialize_state(seed_size)
        self.state = self.initializer.state
        self.state['agents']['automata']['cell_state'] = x0

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
        
