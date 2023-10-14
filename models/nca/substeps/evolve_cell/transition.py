from AgentTorch.substep import SubstepTransition

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from AgentTorch.substep import SubstepTransition

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from substeps.utils import IsoNcaOps

class NCAEvolve(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ops = IsoNcaOps()
        self.CHN = self.config['simulation_metadata']['chn']
        self.ANGLE_CHN = self.config['simulation_metadata']['angle_chn']
        self.SCALAR_CHN = self.CHN-self.ANGLE_CHN
        self.perception = self.ops.get_perception(self.config['simulation_metadata']['model_type'])
        
        # determene the number of perceived channels
        perc_n = self.perception(torch.zeros([1, self.CHN, 8, 8])).shape[1]
        # approximately equalize the param number btw model variants
        hidden_n = 8*1024//(perc_n+self.CHN)
        hidden_n = (hidden_n+31)//32*32
        print('perc_n:', perc_n, 'hidden_n:', hidden_n)
        self.w1 = torch.nn.Conv2d(perc_n, hidden_n, 1)
        self.w1.weight.data.zero_()
        self.w2 = torch.nn.Conv2d(hidden_n, self.CHN, 1, bias=False)
        self.w2.weight.data.zero_()

    def forward(self, state, action, update_rate=0.5):
        x = state['agents']['automata']['cell_state']
        x = x.transpose(1,3)
        alive = action['automata']['AliveMask']
        y = action['automata']['StateVector']
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
        x = x + y*update_mask
        if self.SCALAR_CHN==self.CHN:
            x = x*alive
        else:
            x = torch.cat([x[:,:self.SCALAR_CHN]*alive, x[:,self.SCALAR_CHN:]%(np.pi*2.0)], 1)
        new_state = x.transpose(1,3)
        return {self.output_variables[0]: new_state}