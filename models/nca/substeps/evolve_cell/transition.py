from AgentTorch.substep import SubstepTransition

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from substeps.utils import IsoNcaOps, IsoNcaConfig

class IsoNCAEvolve(SubstepTransition):
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

    def seed(self, n, sz=128, angle=None, seed_size=1):
        x = torch.zeros(n, self.CHN, sz, sz)
        if self.SCALAR_CHN != self.CHN:
            x[:,-1] = torch.rand(n, sz, sz)*np.pi*2.0
        r, s = sz//2, seed_size
        x[:,3:self.SCALAR_CHN,r:r+s, r:r+s] = 1.0
        if angle is not None:
            x[:,-1,r:r+s, r:r+s] = angle
        return x


class NCAEvolve(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config['simulation_metadata']['device'])
        self.channel_n = self.config['simulation_metadata']['n_channels']
        hidden_size = self.config['simulation_metadata']['hidden_size']
        self.fire_rate = self.config['simulation_metadata']['fire_rate']
        self.angle = self.config['simulation_metadata']['angle']

        self.fc0 = nn.Linear(self.channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, self.channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def forward(self, state, action=None):
        x = state['agents']['automata']['cell_state']
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, self.angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>self.fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        new_state = x.transpose(1,3)
        return {self.output_variables[0]: new_state}