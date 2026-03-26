import torch
from mamba_ssm import Mamba
from torch import nn


class Mambas(nn.Module):
    def __init__(self,model,state,conv,expand,k,input_dim,out_dim):
        super(Mambas, self).__init__()
        self.mambas = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=model,  # Model dimension d_model
            d_state=state,  # SSM state expansion factor
            d_conv=conv,  # Local convolution width
            expand=expand,
        ).to('cuda')
        self.flatten1 = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(k*input_dim, out_dim)
        # self.fc1 = nn.Linear(k * 48, out_dim)

    def forward(self, x):
        x = self.mambas(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        return x
