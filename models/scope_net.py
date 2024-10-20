import torch
from torch import nn


gripper_scope_model_state_path=r'gripper_scope_model_state'
suction_scope_model_state_path=r'suction_scope_model_state'

mean_=torch.tensor([0.43,0.,0.11,0.,0.,0.62]).to('cuda')
std_=torch.tensor([0.09,0.12,0.05,0.5,0.5,0.3]).to('cuda')

def scope_standardization(data):
    s_data=data-mean_
    s_data=s_data/std_
    return s_data

class scope_net_vanilla(nn.Module):
    def __init__(self,in_size):
        super().__init__()

        self.encoder=nn.Sequential(
            nn.Linear(in_size, 64, bias=False),
            nn.LayerNorm(64),
            nn.SiLU(True),
            nn.Linear(64, 128,bias=False),
            nn.LayerNorm(128),
            nn.SiLU(True),
            nn.Linear(128, 64,bias=False),
        ).to('cuda')

        self.res=nn.Linear(in_size, 64,bias=False).to('cuda')

        self.decoder = nn.Sequential(
            nn.LayerNorm(64),
            nn.SiLU(True),
            nn.Linear(64, 16, bias=False),
            nn.LayerNorm(16),
            nn.SiLU(True),
            nn.Linear(16, 1),
        ).to('cuda')

    def forward(self, data ):
        s_data=scope_standardization(data)# scope includes the x,y,z point plus the approach direction
        x=self.encoder(s_data)
        res=self.res(s_data)
        x=x+res
        output = self.decoder(x)
        return output

