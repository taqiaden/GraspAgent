import torch
from torch import nn
import torch.nn.functional as F

from models.decoders import res_block_mlp_LN

gripper_scope_model_state_path=r'gripper_scope_model_state'
suction_scope_model_state_path=r'suction_scope_model_state'

class scope_net_vanilla(nn.Module):
    def __init__(self,in_size):
        super().__init__()

        # self.net = nn.Sequential(
        #     nn.Linear(in_size, 32, bias=False),
        #     nn.LayerNorm(32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 16,bias=False),
        #     nn.LayerNorm(16),
        #     nn.ReLU(True),
        #     nn.Linear(16, 1),
        # ).to('cuda')
        self.block1=res_block_mlp_LN(in_size, 32, 16).to('cuda')
        self.block2=res_block_mlp_LN(in_size, 32, 16).to('cuda')
        self.block3=res_block_mlp_LN(in_size, 32, 16).to('cuda')

        self.d = nn.Sequential(
            nn.Linear(48, 16, bias=False),
            nn.LayerNorm(16),
            nn.ReLU(True),
            nn.Linear(16, 1),
        ).to('cuda')


    def forward(self, pose ): # scope includes the x,y,z point plus the approach direction
        b1=self.block1(pose)
        b2=self.block2(pose)
        b3=self.block3(pose)
        features=torch.cat([b1,b2,b3],dim=-1)
        output = self.d(features)
        return output

class scope_net_attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Sequential(
            nn.Linear(3, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 64),
        ).to('cuda')
        self.query = nn.Sequential(
            nn.Linear(9, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 64),
        ).to('cuda')
        self.value = nn.Sequential(
            nn.Linear(3, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 64),
        ).to('cuda')
        self.decoder = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 16,bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to('cuda')

    def forward(self, rotation,transition ): # scope includes the x,y,z point plus the approach direction
        key=self.key(transition)
        query=self.query(rotation)
        value=self.value(transition)

        attention_map=key*query

        attention_map=F.softmax(attention_map,dim=-1)
        attention_score=attention_map*value

        output = self.decoder(attention_score)
        return output