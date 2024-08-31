from torch import nn

from models.decoders import decoder2
from models.resunet import res_unet, batch_norm_relu
import torch.nn.functional as F

from registration import standardize_depth

suction_sampler_model_state_path=r'suction_sampler_model_state'


class suction_sampler_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1,Batch_norm=True,Instance_norm=False).to('cuda')
        self.decoder=decoder2(in_c=64,out_c=3,Batch_norm=True,Instance_norm=False).to('cuda')
    def forward(self, depth ):
        depth = standardize_depth(depth)
        features=self.back_bone(depth)
        output=self.decoder(features)
        output=F.normalize(output,dim=1)
        return output