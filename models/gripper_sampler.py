import torch
import torch.nn.functional as F
from torch import nn
from models.decoders import decoder2
from models.resunet import res_unet

gripper_generator_model_state_path=r'gripper_generator_model_state'
def gripper_output_normalization(output):
    '''approach normalization'''
    approach=output[:, 0:3, :]
    approach=F.normalize(approach, dim=1)

    '''beta normalization'''
    beta=output[:, 3:5, :]
    beta=F.normalize(beta, dim=1)

    dist=output[:, 5:6, :]
    width=output[:, 6:7, :]
    normalized_output=torch.cat([approach,beta,dist,width],dim=1)
    return normalized_output

class gripper_sampler_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1,Batch_norm=False,Instance_norm=True).to('cuda')
        self.decoder=decoder2(in_c=64,out_c=7,Batch_norm=False,Instance_norm=True).to('cuda')
    def forward(self, depth ):
        features=self.back_bone(depth)
        output=self.decoder(features)
        output=gripper_output_normalization(output)
        return output
