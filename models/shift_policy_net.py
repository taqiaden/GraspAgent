import torch
from torch import nn
import torch.nn.functional as F
from lib.models_utils import reshape_for_layer_norm
from models.decoders import   att_res_mlp_LN
from models.resunet import res_unet
from registration import camera, standardize_depth
from visualiztion import view_features

use_bn=False
use_in=True

shift_policy_module_key='shift_policy_net'

class VanillaDecoder(nn.Module):
    def __init__(self,relu_slope=0.):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.LeakyReLU(negative_slope=relu_slope) if relu_slope > 0. else nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.LeakyReLU(negative_slope=relu_slope) if relu_slope > 0. else nn.ReLU(),
            nn.Linear(16, 1),
        ).to('cuda')
    def forward(self, features ):
        return self.decoder(features)

class ShiftPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        '''Total of 11 input channels'''
        # 3 RGB
        # 1 target object/s mask
        self.rgb_back_bone = res_unet(in_c=5, Batch_norm=use_bn, Instance_norm=use_in,relu_negative_slope=0.1).to('cuda')
        self.pre_IN=nn.InstanceNorm2d(5).to('cuda')

        '''clear policy'''
        self.critic=VanillaDecoder(relu_slope=0.1).to('cuda')
        self.actor=VanillaDecoder(relu_slope=0.0).to('cuda')


    def dilated_mask(self,mask,kernel_size=57):
        kernel=torch.ones((kernel_size,kernel_size),dtype=torch.float32,device=mask.device)
        corner_size=int((kernel_size-3)/2)
        kernel[:corner_size,:corner_size]=0.
        kernel[:corner_size,-corner_size:]=0.
        kernel[-corner_size:,:corner_size]=0.
        kernel[-corner_size:,-corner_size:]=0.

        kernel=kernel.view(1,1,kernel_size,kernel_size)
        thickened_mask=F.conv2d(mask,kernel,padding=int((kernel_size-1)/2)).clamp(0,1)>0.5

        return thickened_mask

    def forward(self, rgb,depth,target_mask,shift_mask):
        '''modalities preprocessing'''
        depth = standardize_depth(depth)

        '''action mask'''
        dilated_target_mask=self.dilated_mask(target_mask,kernel_size=71)

        actions_mask=dilated_target_mask & shift_mask

        '''concatenate and decode'''
        input_channels=torch.cat([rgb,depth,target_mask],dim=1)
        input_channels=self.pre_IN(input_channels)
        rgb_features = self.rgb_back_bone(input_channels)
        rgb_features=reshape_for_layer_norm(rgb_features, camera=camera, reverse=False)

        '''q value'''
        q_values=self.critic(rgb_features)

        '''policy probabilities'''
        action_logits=self.actor(rgb_features)

        '''reshape'''
        q_values = reshape_for_layer_norm(q_values, camera=camera, reverse=True)
        action_logits = reshape_for_layer_norm(action_logits, camera=camera, reverse=True)


        '''Categorical policy'''
        action_logits=action_logits.view(-1,1,480*712) # [b,1,w,h]
        actions_mask = actions_mask.view(-1, 1, 480 * 712)

        action_logits[actions_mask]=F.softmax(action_logits[actions_mask],dim=-1)
        action_logits[~actions_mask]*=0.

        action_probs=action_logits.view(-1,1,480,712)

        return q_values,action_probs

