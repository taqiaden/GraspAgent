import torch
from torch import nn
import torch.nn.functional as F
from lib.models_utils import reshape_for_layer_norm
from models.decoders import   att_res_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth
from visualiztion import view_features

use_bn=False
use_in=True

policy_module_key='policy_net'

relu_slope=0.0

class QualityRegressor(nn.Module):
    def __init__(self, in_c2,out_c=1,drop_out_ratio=0.):
        super().__init__()

        self.att_block = att_res_mlp_LN(in_c1=64, in_c2=in_c2, out_c=out_c,relu_negative_slope=relu_slope,shallow_decoder=True,drop_out_ratio=drop_out_ratio).to('cuda')
        # self.sig=nn.Sigmoid()
    def forward(self, features,pose_2d ):
        output_2d = self.att_block(features,pose_2d)
        # output_2d=self.sig(output_2d)
        return output_2d

class VanillaDecoder(nn.Module):
    def __init__(self,relu_negative_slope=0.):
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

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        '''Total of 11 input channels'''
        # 3 RGB
        # 1 target object/s mask
        self.rgb_back_bone = res_unet(in_c=5, Batch_norm=use_bn, Instance_norm=use_in,relu_negative_slope=relu_slope).to('cuda')
        self.pre_IN=nn.InstanceNorm2d(5).to('cuda')

        '''clear policy'''
        self.critic=VanillaDecoder().to('cuda')
        self.actor=VanillaDecoder().to('cuda')

        '''seize policy'''
        self.gripper_grasp = QualityRegressor( in_c2=7,drop_out_ratio=0.5)
        self.suction_grasp = QualityRegressor( in_c2=3,drop_out_ratio=0.5)

        '''handover policy'''
        self.handover_policy = QualityRegressor( in_c2=10,out_c=2,drop_out_ratio=.5)

        # self.spatial_encoding = depth_xy_spatial_data(batch_size=1)
        # self.spatial_encoding=reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

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

    def forward(self, rgb,depth,gripper_pose,suction_direction,target_mask,shift_mask):
        '''modalities preprocessing'''
        depth = standardize_depth(depth)

        '''action mask'''
        dilated_target_mask=self.dilated_mask(target_mask,kernel_size=71)

        # dilated_target_mask2=reshape_for_layer_norm(dilated_target_mask, camera=camera, reverse=False)
        # shift_mask2=reshape_for_layer_norm(shift_mask, camera=camera, reverse=False)
        # view_features(dilated_target_mask2,hide_axis=True)
        # view_features(shift_mask2,hide_axis=True)
        # view_features(dilated_target_mask2&shift_mask2,hide_axis=True)

        actions_mask=dilated_target_mask & shift_mask

        '''concatenate and decode'''
        input_channels=torch.cat([rgb,depth,target_mask],dim=1)
        input_channels=self.pre_IN(input_channels)
        rgb_features = self.rgb_back_bone(input_channels)
        rgb_features=reshape_for_layer_norm(rgb_features, camera=camera, reverse=False)


        '''Spatial data'''
        # if self.spatial_encoding.shape[0] != rgb.shape[0]:
        #     self.spatial_encoding = depth_xy_spatial_data(batch_size=rgb.shape[0])
        #     self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        '''gripper grasp head'''
        gripper_pose_detached=gripper_pose.detach().clone()
        gripper_pose_detached=reshape_for_layer_norm(gripper_pose_detached, camera=camera, reverse=False)
        griper_grasp_score = self.gripper_grasp(rgb_features, gripper_pose_detached)

        '''suction grasp head'''
        suction_direction_detached = suction_direction.detach().clone()
        suction_direction_detached=reshape_for_layer_norm(suction_direction_detached, camera=camera, reverse=False)
        suction_grasp_score = self.suction_grasp(rgb_features, suction_direction_detached)

        '''handover quality'''
        handover_score=self.handover_policy(rgb_features,torch.cat([gripper_pose_detached,suction_direction_detached],dim=-1))

        '''q value'''
        q_values=self.critic(rgb_features)

        '''policy probabilities'''
        action_logits=self.actor(rgb_features)

        '''reshape'''
        griper_grasp_score = reshape_for_layer_norm(griper_grasp_score, camera=camera, reverse=True)
        suction_grasp_score = reshape_for_layer_norm(suction_grasp_score, camera=camera, reverse=True)
        handover_score = reshape_for_layer_norm(handover_score, camera=camera, reverse=True)
        q_values = reshape_for_layer_norm(q_values, camera=camera, reverse=True)
        action_logits = reshape_for_layer_norm(action_logits, camera=camera, reverse=True)

        # depth_features=reshape_for_layer_norm(depth_features, camera=camera, reverse=True)

        '''Categorical policy'''
        action_logits=action_logits.view(-1,1,480*712) # [b,1,w,h]
        actions_mask = actions_mask.view(-1, 1, 480 * 712)

        action_logits[actions_mask]=F.softmax(action_logits[actions_mask],dim=-1)
        action_logits[~actions_mask]*=0.

        action_probs=action_logits.view(-1,1,480,712)

        return griper_grasp_score,suction_grasp_score,q_values,action_probs,handover_score

