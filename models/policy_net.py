import torch
from torch import nn
import torch.nn.functional as F

from lib.models_utils import reshape_for_layer_norm
from models.decoders import   att_res_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera

use_bn=False
use_in=True

policy_module_key='policy_net'


class QualityRegressor(nn.Module):
    def __init__(self, in_c2):
        super().__init__()
        self.att_block = att_res_mlp_LN(in_c1=64, in_c2=in_c2, out_c=1).to('cuda')
        self.sig=nn.Sigmoid()
    def forward(self, features,pose_2d ):
        output_2d = self.att_block(features,pose_2d)
        output_2d=self.sig(output_2d)
        return output_2d

class VanillaDecoder(nn.Module):
    def __init__(self,relu_negative_slope=0.):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.LeakyReLU(negative_slope=0.),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.LeakyReLU(negative_slope=0.),
            nn.Linear(16, 4),
        ).to('cuda')
    def forward(self, features ):
        return self.decoder(features)

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        '''Total of 11 input channels'''
        # 3 RGB
        # 1 target object/s mask
        self.rgb_back_bone = res_unet(in_c=4, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')

        self.critic=VanillaDecoder().to('cuda')
        self.actor=VanillaDecoder().to('cuda')

        self.gripper_grasp = QualityRegressor( in_c2=7)
        self.suction_grasp = QualityRegressor( in_c2=3)
        self.shift_affordance=QualityRegressor( in_c2=3+2)

        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)
        self.spatial_encoding=reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

    def forward(self, rgb,gripper_pose,suction_direction,target_mask):
        '''backbone'''
        input_channels=torch.cat([rgb,target_mask],dim=1)
        rgb_features = self.rgb_back_bone(input_channels)
        rgb_features=reshape_for_layer_norm(rgb_features, camera=camera, reverse=False)

        '''Spatial data'''
        if self.spatial_encoding.shape[0] != rgb.shape[0]:
            self.spatial_encoding = depth_xy_spatial_data(batch_size=rgb.shape[0])
            self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        '''gripper grasp head'''
        gripper_pose_detached=gripper_pose.detach().clone()
        gripper_pose_detached[:,-2:]=torch.clip(gripper_pose_detached[:,-2:],0.,1.)
        gripper_pose_detached=reshape_for_layer_norm(gripper_pose_detached, camera=camera, reverse=False)
        griper_grasp_score = self.gripper_grasp(rgb_features, gripper_pose_detached)

        '''suction grasp head'''
        suction_direction_detached = suction_direction.detach().clone()
        suction_direction_detached=reshape_for_layer_norm(suction_direction_detached, camera=camera, reverse=False)
        suction_grasp_score = self.suction_grasp(rgb_features, suction_direction_detached)

        '''shift affordance'''
        shift_query_features=torch.cat([suction_direction_detached,self.spatial_encoding], dim=-1)
        shift_affordance_classifier = self.shift_affordance(rgb_features,shift_query_features )

        '''q value'''
        q_values=self.critic(rgb_features)

        '''policy probabilities'''
        action_logits=self.actor(rgb_features)

        '''reshape'''
        griper_grasp_score = reshape_for_layer_norm(griper_grasp_score, camera=camera, reverse=True)
        suction_grasp_score = reshape_for_layer_norm(suction_grasp_score, camera=camera, reverse=True)
        shift_affordance_classifier = reshape_for_layer_norm(shift_affordance_classifier, camera=camera, reverse=True)
        q_values = reshape_for_layer_norm(q_values, camera=camera, reverse=True)
        action_logits = reshape_for_layer_norm(action_logits, camera=camera, reverse=True)

        '''Categorical policy'''
        policy_reshaped=action_logits.reshape(action_logits.shape[0],-1)
        action_probs=F.softmax(policy_reshaped,dim=-1)
        action_probs=action_probs.reshape(action_logits.shape)

        return griper_grasp_score,suction_grasp_score,shift_affordance_classifier,q_values,action_probs

