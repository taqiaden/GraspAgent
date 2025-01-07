import torch
from torch import nn

from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN2, res_block_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera

use_bn=False
use_in=True

value_module_key='policy_net'


class RGBDRepresentation(nn.Module):
    def __init__(self,relu_negative_slope=0.):
        super().__init__()
        self.depth_mlp = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope>0. else nn.ReLU(),
        ).to('cuda')
        self.rgb_mlp = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope>0. else nn.ReLU(),
        ).to('cuda')
    def forward(self, rgb_features,depth_features ):
        rgb_features=self.rgb_mlp(rgb_features)
        depth_features=self.depth_mlp(depth_features)
        features=torch.cat([rgb_features,depth_features],dim=-1)
        return features

class AbstractQualityRegressor(nn.Module):
    def __init__(self, in_c2, out_c):
        super().__init__()
        self.att_block = att_res_mlp_LN2(in_c1=64, in_c2=in_c2, out_c=out_c).to('cuda')
        self.sig=nn.Sigmoid()
    def forward(self, features,pose_2d ):
        output_2d = self.att_block(features,pose_2d)
        output_2d=self.sig(output_2d)
        return output_2d

class QValue(nn.Module):
    def __init__(self,relu_negative_slope=0.):
        super().__init__()
        self.res = res_block_mlp_LN(in_c=64, medium_c=32, out_c=16).to('cuda')
        self.decoder = nn.Sequential(
            nn.LayerNorm(16),
            nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope>0. else nn.ReLU(),
            nn.Linear(16, 4),
        ).to('cuda')
    def forward(self, features ):
        output_2d = self.res(features)
        output_2d=self.decoder(output_2d)
        return output_2d

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_back_bone = res_unet(in_c=4, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')
        self.rgb_d_unified_representation=RGBDRepresentation()

        self.q_value=QValue().to('cuda')

        self.gripper_grasp = AbstractQualityRegressor( in_c2=7, out_c=1)
        self.suction_grasp = AbstractQualityRegressor( in_c2=3, out_c=1)
        self.shift_affordance=AbstractQualityRegressor( in_c2=3+2, out_c=1)

        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)
        self.spatial_encoding=reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

    def forward(self, rgb,depth_features,gripper_pose,suction_direction,target_object_mask):
        '''backbone'''
        rgb_mask=torch.cat([rgb,target_object_mask],dim=1)
        rgb_features = self.rgb_back_bone(rgb_mask)
        rgb_features=reshape_for_layer_norm(rgb_features, camera=camera, reverse=False)
        depth_features=reshape_for_layer_norm(depth_features, camera=camera, reverse=False)

        '''Spatial data'''
        if self.spatial_encoding.shape[0] != rgb.shape[0]:
            self.spatial_encoding = depth_xy_spatial_data(batch_size=rgb.shape[0])
            self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        '''RGB D features'''
        rgb_d_features=self.rgb_d_unified_representation(rgb_features,depth_features)

        '''gripper grasp head'''
        gripper_pose_detached=gripper_pose.detach().clone()
        gripper_pose_detached[:,-2:]=torch.clip(gripper_pose_detached[:,-2:],0.,1.)
        gripper_pose_detached=reshape_for_layer_norm(gripper_pose_detached, camera=camera, reverse=False)
        griper_grasp_score = self.gripper_grasp(rgb_d_features, gripper_pose_detached)

        '''suction grasp head'''
        suction_direction_detached = suction_direction.detach().clone()
        suction_direction_detached=reshape_for_layer_norm(suction_direction_detached, camera=camera, reverse=False)
        suction_grasp_score = self.suction_grasp(rgb_d_features, suction_direction_detached)

        '''shift affordance'''
        shift_query_features=torch.cat([suction_direction_detached,self.spatial_encoding], dim=-1)
        shift_affordance_classifier = self.shift_affordance(rgb_d_features,shift_query_features )

        '''q value'''
        q_value=self.q_value(rgb_features)

        '''reshape'''
        griper_grasp_score = reshape_for_layer_norm(griper_grasp_score, camera=camera, reverse=True)
        suction_grasp_score = reshape_for_layer_norm(suction_grasp_score, camera=camera, reverse=True)
        shift_affordance_classifier = reshape_for_layer_norm(shift_affordance_classifier, camera=camera, reverse=True)
        q_value = reshape_for_layer_norm(q_value, camera=camera, reverse=True)

        return griper_grasp_score,suction_grasp_score,shift_affordance_classifier,q_value

