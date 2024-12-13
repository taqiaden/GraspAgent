import torch
from torch import nn
from lib.models_utils import reshape_for_layer_norm
from models.decoders import  att_res_mlp_LN2
from models.resunet import res_unet
from registration import camera

use_bn=False
use_in=True

class AbstractQualityRegressor(nn.Module):
    def __init__(self, in_c2, out_c):
        super().__init__()
        self.att_block = att_res_mlp_LN2(in_c1=64, in_c2=in_c2, out_c=out_c).to('cuda')
        self.depth_mlp = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(),
        ).to('cuda')
        self.rgb_mlp = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(),
        ).to('cuda')
    def forward(self, rgb_features,depth_features,pose_2d ):
        rgb_features=self.rgb_mlp(rgb_features)
        depth_features=self.depth_mlp(depth_features)
        features=torch.cat([rgb_features,depth_features],dim=-1)
        output_2d = self.att_block(features,pose_2d)
        return output_2d

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_back_bone = res_unet(in_c=3, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')


        self.gripper_grasp = AbstractQualityRegressor( in_c2=7, out_c=1)
        self.suction_grasp = AbstractQualityRegressor( in_c2=3, out_c=1)

    def forward(self, rgb,depth_features,gripper_pose,suction_direction):
        '''backbone'''
        rgb_features = self.rgb_back_bone(rgb)
        rgb_features=reshape_for_layer_norm(rgb_features, camera=camera, reverse=False)
        depth_features=reshape_for_layer_norm(depth_features, camera=camera, reverse=False)

        '''gripper grasp head'''
        gripper_pose_detached=torch.clip(gripper_pose.detach(),0.,1.)
        gripper_pose_detached=reshape_for_layer_norm(gripper_pose_detached, camera=camera, reverse=False)
        griper_grasp_score = self.gripper_grasp(rgb_features,depth_features, gripper_pose_detached)

        '''suction grasp head'''
        suction_direction_detached=torch.clip(suction_direction.detach(),0.,1.)
        suction_direction_detached=reshape_for_layer_norm(suction_direction_detached, camera=camera, reverse=False)
        suction_grasp_score = self.suction_grasp(rgb_features,depth_features, suction_direction_detached)

        '''reshape'''
        griper_grasp_score = reshape_for_layer_norm(griper_grasp_score, camera=camera, reverse=True)
        suction_grasp_score = reshape_for_layer_norm(suction_grasp_score, camera=camera, reverse=True)


        return griper_grasp_score,suction_grasp_score

