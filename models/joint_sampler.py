import torch
from torch import nn
import torch.nn.functional as F
from lib.custom_activations import GripperGraspRegressor
from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import SpatialEncoder
from registration import camera, standardize_depth

use_bn=False
use_in=True

class GripperPartSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoding = SpatialEncoder()
        self.get_approach=att_res_mlp_LN(in_c1=64, in_c2=2, out_c=3).to('cuda')
        self.get_beta_dist=att_res_mlp_LN(in_c1=64, in_c2=5, out_c=3).to('cuda')
        self.get_width=att_res_mlp_LN(in_c1=64, in_c2=8, out_c=1).to('cuda')

        self.gripper_regressor_layer=GripperGraspRegressor()

    def forward(self,representation_2d, sampled_approach=None):
        '''spatial data'''
        spatial_data_2d = self.spatial_encoding(representation_2d.shape[0])

        '''approach'''
        approach=self.get_approach(representation_2d,spatial_data_2d) if sampled_approach is None else sampled_approach

        '''beta & distance'''
        params2=torch.cat([approach,spatial_data_2d],dim=1)
        beta_dist=self.get_beta_dist(representation_2d,params2)

        '''width'''
        params3=torch.cat([beta_dist,params2],dim=1)
        width=self.get_width(representation_2d,params3)

        '''reshape and final regressor'''
        output_2d = torch.cat([approach, beta_dist, width], dim=1)
        output=reshape_for_layer_norm(output_2d, camera=camera, reverse=True)
        output=self.gripper_regressor_layer(output)
        return output

class SuctionPartSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder= nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 3),
        ).to('cuda')
    def forward(self, representation_2d ):
        '''decode'''
        output_2d=self.decoder(representation_2d)
        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)

        '''normalize'''
        output=F.normalize(output,dim=1)
        return output

class GraspSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')
        self.gripper_sampler=GripperPartSampler()
        self.suction_sampler=SuctionPartSampler()


    def forward(self, depth,sampled_approach=None):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        representation = self.back_bone(depth)
        representation_2d=reshape_for_layer_norm(representation, camera=camera, reverse=False)

        '''gripper parameters'''
        gripper_pose=self.gripper_sampler(representation_2d)

        '''suction direction'''
        suction_direction=self.suction_sampler(representation_2d)
        return gripper_pose,suction_direction

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')
        self.att_block = att_res_mlp_LN(in_c1=64,in_c2=7, out_c=1).to('cuda')

    def forward(self, depth,pose):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        features = self.back_bone(depth)
        features_2d=reshape_for_layer_norm(features, camera=camera, reverse=False)
        pose_2d=reshape_for_layer_norm(pose, camera=camera, reverse=False)

        '''decode'''
        output_2d = self.att_block(features_2d,pose_2d)

        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)

        return output

