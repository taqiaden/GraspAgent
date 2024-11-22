import torch
from torch import nn
import torch.nn.functional as F
from Configurations.config import theta_scope, phi_scope
from lib.custom_activations import GripperGraspRegressor
from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import SpatialEncoder
from registration import camera, standardize_depth

use_bn=False
use_in=True

def randomize_approach(approach,randomization_ratio=0.0):
    random_tensor=torch.rand_like(approach)

    '''fit to scope'''
    assert theta_scope==90. and phi_scope==360.
    random_tensor[:,0:2]=(random_tensor[:,0:2]*2)-1

    '''add the randomization'''
    randomized_approach=approach*(1-randomization_ratio)+random_tensor*randomization_ratio

    return randomized_approach

class GripperPartSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.get_approach=att_res_mlp_LN(in_c1=64, in_c2=8, out_c=3).to('cuda')
        self.get_beta_dist_width=att_res_mlp_LN(in_c1=64, in_c2=11, out_c=4).to('cuda')

        self.gripper_regressor_layer=GripperGraspRegressor()

    def forward(self,representation_2d,spatial_data_2d, approach_randomness_ratio=0.):
        '''Approach'''
        approach=self.get_approach(representation_2d,spatial_data_2d)
        if approach_randomness_ratio>0.:
            approach=randomize_approach(approach, randomization_ratio=approach_randomness_ratio)

        '''Beta, distance, and width'''
        position_approach=torch.cat([spatial_data_2d,approach],dim=1)
        beta_dist_width=self.get_beta_dist_width(representation_2d,position_approach)

        '''Regress'''
        output_2d = torch.cat([approach, beta_dist_width], dim=1)
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
        self.spatial_encoding = SpatialEncoder()
        self.gripper_sampler=GripperPartSampler()
        self.suction_sampler=SuctionPartSampler()

    def forward(self, depth,sampled_approach=None):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        representation = self.back_bone(depth)
        representation_2d=reshape_for_layer_norm(representation, camera=camera, reverse=False)

        '''Spatial data'''
        spatial_data_2d = self.spatial_encoding(depth.shape[0])

        '''gripper parameters'''
        gripper_pose=self.gripper_sampler(representation_2d,spatial_data_2d,approach_randomness_ratio=0.0)

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
