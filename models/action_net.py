import torch
import torch.nn.functional as F
from torch import nn
from Configurations.config import theta_scope, phi_scope
from lib.custom_activations import GripperGraspRegressor2
from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN, att_res_mlp_LN2, res_block_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth

use_bn=False
use_in=True
action_module_key='action_net'

def random_approach_tensor(size):
    # random_tensor = torch.rand_like(approach)
    random_tensor = torch.rand(size=(size,3),device='cuda')
    '''fit to scope'''
    assert theta_scope == 90. and phi_scope == 360.
    random_tensor[:, 0:2] = (random_tensor[:, 0:2] * 2) - 1

    return random_tensor

def randomize_approach(approach,alpha=0.0,random_tensor=None):
    random_tensor_=random_approach_tensor(approach.shape[0]) if random_tensor is None else random_tensor.clone()
    '''scale to the size of the base vector'''
    norm=torch.norm(approach,dim=-1,keepdim=True).detach()
    random_norm=torch.norm(random_tensor_,dim=-1,keepdim=True).detach()

    random_tensor_*=(norm/random_norm)

    '''add the randomization'''
    randomized_approach=approach*(1-alpha)+random_tensor_*alpha

    return randomized_approach

class GripperPartSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.get_approach=att_res_mlp_LN(in_c1=64, in_c2=2, out_c=3).to('cuda')
        self.get_beta_dist_width=att_res_mlp_LN(in_c1=64, in_c2=5, out_c=4).to('cuda')

        self.gripper_regressor_layer=GripperGraspRegressor2()

    def forward(self,representation_2d,spatial_data_2d, alpha=0.,random_tensor=None):
        '''Approach'''
        approach=self.get_approach(representation_2d,spatial_data_2d)
        if alpha>0.:
            approach=randomize_approach(approach, alpha=alpha,random_tensor=random_tensor)

        '''Beta, distance, and width'''
        position_approach=torch.cat([spatial_data_2d,approach],dim=1)
        beta_dist_width=self.get_beta_dist_width(representation_2d,position_approach)

        '''Regress'''
        output_2d = torch.cat([approach, beta_dist_width], dim=1)
        output_2d=self.gripper_regressor_layer(output_2d)
        return output_2d

class SuctionPartSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder= nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(),
            nn.Linear(16, 3),
        ).to('cuda')
    def forward(self, representation_2d ):
        '''decode'''
        output_2d=self.decoder(representation_2d)

        '''normalize'''
        output_2d=F.normalize(output_2d,dim=1)
        return output_2d

class AbstractQualityClassifier(nn.Module):
    def __init__(self,in_c1, in_c2, out_c):
        super().__init__()
        self.att_block = att_res_mlp_LN2(in_c1=in_c1, in_c2=in_c2, out_c=out_c).to('cuda')

    def forward(self, features_2d,pose_2d ):
        output_2d = self.att_block(features_2d,pose_2d)
        return output_2d

class BackgroundDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.res=res_block_mlp_LN(in_c=64,medium_c=32,out_c=16,activation=nn.ReLU()).to('cuda')
        self.decoder= nn.Sequential(
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        ).to('cuda')
        self.sig=nn.Sigmoid()

    def forward(self, depth_features_2d ):
        '''decode'''
        output_2d = self.decoder(self.res(depth_features_2d))
        output_2d=self.sig(output_2d)
        return output_2d

class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')

        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)
        self.spatial_encoding=reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        self.gripper_sampler=GripperPartSampler()
        self.suction_sampler=SuctionPartSampler()

        self.gripper_collision = AbstractQualityClassifier(in_c1=64, in_c2=7, out_c=1)
        self.suction_quality = AbstractQualityClassifier(in_c1=64, in_c2=3, out_c=1)
        self.shift_affordance = AbstractQualityClassifier(in_c1=64, in_c2=2+3, out_c=1)
        self.background_detector=AbstractQualityClassifier(in_c1=64, in_c2=2, out_c=1)

        self.sigmoid=nn.Sigmoid()


    def forward(self, depth,alpha=0.0,random_tensor=None):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        features = self.back_bone(depth)
        depth_features=features.detach().clone()
        features=reshape_for_layer_norm(features, camera=camera, reverse=False)

        '''Spatial data'''
        if self.spatial_encoding.shape[0] != depth.shape[0]:
            self.spatial_encoding = depth_xy_spatial_data(batch_size=depth.shape[0])
            self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        '''gripper parameters'''
        gripper_pose=self.gripper_sampler(features,self.spatial_encoding,alpha=alpha,random_tensor=random_tensor)

        '''suction direction'''
        suction_direction=self.suction_sampler(features)

        '''gripper collision head'''
        gripper_pose_detached=gripper_pose.detach().clone()
        gripper_pose_detached[:,-2:]=torch.clip(gripper_pose_detached[:,-2:],0.,1.)
        griper_collision_classifier = self.gripper_collision(features, gripper_pose_detached)

        '''suction quality head'''
        suction_direction_detached=suction_direction.detach().clone()
        suction_quality_classifier = self.suction_quality(features, suction_direction_detached)

        '''shift affordance head'''
        shift_query_features=torch.cat([suction_direction_detached,self.spatial_encoding], dim=-1)
        shift_affordance_classifier = self.shift_affordance(features,shift_query_features )

        '''background detection'''
        background_class=self.background_detector(features,self.spatial_encoding)

        '''sigmoid'''
        griper_collision_classifier=self.sigmoid(griper_collision_classifier)
        suction_quality_classifier=self.sigmoid(suction_quality_classifier)
        shift_affordance_classifier=self.sigmoid(shift_affordance_classifier)
        background_class=self.sigmoid(background_class)

        '''reshape'''
        gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=True)
        suction_direction = reshape_for_layer_norm(suction_direction, camera=camera, reverse=True)
        griper_collision_classifier = reshape_for_layer_norm(griper_collision_classifier, camera=camera, reverse=True)
        suction_quality_classifier = reshape_for_layer_norm(suction_quality_classifier, camera=camera, reverse=True)
        shift_affordance_classifier = reshape_for_layer_norm(shift_affordance_classifier, camera=camera, reverse=True)
        background_class=reshape_for_layer_norm(background_class, camera=camera, reverse=True)

        return gripper_pose,suction_direction,griper_collision_classifier,suction_quality_classifier,shift_affordance_classifier,background_class,depth_features

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')

        self.att_block = att_res_mlp_LN(in_c1=64,in_c2=9, out_c=1).to('cuda')

        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)
        self.spatial_encoding=reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)


    def forward(self, depth,pose):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        features = self.back_bone(depth)
        features_2d=reshape_for_layer_norm(features, camera=camera, reverse=False)
        pose_2d=reshape_for_layer_norm(pose, camera=camera, reverse=False)

        '''Spatial data'''
        if self.spatial_encoding.shape[0] != depth.shape[0]:
            self.spatial_encoding = depth_xy_spatial_data(batch_size=depth.shape[0])
            self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        pose_2d=torch.cat([pose_2d,self.spatial_encoding],dim=-1)

        '''decode'''
        output_2d = self.att_block(features_2d,pose_2d)

        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)

        return output
