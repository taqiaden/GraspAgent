import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore
from torch import nn
from Configurations.config import theta_scope, phi_scope
from lib.cuda_utils import cuda_memory_report
from lib.custom_activations import GripperGraspRegressor2
from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN, att_res_mlp_LN2
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth
from visualiztion import view_features

use_bn=False
use_in=True
action_module_key='action_net'
critic_relu_slope=0.2
classification_relu_slope=0.2
generator_backbone_relu_slope=0.2
gripper_sampler_relu_slope=0.2
suction_sampler_relu_slope=0.2

def random_approach_tensor(size):
    # random_tensor = torch.rand_like(approach)
    random_tensor = torch.rand(size=(size,3),device='cuda')
    '''fit to scope'''
    assert theta_scope == 90. and phi_scope == 360.
    random_tensor[:, 0:2] = (random_tensor[:, 0:2] * 2) - 1

    return random_tensor

def randomize_approach(approach,alpha=0.0,random_tensor=None):
    '''scale to the size of the base vector'''
    norm=torch.norm(approach,dim=-1,keepdim=True).detach()
    random_norm=torch.norm(random_tensor,dim=-1,keepdim=True).detach()

    random_tensor*=(norm/random_norm)

    '''add the randomization'''
    randomized_approach=approach*(1-alpha)+random_tensor*alpha

    return randomized_approach

class GripperPartSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=nn.Sequential(
            nn.Linear(64, 16, bias=False),
            nn.LayerNorm([16]),
            nn.LeakyReLU(negative_slope=gripper_sampler_relu_slope) if gripper_sampler_relu_slope>0. else nn.ReLU(),
            nn.Linear(16, 7),
        ).to('cuda')

        # self.decoder_=self_att_res_mlp_LN(in_c1=64,out_c=7,relu_negative_slope=0.0).to('cuda')
        self.residual_1=att_res_mlp_LN(in_c1=64, in_c2=7, out_c=4,relu_negative_slope=gripper_sampler_relu_slope).to('cuda')
        self.residual_2=att_res_mlp_LN(in_c1=64, in_c2=7, out_c=2,relu_negative_slope=gripper_sampler_relu_slope).to('cuda')

        self.gripper_regressor_layer=GripperGraspRegressor2()

    def forward(self,representation_2d,alpha=0.,random_tensor=None,clip=False,refine_grasp=True):
        prediction=self.decoder(representation_2d)

        if alpha > 0.:
            random_tensor_ = random_approach_tensor(
                representation_2d.shape[0]) if random_tensor is None else random_tensor.clone()
            prediction[:,0:3] = randomize_approach(prediction[:,0:3], alpha=alpha, random_tensor=random_tensor_)

        residuals1=self.residual_1(representation_2d,prediction)
        approach=prediction[:,0:3]
        rest_of_pose=prediction[:,3:]
        rest_of_pose=rest_of_pose+residuals1
        prediction=torch.cat([approach,rest_of_pose],dim=-1)
        residuals2=self.residual_2(representation_2d,prediction)
        approach_beta=prediction[:,0:5]
        rest_of_pose=prediction[:,5:]
        rest_of_pose=rest_of_pose+residuals2
        pose=torch.cat([approach_beta,rest_of_pose],dim=-1)

        # print(residuals)
        # print(prediction)
        # a=int(refine_grasp)
        # prediction=residuals*a+prediction.detach()*a+prediction*(1-a)

        output=self.gripper_regressor_layer(pose,clip=clip)
        return output


class SuctionPartSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder= nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.LeakyReLU(negative_slope=suction_sampler_relu_slope) if suction_sampler_relu_slope>0. else nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.LeakyReLU(negative_slope=suction_sampler_relu_slope) if suction_sampler_relu_slope>0. else nn.ReLU(),
            nn.Linear(16, 3),
        ).to('cuda')
    def forward(self, representation_2d ):
        '''decode'''
        output_2d=self.decoder(representation_2d)

        '''normalize'''
        output_2d=F.normalize(output_2d,dim=1)
        return output_2d

class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in,relu_negative_slope=generator_backbone_relu_slope).to('cuda')

        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)
        self.spatial_encoding=reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        self.gripper_sampler=GripperPartSampler()
        self.suction_sampler=SuctionPartSampler()

        self.gripper_collision = att_res_mlp_LN(in_c1=64, in_c2=7, out_c=1,relu_negative_slope=classification_relu_slope).to('cuda')

        self.suction_quality = att_res_mlp_LN(in_c1=64, in_c2=3, out_c=1,relu_negative_slope=classification_relu_slope).to('cuda')

        self.shift_affordance = att_res_mlp_LN(in_c1=64, in_c2=5, out_c=1,relu_negative_slope=classification_relu_slope).to('cuda')

        self.background_detector=att_res_mlp_LN(in_c1=64, in_c2=2, out_c=1,relu_negative_slope=classification_relu_slope).to('cuda')

        self.sigmoid=nn.Sigmoid()

    def forward(self, depth,alpha=0.0,random_tensor=None,detach_backbone=False,clip=False,refine_grasp=True):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        # depth_features=features.detach().clone()
        features=reshape_for_layer_norm(features, camera=camera, reverse=False)
        # if detach_backbone: features=features.detach()
        # view_features(features)

        # print(features[0])
        # exit()
        '''check exploded values'''
        if self.training:
            max_=features.max()
            if max_>100:
                print(Fore.RED,f'Warning: Res U net outputs high values up to {max_}',Fore.RESET)

        '''Spatial data'''
        if self.spatial_encoding.shape[0] != depth.shape[0]:
            self.spatial_encoding = depth_xy_spatial_data(batch_size=depth.shape[0])
            self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        '''gripper parameters'''
        gripper_pose=self.gripper_sampler(features,alpha=alpha,random_tensor=random_tensor,clip=clip,refine_grasp=refine_grasp)

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

        return gripper_pose,suction_direction,griper_collision_classifier,suction_quality_classifier,shift_affordance_classifier,background_class,features

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in,relu_negative_slope=critic_relu_slope).to('cuda')

        self.att_block1= att_res_mlp_LN2(in_c1=64,in_c2=7, out_c=1,relu_negative_slope=critic_relu_slope).to('cuda')
        self.att_block2= att_res_mlp_LN2(in_c1=64,in_c2=7, out_c=1,relu_negative_slope=critic_relu_slope).to('cuda')
        self.att_block3= att_res_mlp_LN2(in_c1=64,in_c2=7, out_c=1,relu_negative_slope=critic_relu_slope).to('cuda')

    def forward(self, depth,pose,detach_backbone=False):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        features_2d=reshape_for_layer_norm(features, camera=camera, reverse=False)
        pose_2d=reshape_for_layer_norm(pose, camera=camera, reverse=False)

        # view_features(features_2d)
        '''decode'''
        output_2d = self.att_block1(features_2d,pose_2d)*self.att_block2(features_2d,pose_2d)+self.att_block3(features_2d,pose_2d)

        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)

        return output