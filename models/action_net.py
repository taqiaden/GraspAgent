import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore
from torch import nn
from Configurations.config import theta_scope, phi_scope
from lib.custom_activations import GripperGraspRegressor2
from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth
from visualiztion import view_features

use_bn=False
use_in=True
action_module_key='action_net'
action_module_key2='action_net2'

critic_relu_slope=0.1
classification_relu_slope=0.1
generator_backbone_relu_slope=0.1
gripper_sampler_relu_slope=0.1
suction_sampler_relu_slope=0.1

def random_approach_tensor(size):
    # random_tensor = torch.rand_like(approach)
    random_tensor = torch.rand(size=(size,3),device='cuda')

    '''fit to scope'''
    assert theta_scope == 90. and phi_scope == 360.
    random_tensor[:, 0:2] = ((random_tensor[:, 0:2] * 2) - 1)*0.1
    random_tensor[2]=(1.-random_tensor[2]**2)*0.9+0.1

    return random_tensor

def randomize_approach(approach,alpha=0.0,random_tensor=None):
    '''scale to the size of the base vector'''
    norm=torch.norm(approach,dim=-1,keepdim=True).detach()
    random_norm=torch.norm(random_tensor,dim=-1,keepdim=True).detach()

    random_tensor*=(norm/random_norm)

    '''add the randomization'''
    randomized_approach=approach*(1-alpha)+random_tensor*alpha

    return randomized_approach


def randomize_beta(beta,alpha=0.0):
    '''scale to the size of the base vector'''
    random_tensor=torch.randn_like(beta)

    norm=torch.norm(beta,dim=-1,keepdim=True).detach()
    random_norm=torch.norm(random_tensor,dim=-1,keepdim=True).detach()
    random_tensor*=(norm/random_norm)

    '''add the randomization'''
    randomized_approach=beta*(1-alpha)+random_tensor*alpha

    return randomized_approach

class GripperPartSampler(nn.Module):
    def __init__(self):
        super().__init__()

        # self.approach_decoder=att_res_mlp_LN(in_c1=64, in_c2=3, out_c=3,relu_negative_slope=gripper_sampler_relu_slope).to('cuda')
        self.beta_decoder=att_res_mlp_LN(in_c1=64, in_c2=3, out_c=2,relu_negative_slope=gripper_sampler_relu_slope).to('cuda')
        self.dist_width_decoder=att_res_mlp_LN(in_c1=64, in_c2=5, out_c=2,relu_negative_slope=gripper_sampler_relu_slope).to('cuda')
        self.gripper_regressor_layer=GripperGraspRegressor2()

    def latent_noise_injection(self,approach,beta,dist_width,randomization_factor):
        beta = F.normalize(beta, dim=-1)
        dist_width = torch.clip(dist_width, 0.0, 1.)

        if np.random.rand() < randomization_factor:
            beta_noise = torch.randn((beta.shape[0], 2), device='cuda')
            beta = beta * (1.0 - randomization_factor) + beta_noise * randomization_factor

        if np.random.rand() < randomization_factor:
            width_noise = 1. - torch.rand(size=(beta.shape[0], 1), device='cuda') ** 5
            dist_width[:, 1:2] = (1.0 - randomization_factor) * dist_width[:, 1:2] + width_noise * randomization_factor

        if np.random.rand() < randomization_factor:
            dist_noise = torch.rand(size=(beta.shape[0], 1), device='cuda') ** 5
            dist_width[:, 0:1] = (1.0 - randomization_factor) * dist_width[:, 0:1] + dist_noise * randomization_factor

        pose = torch.cat([approach, beta, dist_width], dim=-1)
        pose = self.gripper_regressor_layer(pose, clip=True, sigmoid=False)

        return pose

    def forward(self,representation_2d,approach_seed=None,clip=False,randomization_factor=0.0,dist_width_sigmoid=False):
        approach=approach_seed
        beta = self.beta_decoder(representation_2d, approach)
        dist_width = self.dist_width_decoder(representation_2d, torch.cat([approach, beta], dim=-1))

        if randomization_factor>0.:
            pose =self.latent_noise_injection(approach,beta,dist_width, randomization_factor)
        else:
            pose = torch.cat([approach, beta, dist_width], dim=-1)
            pose = self.gripper_regressor_layer(pose, clip=clip, sigmoid=dist_width_sigmoid)

        return pose

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

        self.gripper_collision = att_res_mlp_LN(in_c1=64, in_c2=7, out_c=2,relu_negative_slope=classification_relu_slope).to('cuda')

        self.suction_quality = att_res_mlp_LN(in_c1=64, in_c2=3, out_c=1,relu_negative_slope=classification_relu_slope).to('cuda')

        self.shift_affordance = att_res_mlp_LN(in_c1=64, in_c2=5, out_c=1,relu_negative_slope=classification_relu_slope).to('cuda')

        self.background_detector=att_res_mlp_LN(in_c1=64, in_c2=2, out_c=1,relu_negative_slope=classification_relu_slope).to('cuda')

        self.sigmoid=nn.Sigmoid()


    def set_seeds(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(self, depth,alpha=0.0,random_tensor=None,seed=None,detach_backbone=False,clip=False,randomization_factor=0.0,dist_width_sigmoid=False):
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

        '''suction direction'''
        normal_direction=self.suction_sampler(features)

        '''gripper parameters'''
        if self.training:
            vertical_direction=torch.zeros_like(normal_direction)
            vertical_direction[:,2]+=1.0

            self.set_seeds(seed)
            random_mask=torch.rand_like(normal_direction)>0.5
            gripper_approach_direction=normal_direction.clone().detach()
            gripper_approach_direction[random_mask]=vertical_direction[random_mask]
            gripper_pose=self.gripper_sampler(features,approach_seed=gripper_approach_direction,clip=clip,randomization_factor=randomization_factor)

            '''gripper collision head'''
            gripper_pose_detached = gripper_pose.detach().clone()
            gripper_pose_detached[:, -2:] = torch.clip(gripper_pose_detached[:, -2:], 0., 1.)
            griper_collision_classifier = self.gripper_collision(features, gripper_pose_detached)
        else:
            '''first forward with approach set to normal'''
            gripper_pose_1 = self.gripper_sampler(features, approach_seed=normal_direction.detach(), clip=clip,
                                                randomization_factor=randomization_factor)

            #gripper collision head
            gripper_pose_1_detached = gripper_pose_1.detach().clone()
            gripper_pose_1_detached[:, -2:] = torch.clip(gripper_pose_1_detached[:, -2:], 0., 1.)
            griper_collision_classifier_1 = self.gripper_collision(features, gripper_pose_1_detached)

            '''second forward with approach set to vertical'''
            vertical_direction = torch.zeros_like(normal_direction)
            vertical_direction[:, 2] += 1.0
            gripper_pose_2 = self.gripper_sampler(features, approach_seed=vertical_direction, clip=clip,
                                                  randomization_factor=randomization_factor)

            # gripper collision head
            gripper_pose_2_detached = gripper_pose_2.detach().clone()
            gripper_pose_2_detached[:, -2:] = torch.clip(gripper_pose_2_detached[:, -2:], 0., 1.)
            griper_collision_classifier_2 = self.gripper_collision(features, gripper_pose_2_detached)

            '''rank best approach based on collision avoidance'''
            # collision_with bin
            collision_mask_1=(griper_collision_classifier_1[:,0]<0.5) & (griper_collision_classifier_1[:,1]<0.5)
            collision_mask_2=(griper_collision_classifier_2[:,0]<0.5) & (griper_collision_classifier_2[:,1]<0.5)

            # check 1: use vertical direction if it solves collision better than normal
            bad_normal_approach_mask=collision_mask_2 & (~collision_mask_1)

            # print(bad_normal_approach_mask.shape)
            # print(collision_mask_1.shape)
            # print((gripper_pose_2[:,-2]>gripper_pose_1[:,-2]).shape)

            # check 2: if both direction are good at avoiding collision, check for better firmness
            bad_normal_approach_mask=bad_normal_approach_mask | (collision_mask_2 & collision_mask_1 & (gripper_pose_2[:,-2]>gripper_pose_1[:,-2]))

            '''superiority-rank based sampling of poses and associated collision scores'''

            griper_collision_classifier=griper_collision_classifier_1
            griper_collision_classifier[bad_normal_approach_mask]=griper_collision_classifier_2[bad_normal_approach_mask]

            gripper_pose=gripper_pose_1
            gripper_pose[bad_normal_approach_mask]=gripper_pose_2[bad_normal_approach_mask]








        '''suction quality head'''
        normal_direction_detached=normal_direction.detach().clone()
        suction_quality_classifier = self.suction_quality(features, normal_direction_detached)

        '''shift affordance head'''
        shift_query_features=torch.cat([normal_direction_detached,self.spatial_encoding], dim=-1)
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
        normal_direction = reshape_for_layer_norm(normal_direction, camera=camera, reverse=True)
        griper_collision_classifier = reshape_for_layer_norm(griper_collision_classifier, camera=camera, reverse=True)
        suction_quality_classifier = reshape_for_layer_norm(suction_quality_classifier, camera=camera, reverse=True)
        shift_affordance_classifier = reshape_for_layer_norm(shift_affordance_classifier, camera=camera, reverse=True)
        background_class=reshape_for_layer_norm(background_class, camera=camera, reverse=True)

        return gripper_pose,normal_direction,griper_collision_classifier,suction_quality_classifier,shift_affordance_classifier,background_class,features

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in,relu_negative_slope=critic_relu_slope).to('cuda')

        self.att_block= att_res_mlp_LN(in_c1=64,in_c2=7, out_c=1,relu_negative_slope=critic_relu_slope).to('cuda')
        # self.att_block_2= att_res_mlp_LN_light(in_c1=64,in_c2=7, out_c=1,relu_negative_slope=critic_relu_slope).to('cuda')

        # self.att_block2= att_res_mlp_LN2(in_c1=64,in_c2=7, out_c=1,relu_negative_slope=critic_relu_slope).to('cuda')
        # self.att_block3= att_res_mlp_LN2(in_c1=64,in_c2=7, out_c=1,relu_negative_slope=critic_relu_slope).to('cuda')

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
        output_2d = self.att_block(features_2d,pose_2d)#*self.att_block2(features_2d,pose_2d)+self.att_block3(features_2d,pose_2d)
        # output_2 = self.att_block_2(features_2d,pose_2d)#*self.att_block2(features_2d,pose_2d)+self.att_block3(features_2d,pose_2d)

        # output_2d=torch.cat([output_1,output_2],dim=-1)
        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)

        return output