import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore
from torch import nn
from torch.nn.utils import spectral_norm

from Configurations.config import theta_scope, phi_scope
from lib.cuda_utils import cuda_memory_report
from lib.custom_activations import GripperGraspRegressor2, LGRelu
from lib.models_utils import reshape_for_layer_norm, number_of_parameters
from models.Grasp_GAN import GripperGraspSampler
from models.decoders import att_res_conv_normalized, LayerNorm2D
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth
from visualiztion import view_features, plt_features

# activation=LGRelu(slope=0.01)
silu=nn.SiLU()
relu=nn.ReLU()

action_module_key = 'action_net'


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


mish = Mish()

class SuctionPartSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # LayerNorm2D(64),
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            silu,
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            LayerNorm2D(16),
            silu,
            nn.Conv2d(16, 3, kernel_size=1),
        ).to('cuda')

    def forward(self, representation_2d):
        '''decode'''
        output_2d = self.decoder(representation_2d)

        '''normalize'''
        output_2d = F.normalize(output_2d, dim=1)
        return output_2d


class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.0,activation=silu).to('cuda')

        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)
        # self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        self.gripper_sampler1 = GripperGraspSampler()
        # self.gripper_sampler2 = GripperGraspSampler(activation,use_sig=False)

        self.suction_sampler = SuctionPartSampler()

        # self.LN = LayerNorm2D(64).to('cuda')

        self.gripper_collision_ = att_res_conv_normalized(in_c1=64, in_c2=7 +1, out_c=2,
                                                relu_negative_slope=0.0,activation=silu,drop_out_ratio=0.).to(
            'cuda')

        self.suction_quality_ = att_res_conv_normalized(in_c1=64, in_c2=3, out_c=1,
                                              relu_negative_slope=0.0,activation=silu,drop_out_ratio=0.).to(
            'cuda')

        self.shift_affordance_ = att_res_conv_normalized(in_c1=64, in_c2=5, out_c=1,
                                               relu_negative_slope=0.0,activation=silu,drop_out_ratio=0.).to(
            'cuda')

        self.background_detector_ = att_res_conv_normalized(in_c1=64, in_c2=2, out_c=1,
                                                  relu_negative_slope=0.0,activation=silu,drop_out_ratio=0.).to(
            'cuda')

        # self.LN = LayerNorm2D(64).to('cuda')

        self.sigmoid = nn.Sigmoid()


    def gripper_sample_and_classify_(self,features, depth,approach_direction,sampling_module):
        # cuda_memory_report()
        # print(features.shape)
        # print(approach_direction.shape)

        gripper_pose = sampling_module(features, approach=approach_direction)
        # cuda_memory_report()

        '''gripper collision head'''
        gripper_pose_detached = gripper_pose.detach()
        gripper_pose_detached[:, -2:] = torch.clip(gripper_pose_detached[:, -2:], 0., 1.)
        gripper_pose_detached[:,0:3] = F.normalize(gripper_pose_detached[:,0:3], dim=1)
        gripper_pose_detached[:,3:5] = F.normalize(gripper_pose_detached[:,3:5], dim=1)
        griper_collision_classifier = self.gripper_collision_(features, torch.cat([gripper_pose_detached,depth],dim=1))
        griper_collision_classifier=self.sigmoid(griper_collision_classifier)
        # cuda_memory_report()

        return gripper_pose,griper_collision_classifier

    def get_best_approach(self,depth_):
        depth = standardize_depth(depth_)
        features = self.back_bone(depth)

        # features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        # depth = reshape_for_layer_norm(depth, camera=camera, reverse=False)

        predicted_normal = self.suction_sampler(features)

        gripper_pose,griper_collision_classifier,approach=self.gripper_sampling_and_quality_inference( features, depth,predicted_normal.detach() )


        return gripper_pose,griper_collision_classifier,approach

    def collision_prediction(self,depth,gripper_pose):
        depth = standardize_depth(depth)

        features = self.back_bone(depth)

        # features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        # depth = reshape_for_layer_norm(depth, camera=camera, reverse=False)
        # gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=False)


        gripper_pose_detached = gripper_pose.detach()
        gripper_pose_detached[:, -2:] = torch.clip(gripper_pose_detached[:, -2:], 0., 1.)
        gripper_pose_detached[:,0:3] = F.normalize(gripper_pose_detached[:,0:3], dim=1)
        gripper_pose_detached[:,3:5] = F.normalize(gripper_pose_detached[:,3:5], dim=1)
        griper_collision_classifier = self.gripper_collision_(features, torch.cat([gripper_pose_detached, depth], dim=1))
        griper_collision_classifier = self.sigmoid(griper_collision_classifier)

        # griper_collision_classifier = reshape_for_layer_norm(griper_collision_classifier, camera=camera,
        #                                                      reverse=True)

        return griper_collision_classifier

    def gripper_sampling_and_quality_inference(self,features, depth,normal_direction=None,approach1=None,approach2=None):
        if (approach1 is not None) and (approach2 is not None):

            # approach1 = reshape_for_layer_norm(approach1, camera=camera, reverse=False)
            # approach2 = reshape_for_layer_norm(approach2, camera=camera, reverse=False)

            gripper_pose_1, griper_collision_classifier_1 = self.gripper_sample_and_classify_(features, depth,
                                                                                             approach1,self.gripper_sampler1)

            # gripper_pose_1 = reshape_for_layer_norm(gripper_pose_1, camera=camera, reverse=True)
            # griper_collision_classifier_1 = reshape_for_layer_norm(griper_collision_classifier_1, camera=camera,
            #                                                      reverse=True)
            gripper_pose_2, griper_collision_classifier_2 = self.gripper_sample_and_classify_(features, depth,
                                                                                              approach2,self.gripper_sampler1)

            # gripper_pose_2 = reshape_for_layer_norm(gripper_pose_2, camera=camera, reverse=True)
            # griper_collision_classifier_2 = reshape_for_layer_norm(griper_collision_classifier_2, camera=camera,
            #                                                      reverse=True)

            return (gripper_pose_1,gripper_pose_2), (griper_collision_classifier_1,griper_collision_classifier_2),None

        elif approach1 is not None:

            # approach1 = reshape_for_layer_norm(approach1, camera=camera, reverse=False)

            gripper_pose, griper_collision_classifier = self.gripper_sample_and_classify_(features, depth,
                                                                                              approach1,
                                                                                              self.gripper_sampler1)

            # gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=True)
            # griper_collision_classifier = reshape_for_layer_norm(griper_collision_classifier, camera=camera,
            #                                                        reverse=True)

            return gripper_pose,griper_collision_classifier,None
        else:
            vertical_direction = torch.zeros_like(normal_direction)
            vertical_direction[:, 2] = vertical_direction[:, 2] + 1.0
            '''first forward with approach set to normal'''
            gripper_pose_1,griper_collision_classifier_1=self.gripper_sample_and_classify_( features, depth, vertical_direction,self.gripper_sampler1)

            '''second forward with approach set to vertical'''
            gripper_pose_2,griper_collision_classifier_2=self.gripper_sample_and_classify_( features, depth, normal_direction.detach(),self.gripper_sampler1)

            '''rank best approach based on collision avoidance'''
            # collision_with bin
            no_collision_mask_1 = (griper_collision_classifier_1[:, 0] < 0.5) & (griper_collision_classifier_1[:, 1] < 0.5)
            no_collision_mask_2 = (griper_collision_classifier_2[:, 0] < 0.5) & (griper_collision_classifier_2[:, 1] < 0.5)



            # check 1: use vertical direction if it solves collision better than normal
            good_approach_mask_2 = no_collision_mask_2 & (~no_collision_mask_1)

            good_approach_mask_2=good_approach_mask_2 | ((griper_collision_classifier_2[:, 0]<griper_collision_classifier_1[:, 0]) & (griper_collision_classifier_2[:, 1]<griper_collision_classifier_1[:, 1]))

            good_approach_mask_2=good_approach_mask_2 | ((griper_collision_classifier_2[:, 0]<griper_collision_classifier_1[:, 0]) & (griper_collision_classifier_2[:, 1]<0.5) & (griper_collision_classifier_1[:, 1]<0.5))

            good_approach_mask_2=good_approach_mask_2 | ((griper_collision_classifier_2[:, 0]<0.5)&(griper_collision_classifier_1[:, 0]<0.5) & (griper_collision_classifier_2[:, 1]<griper_collision_classifier_1[:, 1]))

            # check 2: if both direction are good at avoiding collision, check for better firmness
            # good_approach_mask_2 = good_approach_mask_2 | (
            #         no_collision_mask_2 & no_collision_mask_1 & (gripper_pose_2[:, -2] > gripper_pose_1[:, -2]))

            '''superiority-rank based sampling of poses and associated collision scores'''
            griper_collision_classifier=torch.where(good_approach_mask_2[:,None],griper_collision_classifier_2,griper_collision_classifier_1)

            gripper_pose=torch.where(good_approach_mask_2[:,None],gripper_pose_2,gripper_pose_1)
            approach=torch.where(good_approach_mask_2[:,None],normal_direction,vertical_direction)

            # '''reshape'''
            # gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=True)
            # griper_collision_classifier = reshape_for_layer_norm(griper_collision_classifier, camera=camera,
            #                                                      reverse=True)
            # approach = reshape_for_layer_norm(approach, camera=camera, reverse=True)


            return gripper_pose,griper_collision_classifier,approach

    def forward(self, depth,approach1=None,approach2=None, detach_backbone=False):
        '''input standardization'''
        depth = standardize_depth(depth)
        batch_size=depth.shape[0]

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth).detach()
        else:
            features = self.back_bone(depth)

        # features=self.LN(features)


        # features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        # depth = reshape_for_layer_norm(depth, camera=camera, reverse=False)

        '''Spatial data'''
        if self.spatial_encoding.shape[0] != depth.shape[0]:
            self.spatial_encoding = depth_xy_spatial_data(batch_size=batch_size)
            # self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)


        # cuda_memory_report()

        '''background detection'''
        background_class = self.background_detector_(features, self.spatial_encoding)

        # cuda_memory_report()

        '''suction direction'''
        predicted_normal = self.suction_sampler(features)

        # cuda_memory_report()

        '''gripper sampler and quality inference'''
        gripper_pose,griper_collision_classifier,_=self.gripper_sampling_and_quality_inference( features, depth, predicted_normal.detach() ,approach1,approach2)

        # cuda_memory_report()

        # view_features(gripper_pose)
        # plt_features(gripper_pose)

        '''suction quality head'''
        normal_direction_detached = predicted_normal.detach()
        suction_quality_classifier = self.suction_quality_(features, normal_direction_detached)

        # cuda_memory_report()

        '''shift affordance head'''
        shift_query_features = torch.cat([normal_direction_detached, self.spatial_encoding], dim=1)
        shift_affordance_classifier = self.shift_affordance_(features, shift_query_features)

        '''sigmoid'''
        suction_quality_classifier = self.sigmoid(suction_quality_classifier)
        shift_affordance_classifier = self.sigmoid(shift_affordance_classifier)
        background_class = self.sigmoid(background_class)

        # '''reshape'''
        # predicted_normal = reshape_for_layer_norm(predicted_normal, camera=camera, reverse=True)
        #
        # suction_quality_classifier = reshape_for_layer_norm(suction_quality_classifier, camera=camera, reverse=True)
        # shift_affordance_classifier = reshape_for_layer_norm(shift_affordance_classifier, camera=camera,
        #                                                      reverse=True)
        # background_class = reshape_for_layer_norm(background_class, camera=camera, reverse=True)

        return gripper_pose, predicted_normal, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier, background_class
