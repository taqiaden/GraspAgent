import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore
from torch import nn
from torch.nn.utils import spectral_norm

from Configurations.config import theta_scope, phi_scope
from lib.cuda_utils import cuda_memory_report
from lib.custom_activations import GripperGraspRegressor2
from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN, att_res_mlp_BN
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth
from visualiztion import view_features, plt_features

# backbone_with_IN = True
# critic_backbone_with_IN = True
#
# decoder_with_LN = True

action_module_key = 'action_net'
action_module_key2 = 'action_net2'
action_module_key3 = 'action_net3'

critic_relu_slope = 0.
classification_relu_slope = 0.
generator_backbone_relu_slope = 0.
gripper_sampler_relu_slope = 0.
suction_sampler_relu_slope = 0.


class GripperGraspSampler(nn.Module):
    def __init__(self):
        super().__init__()
        # self.d = nn.Sequential(
        #     nn.Linear(64, 32, bias=False),
        #     nn.LayerNorm([32]),
        #     nn.LeakyReLU(negative_slope=suction_sampler_relu_slope) if suction_sampler_relu_slope > 0. else nn.ReLU(),
        #     nn.Linear(32, 16, bias=False),
        #     nn.LayerNorm([16]),
        #     nn.LeakyReLU(negative_slope=suction_sampler_relu_slope) if suction_sampler_relu_slope > 0. else nn.ReLU(),
        #     nn.Linear(16, 7),
        # ).to('cuda')
        # self.approach_decoder=att_res_mlp_LN(in_c1=64, in_c2=3, out_c=3,relu_negative_slope=gripper_sampler_relu_slope).to('cuda')
        self.beta_decoder = att_res_mlp_LN(in_c1=64, in_c2=3 +1, out_c=2,
                                           relu_negative_slope=gripper_sampler_relu_slope).to(
            'cuda')
        self.dist_width_decoder = att_res_mlp_LN(in_c1=64, in_c2=5+1, out_c=2,
                                                 relu_negative_slope=gripper_sampler_relu_slope).to(
            'cuda')
        self.gripper_regressor_layer = GripperGraspRegressor2()
        self.dropout=nn.Dropout(0.1)

        # self.soft_plus = nn.Softplus()
        self.tanh = nn.Tanh()
        # self.dist_biased_tanh=BiasedTanh(weight=False)
        # self.width_biased_tanh=BiasedTanh(weight=False)

    def change_noise_dist(self,data,noise,k):
        shift_mean=-(k*(data.mean()))/k
        noise=noise+shift_mean
        return noise

    def add_noise(self,data,noise,k):
        data=(1-k)*data+k*noise
        return data

    def latent_noise_injection(self, approach, beta, dist_width, randomization_factor=0.):
        beta = F.normalize(beta, dim=1)
        dist_width = torch.clip(dist_width, 0.0, 1.)

        beta_noise = torch.randn((beta.shape[0], 2), device='cuda')

        # beta_noise[:,0]=self.change_noise_dist(beta[:,0],beta_noise[:,0],k=randomization_factor)
        # beta_noise[:,1]=self.change_noise_dist(beta[:,1],beta_noise[:,1],k=randomization_factor)

        beta=self.add_noise(beta,beta_noise,k=randomization_factor)

        # print(beta)

        # width_noise = 1. - torch.rand(size=(beta.shape[0], 1),
        #                               device='cuda') ** 5. #if decoder_with_LN else 1. - torch.rand(
            # size=(beta.shape[0], 1, beta.shape[2], beta.shape[3]), device='cuda') ** 2
        # width_noise=np.random.lognormal(mean=1.312,sigma=0.505,size=(beta.shape[0], 1)).cuda()
        width_noise = torch.distributions.LogNormal(loc=-1.312, scale=0.505)
        width_noise = 1.-width_noise.sample((beta.shape[0], 1)).cuda()
        dist_width[:, 1:2] = (1.0 - randomization_factor) * dist_width[:, 1:2] + width_noise * randomization_factor

        # dist_noise = torch.rand(size=(beta.shape[0], 1), device='cuda') ** 5. # if decoder_with_LN else torch.rand(
            # size=(beta.shape[0], 1, beta.shape[2], beta.shape[3]), device='cuda') ** 2
        dist_noise = torch.distributions.LogNormal(loc=-1.337, scale=0.791)
        dist_noise = dist_noise.sample((beta.shape[0], 1)).cuda()
        # dist_noise=np.random.lognormal(mean=1.337,sigma=0.791,size=(beta.shape[0], 1)).cuda()

        dist_width[:, 0:1] = (1.0 - randomization_factor) * dist_width[:, 0:1] + dist_noise * randomization_factor

        pose = torch.cat([approach, beta, dist_width], dim=1)
        pose = self.gripper_regressor_layer(pose, sigmoid=False)
        pose = torch.floor(pose * 10) / 10

        return pose

    def latent_noise_injection2(self, pose, randomization_factor=0.):
        pose[:,3:5] = F.normalize(pose[:,3:5], dim=1)
        pose[:,5:] = torch.clip(pose[:,5:], 0.0, 1.)

        beta_noise = torch.randn((pose.shape[0], 2), device='cuda')

        pose[:,3:5]=self.add_noise(pose[:,3:5],beta_noise,k=randomization_factor)
        pose[:,3:5] = torch.floor(pose[:,3:5] * 10) / 10

        width_noise = torch.distributions.LogNormal(loc=-1.312, scale=0.505)
        width_noise = 1.-width_noise.sample((pose.shape[0], 1)).cuda()

        pose[:, 6:7] = (1.0 - randomization_factor) * pose[:, 6:7] + width_noise * randomization_factor
        dist_noise = torch.distributions.LogNormal(loc=-1.337, scale=0.791)
        dist_noise = dist_noise.sample((pose.shape[0], 1)).cuda()

        pose[:, 5:6] = (1.0 - randomization_factor) * pose[:, 5:6] + dist_noise * randomization_factor

        pose[:,3:5] = F.normalize(pose[:,3:5], dim=1)
        pose[:,5:] = torch.clip(pose[:,5:], 0.0, 1.)
        pose[:,0:3] = F.normalize(pose[:,0:3], dim=1)

        return pose


    def add_epsilon_noise(self, pose):

        beta_noise_x=torch.randint(0,2,(pose.shape[0], 1),device='cuda').float()-0.5
        beta_noise_y=torch.randint(0,2,(pose.shape[0], 1),device='cuda').float()-0.5 # -0.5, 0.5
        random_mask=torch.randn((pose.shape[0], 1))>0.
        pose[:, 3:4][random_mask]=pose[:, 3:4][random_mask]+beta_noise_x[random_mask]
        pose[:, 4:5][~random_mask]=pose[:, 4:5][~random_mask]+beta_noise_y[~random_mask]

        width_noise=torch.ones_like(pose[:,0:1]).float()*0.2
        biased_mask=torch.randn((pose.shape[0], 1))+pose[:, 6:7]>0.7
        width_noise[biased_mask]*=-1

        dist_noise=torch.ones_like(pose[:,0:1]).float()*0.1
        biased_mask = torch.randn((pose.shape[0], 1)) + pose[:, 5:6] > 0.3
        dist_noise[biased_mask] *= -1

        pose[:, 6:7] =  pose[:, 6:7] + width_noise
        pose[:, 5:6] =  pose[:, 5:6] + dist_noise

        pose[:,5:] = torch.clip(pose[:,5:], 0.0, 1.)

        return pose

    def forward(self, representation_2d, depth,latent_vector, approach_seed=None, randomization_factor=0.0  ):
        representation_2d=self.dropout(representation_2d)
        approach = approach_seed
        # approach=self.approach_decoder(representation_2d,approach_seed)

        beta = self.beta_decoder(representation_2d, torch.cat([approach,depth],dim=1))
        dist_width = self.dist_width_decoder(representation_2d, torch.cat([ approach, beta,depth], dim=1))
        # pose=self.d(representation_2d)
        # dist_width[:, 0] = self.tanh(dist_width[:, 0])+0.32
        # dist_width[:, -1] = self.tanh(dist_width[:, -1])+0.7
        # beta=self.tanh(beta)
        # dist_width=(self.tanh(dist_width)+1.)/2
        pose = torch.cat([approach, beta, dist_width], dim=1)

        pose = self.gripper_regressor_layer(pose)

        # if randomization_factor > 0.:
        #     # print(beta)
        #     # pose = self.latent_noise_injection(approach, beta, dist_width, randomization_factor)
        #     pose = self.latent_noise_injection2(pose,randomization_factor)

            # print(pose[:,3:5])

            # print('------',randomization_factor)
        # else:
        #     # print('no randomization')
        #     # pose = torch.cat([approach, beta, dist_width], dim=1)
        #     pose = self.gripper_regressor_layer(pose)
        #     # clamp_mask=pose[:,-1]<1.1
            # pose[:,-1][clamp_mask]=torch.clamp(pose[:,-1][clamp_mask],max=1.0)
            # clamp_mask=pose[:,-2]>-0.1
            # pose[:,-2][clamp_mask]=torch.clamp(pose[:,-2][clamp_mask],min=0.0)

        return pose

class SuctionPartSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.LeakyReLU(negative_slope=suction_sampler_relu_slope) if suction_sampler_relu_slope > 0. else nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.LeakyReLU(negative_slope=suction_sampler_relu_slope) if suction_sampler_relu_slope > 0. else nn.ReLU(),
            nn.Linear(16, 3),
        ).to('cuda')

    def forward(self, representation_2d):
        '''decode'''
        output_2d = self.decoder(representation_2d)

        '''normalize'''
        output_2d = F.normalize(output_2d, dim=1)
        return output_2d

class CollisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.object_collision_decoder = att_res_mlp_LN(in_c1=64, in_c2=7 +1, out_c=1,
                                                    relu_negative_slope=classification_relu_slope,shallow_decoder=True).to(
            'cuda')
        self.bin_collision_decoder = att_res_mlp_LN(in_c1=64, in_c2=7 +1, out_c=1,
                                                relu_negative_slope=classification_relu_slope,shallow_decoder=True).to(
            'cuda')

        self.sigmoid = nn.Sigmoid()

    def forward(self, representation, query):

        objects_collision=self.object_collision_decoder(representation, query)

        bin_collision=self.bin_collision_decoder(representation, query)

        output=torch.cat([objects_collision,bin_collision],dim=1)
        output=self.sigmoid(output)

        return output

class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=generator_backbone_relu_slope).to('cuda')

        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)

        self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        self.gripper_sampler = GripperGraspSampler()
        self.suction_sampler = SuctionPartSampler()

        self.gripper_collision = CollisionNet()

        # self.pre_IN=nn.InstanceNorm2d(64).to('cuda')


        self.suction_quality = att_res_mlp_LN(in_c1=64, in_c2=3, out_c=1,
                                              relu_negative_slope=classification_relu_slope).to(
            'cuda')

        self.shift_affordance = att_res_mlp_LN(in_c1=64, in_c2=5, out_c=1,
                                               relu_negative_slope=classification_relu_slope).to(
            'cuda')

        self.background_detector = att_res_mlp_LN(in_c1=64, in_c2=2, out_c=1,
                                                  relu_negative_slope=classification_relu_slope).to(
            'cuda')

        self.sigmoid = nn.Sigmoid()

    def set_seeds(self, seed):
        if seed is None: return
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def gripper_sample_and_classify_(self,features, depth,approach_direction,randomization_factor,latent_vector):

        gripper_pose = self.gripper_sampler(features, depth,latent_vector, approach_seed=approach_direction,
                                            randomization_factor=randomization_factor)
        '''gripper collision head'''
        gripper_pose_detached = gripper_pose.detach().clone()
        gripper_pose_detached[:, -2:] = torch.clip(gripper_pose_detached[:, -2:], 0., 1.)
        griper_collision_classifier = self.gripper_collision(features, torch.cat([gripper_pose_detached,depth],dim=1))
        return gripper_pose,griper_collision_classifier

    def gripper_sampling_and_quality_inference(self,features, depth,normal_direction,seed,randomization_factor,latent_vector):
        '''gripper parameters'''
        if self.training:
            vertical_direction = torch.zeros_like(normal_direction)
            vertical_direction[:, 2] += 1.0

            self.set_seeds(seed)
            random_mask = torch.rand_like(normal_direction[:, 0]) > 0.5
            gripper_approach_direction = normal_direction.clone().detach()
            gripper_approach_direction[random_mask] = vertical_direction[random_mask]

            gripper_pose,griper_collision_classifier=self.gripper_sample_and_classify_( features, depth, gripper_approach_direction, randomization_factor,latent_vector)
        else:
            '''first forward with approach set to normal'''
            gripper_pose_1,griper_collision_classifier_1=self.gripper_sample_and_classify_( features, depth, normal_direction.detach(), randomization_factor,latent_vector)

            '''second forward with approach set to vertical'''
            vertical_direction = torch.zeros_like(normal_direction)
            vertical_direction[:, 2] += 1.0
            gripper_pose_2,griper_collision_classifier_2=self.gripper_sample_and_classify_( features, depth, vertical_direction, randomization_factor,latent_vector)

            '''rank best approach based on collision avoidance'''
            # collision_with bin
            selective_mask_1 = (griper_collision_classifier_1[:, 0] < 0.5) & (griper_collision_classifier_1[:, 1] < 0.5)
            selective_mask_2 = (griper_collision_classifier_2[:, 0] < 0.5) & (griper_collision_classifier_2[:, 1] < 0.5)

            # check 1: use vertical direction if it solves collision better than normal
            bad_normal_approach_mask = selective_mask_2 & (~selective_mask_1)

            # check 2: if both direction are good at avoiding collision, check for better firmness
            bad_normal_approach_mask = bad_normal_approach_mask | (
                    selective_mask_2 & selective_mask_1 & (gripper_pose_2[:, -2] > gripper_pose_1[:, -2]))

            '''superiority-rank based sampling of poses and associated collision scores'''
            griper_collision_classifier = griper_collision_classifier_1
            griper_collision_classifier[bad_normal_approach_mask] = griper_collision_classifier_2[
                bad_normal_approach_mask]

            gripper_pose = gripper_pose_1
            gripper_pose[bad_normal_approach_mask] = gripper_pose_2[bad_normal_approach_mask]

        return gripper_pose,griper_collision_classifier

    def sample_random_dist(self, dist, randomization_factor=0.1):

        dist_noise = torch.distributions.LogNormal(loc=-1.312, scale=0.505)
        dist_noise = dist_noise.sample((dist.shape[0], 1)).cuda()

        dist = (1.0 - randomization_factor) * dist + dist_noise * randomization_factor
        dist = torch.clip(dist, 0.0, 1.)

        return dist

    def sample_random_width(self, width, randomization_factor=0.1):

        width_noise = torch.distributions.LogNormal(loc=-1.337, scale=0.791)
        width_noise = width_noise.sample((width.shape[0], 1)).cuda()

        width = (1.0 - randomization_factor) * width + width_noise * randomization_factor
        width = torch.clip(width, 0.0, 1.)

        return width
    def ref_generator_forward(self,depth,latent_vector,approach_direction,randomization_factor=0.):
        '''input standardization'''
        depth = standardize_depth(depth)
        features = self.back_bone(depth)

        '''reshape'''
        depth = reshape_for_layer_norm(depth, camera=camera, reverse=False)
        features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        approach_direction = reshape_for_layer_norm(approach_direction, camera=camera, reverse=False)

        '''sampling'''
        gripper_pose = self.gripper_sampler(features, depth,latent_vector, approach_seed=approach_direction,
                                            randomization_factor=randomization_factor)

        # noisy_dist=gripper_pose[:,5:6].detach().clone()
        # noisy_dist=self.dist_noisfication( noisy_dist, randomization_factor=0.3)
        #
        # noisy_width=gripper_pose[:,6:7].detach().clone()
        # noisy_width=self.width_noisfication( noisy_width, randomization_factor=0.3)

        gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=True)
        # noisy_dist = reshape_for_layer_norm(noisy_dist, camera=camera, reverse=True)
        # noisy_width = reshape_for_layer_norm(noisy_width, camera=camera, reverse=True)

        return gripper_pose#,noisy_dist,noisy_width

    def forward(self, depth,latent_vector, seed=None, detach_backbone=False, randomization_factor=0.0):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        # features = self.pre_IN(features)

        # depth_features=features.detach().clone()
        features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        depth = reshape_for_layer_norm(depth, camera=camera, reverse=False)

        # if detach_backbone: features=features.detach()
        # view_features(features)

        # print(features[0])
        # exit()
        '''check exploded values'''
        if self.training:
            max_ = features.max()
            if max_ > 100:
                print(Fore.RED, f'Warning: Res U net outputs high values up to {max_}', Fore.RESET)

        '''Spatial data'''
        if self.spatial_encoding.shape[0] != depth.shape[0]:
            self.spatial_encoding = depth_xy_spatial_data(batch_size=depth.shape[0])
            self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        '''suction direction'''
        normal_direction = self.suction_sampler(features)


        '''gripper sampler and quality inference'''
        gripper_pose,griper_collision_classifier=self.gripper_sampling_and_quality_inference( features, depth, normal_direction, seed, randomization_factor ,latent_vector)

        # view_features(gripper_pose)
        # plt_features(gripper_pose)

        '''suction quality head'''
        normal_direction_detached = normal_direction.detach().clone()
        suction_quality_classifier = self.suction_quality(features, normal_direction_detached)

        '''shift affordance head'''
        shift_query_features = torch.cat([normal_direction_detached, self.spatial_encoding], dim=1)
        shift_affordance_classifier = self.shift_affordance(features, shift_query_features)

        '''background detection'''
        background_class = self.background_detector(features, self.spatial_encoding)

        '''sigmoid'''
        # griper_collision_classifier = self.sigmoid(griper_collision_classifier)
        suction_quality_classifier = self.sigmoid(suction_quality_classifier)
        shift_affordance_classifier = self.sigmoid(shift_affordance_classifier)
        background_class = self.sigmoid(background_class)

        # print(griper_collision_classifier[:,0].max())
        # print(griper_collision_classifier[:,0].min())
        #
        # print(griper_collision_classifier[:,1].max())
        # print(griper_collision_classifier[:,1].min())
        # exit()

        '''reshape'''
        gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=True)
        normal_direction = reshape_for_layer_norm(normal_direction, camera=camera, reverse=True)
        griper_collision_classifier = reshape_for_layer_norm(griper_collision_classifier, camera=camera,
                                                             reverse=True)
        suction_quality_classifier = reshape_for_layer_norm(suction_quality_classifier, camera=camera, reverse=True)
        shift_affordance_classifier = reshape_for_layer_norm(shift_affordance_classifier, camera=camera,
                                                             reverse=True)
        background_class = reshape_for_layer_norm(background_class, camera=camera, reverse=True)

        return gripper_pose, normal_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier, background_class, features


def add_spectral_norm_selective(model, layer_types=(nn.Conv2d, nn.Linear)):
    for name, layer in model.named_children():
        if isinstance(layer, layer_types):
            setattr(model, name, spectral_norm(layer))
        else:
            add_spectral_norm_selective(layer, layer_types)
    return model


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=critic_relu_slope).to('cuda')
        # self.dropout=nn.Dropout2d(0.2)
        # self.pre_IN=nn.InstanceNorm2d(64).to('cuda')


        self.att_block = att_res_mlp_LN(in_c1=64, in_c2=7 +1, out_c=8, relu_negative_slope=critic_relu_slope,
                                        drop_out_ratio=0.0).to('cuda')
        self.att_block2 = att_res_mlp_LN(in_c1=64, in_c2=7 + 1 + 8, out_c=8, relu_negative_slope=critic_relu_slope,
                                         drop_out_ratio=0.0).to('cuda')
        #
        self.decoder = nn.Sequential(
            nn.LayerNorm([16]),
            nn.LeakyReLU(negative_slope=critic_relu_slope) if critic_relu_slope > 0. else nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(16, 1),
        ).to('cuda')

        # add_spectral_norm_selective(self)

    def forward(self, depth, pose, detach_backbone=False):
        '''input standardization'''
        depth = standardize_depth(depth)

        # normalized_depth=self.pre_IN(depth)
        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        # features=self.pre_IN(features)

        features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        depth = reshape_for_layer_norm(depth, camera=camera, reverse=False)
        pose = reshape_for_layer_norm(pose, camera=camera, reverse=False)

        if self.training:
            max_ = features.max()
            if max_ > 100:
                print(Fore.RED, f'Warning: Critic ----- Res U net outputs high values up to {max_}', Fore.RESET)

        # view_features(features_2d)
        query = torch.cat([depth, pose], dim=1)

        '''decode'''
        output1= self.att_block(features,query)  #+ self.att_block1(features,query) * self.tanh(self.att_block2(features,query))
        output2 = self.att_block2(features, torch.cat([query, output1], dim=1))
        output = self.decoder(torch.cat([output1, output2], dim=1))

        # output_2d=torch.cat([output_1,output_2],dim=1)
        output = reshape_for_layer_norm(output, camera=camera, reverse=True)

        return output
