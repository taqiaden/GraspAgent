import torch
from colorama import Fore
from torch import nn
from torch.nn.utils import spectral_norm

from lib.cuda_utils import cuda_memory_report
from lib.custom_activations import LGRelu
from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN, att_res_mlp_LN_sparse
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth
import torch.nn.functional as F

use_bn=False
use_in=True

# critic_backbone_relu_slope = 0.1
# critic_branch_relu_slope = 0.1
# activation=LGRelu(slope=0.01)
gelu=nn.GELU()


# generator_backbone_relu_slope = 0.0
# gripper_sampler_relu_slope = 0.0

gripper_sampling_module_key = 'gripper_sampling_net'
N_gripper_sampling_module_key = 'N_gripper_sampling_net'
MH_gripper_sampling_module_key = 'MH_gripper_sampling_net'


class GripperGraspSampler(nn.Module):
    def __init__(self,use_sig=False):
        super().__init__()

        self.beta_decoder = att_res_mlp_LN_sparse(in_c1=64, in_c2=3 , out_c=2,
                                           relu_negative_slope=0.1,use_sigmoid=True,activation=gelu).to(
            'cuda')
        self.dist_width_decoder = att_res_mlp_LN_sparse(in_c1=64, in_c2=5, out_c=2,
                                                 relu_negative_slope=0.1,use_sigmoid=True,activation=gelu).to(
            'cuda')

        self.sig=nn.Sigmoid()
        self.use_sig=use_sig

    def forward(self, representation_2d,  approach=None  ):
        approach=F.normalize(approach, dim=1).detach()

        beta = self.beta_decoder(representation_2d,approach)
        beta=F.normalize(beta, dim=1)
        dist_width = self.dist_width_decoder( representation_2d,torch.cat([ approach, beta.detach()], dim=1))
        if self.use_sig:
            dist_width=self.sig(dist_width)

        pose = torch.cat([approach, beta,dist_width], dim=1)

        return pose

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.1,activation=gelu).to('cuda')

        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)

        self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        self.gripper_sampler = GripperGraspSampler()

    def forward(self, depth,approach, detach_backbone=False):
        '''input standardization'''
        depth = standardize_depth(depth)
        # cuda_memory_report()

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        # cuda_memory_report()


        features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        approach= reshape_for_layer_norm(approach, camera=camera, reverse=False)
        '''check exploded values'''
        if self.training:
            max_ = features.max()
            if max_ > 100:
                print(Fore.RED, f'Warning: Res U net outputs high values up to {max_}', Fore.RESET)
        gripper_pose=self.gripper_sampler(features,approach)
        # view_features(gripper_pose)
        # plt_features(gripper_pose)

        '''reshape'''
        gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=True)

        # cuda_memory_report()

        return gripper_pose

class MH_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.,activation=gelu).to('cuda')

        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)

        self.spatial_encoding = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        self.gripper_sampler1 = GripperGraspSampler()
        self.gripper_sampler2 = GripperGraspSampler()


    def forward(self, depth,approach1,approach2, detach_backbone=False):
        '''input standardization'''
        depth = standardize_depth(depth)
        # cuda_memory_report()

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        # cuda_memory_report()


        features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        approach1= reshape_for_layer_norm(approach2, camera=camera, reverse=False)
        approach2= reshape_for_layer_norm(approach2, camera=camera, reverse=False)

        '''check exploded values'''
        if self.training:
            max_ = features.max()
            if max_ > 100:
                print(Fore.RED, f'Warning: Res U net outputs high values up to {max_}', Fore.RESET)
        gripper_pose1=self.gripper_sampler1(features,approach1)
        gripper_pose2=self.gripper_sampler2(features,approach2)

        # view_features(gripper_pose)
        # plt_features(gripper_pose)

        '''reshape'''
        gripper_pose1 = reshape_for_layer_norm(gripper_pose1, camera=camera, reverse=True)
        gripper_pose2 = reshape_for_layer_norm(gripper_pose2, camera=camera, reverse=True)

        # cuda_memory_report()

        return (gripper_pose1,gripper_pose2)

def add_spectral_norm_selective(model, layer_types=(nn.Conv2d, nn.Linear)):
    for name, layer in model.named_children():
        if isinstance(layer, layer_types):
            setattr(model, name, spectral_norm(layer))
        else:
            add_spectral_norm_selective(layer, layer_types)
    return model

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.1,activation=gelu).to('cuda')
        self.att_block = att_res_mlp_LN(in_c1=64, in_c2=7, out_c=1, relu_negative_slope=0.1,
                                        drop_out_ratio=0., shallow_decoder=False, use_sigmoid=True,activation=gelu).to(
            'cuda')

        add_spectral_norm_selective(self)

    def forward(self, depth, pose, detach_backbone=False):
        '''input standardization'''
        depth = standardize_depth(depth)
        # cuda_memory_report()

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)
        # cuda_memory_report()

        features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        pose = reshape_for_layer_norm(pose, camera=camera, reverse=False)

        # if self.training:
        #     max_ = features.max()
        #     if max_ > 100:
        #         print(Fore.RED, f'Warning: Critic ----- Res U net outputs high values up to {max_}', Fore.RESET)

        '''decode'''
        output= self.att_block(features,pose)

        output = reshape_for_layer_norm(output, camera=camera, reverse=True)
        # cuda_memory_report()

        return output

