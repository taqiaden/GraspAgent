import torch
from colorama import Fore
from torch import nn
from torch.nn.utils import spectral_norm
# from torch.nn.utils.parametrizations import spectral_norm

from lib.cuda_utils import cuda_memory_report
from lib.custom_activations import LGRelu
from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN, att_res_mlp_LN_sparse, att_res_mlp_LN_SwiGLUBBlock
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth
import torch.nn.functional as F

use_bn=False
use_in=True

silu=nn.SiLU()

gripper_sampling_module_key = 'gripper_sampling_net'
N_gripper_sampling_module_key = 'N_gripper_sampling_net'
MH_gripper_sampling_module_key = 'MH_gripper_sampling_net'


class GripperGraspSampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.beta_decoder = att_res_mlp_LN_sparse(in_c1=64, in_c2=3 , out_c=2,
                                           relu_negative_slope=0.,activation=silu).to(
            'cuda')
        self.width_decoder = att_res_mlp_LN_sparse(in_c1=64, in_c2=5, out_c=1,
                                                 relu_negative_slope=0.,activation=silu).to(
            'cuda')
        self.dist_decoder = att_res_mlp_LN_sparse(in_c1=64, in_c2=6, out_c=1,
                                                   relu_negative_slope=0., activation=silu).to(
            'cuda')

        self.sig=nn.Sigmoid()

    def forward(self, representation_2d,  approach=None  ):
        approach=F.normalize(approach, dim=1).detach()

        beta = self.beta_decoder(representation_2d,approach)
        beta=F.normalize(beta, dim=1)
        width = self.width_decoder( representation_2d,torch.cat([ approach, beta], dim=1))
        dist = self.dist_decoder( representation_2d,torch.cat([ approach, beta,width], dim=1))


        pose = torch.cat([approach, beta,dist,width], dim=1)

        return pose

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.,activation=silu).to('cuda')

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

        return gripper_pose


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
                                  relu_negative_slope=0.,activation=silu).to('cuda')
        self.att_block_ = att_res_mlp_LN_sparse(in_c1=64, in_c2=7 , out_c=1,
                                           relu_negative_slope=0.,activation=silu,drop_out_ratio=0.1).to(
            'cuda')
        self.drop_out=nn.Dropout(0.1)

        add_spectral_norm_selective(self)

    def forward(self, depth, pose, detach_backbone=False):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        pose = reshape_for_layer_norm(pose, camera=camera, reverse=False)

        features=self.drop_out(features)


        '''decode'''
        output= self.att_block_(features,pose)
        # output2= self.att_block_2(features,pose)
        # output=self.scale*output+output2*self.scale2

        output = reshape_for_layer_norm(output, camera=camera, reverse=True)

        return output

