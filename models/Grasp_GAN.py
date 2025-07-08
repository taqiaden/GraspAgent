import torch
from colorama import Fore
from torch import nn
from torch.nn.utils import spectral_norm
# from torch.nn.utils.parametrizations import spectral_norm

from lib.cuda_utils import cuda_memory_report
from lib.custom_activations import LGRelu
from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_conv_normalized, att_conv_normalized, att_res_conv_norma_free, \
    att_conv_normalize_free, att_conv_I_N
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth
import torch.nn.functional as F

use_bn=False
use_in=True

silu=nn.SiLU()

relu=nn.ReLU()

class Mish(nn.Module):
    def forward(self,x):
        return x* torch.tanh(F.softplus(x))

mish=Mish()


gripper_sampling_module_key = 'gripper_sampling_net'
N_gripper_sampling_module_key = 'N_gripper_sampling_net'
MH_gripper_sampling_module_key = 'MH_gripper_sampling_net'

class GripperGraspSampler(nn.Module):
    def __init__(self,relu_negative_slope=0.0,decoder=None,activation=silu):
        super().__init__()
        # self.approach_decoder = att_res_mlp_IN2(in_c1=64, in_c2=3 , out_c=3,
        #                                    relu_negative_slope=0.,activation=activation).to(
        #     'cuda')
        # self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)

        if decoder is None: decoder=att_conv_normalized

        self.beta_decoder = decoder(in_c1=64, in_c2=3 , out_c=2,
                                           relu_negative_slope=relu_negative_slope,activation=activation).to(
            'cuda')
        self.width_decoder = decoder(in_c1=64, in_c2=5, out_c=1,
                                                 relu_negative_slope=relu_negative_slope,activation=activation).to(
            'cuda')
        self.dist_decoder = decoder(in_c1=64, in_c2=6, out_c=1,
                                                   relu_negative_slope=relu_negative_slope, activation=activation).to(
            'cuda')

        # self.ln = nn.LayerNorm([64]).to('cuda')

        self.sig=nn.Sigmoid()

    def forward(self, representation_2d,  approach=None  ):
        # representation_2d = reshape_for_layer_norm(representation_2d, camera=camera, reverse=True)
        # if approach is not None:approach = reshape_for_layer_norm(approach, camera=camera, reverse=True)

        # normalized_features=self.ln(representation_2d)
        #
        # approach=F.normalize(approach, dim=1).detach()
        # approach_delta = self.approach_decoder(representation_2d,approach)
        # print(self.scale)
        # exit()

        # print(approach)
        # print(approach_delta)

        # approach=F.normalize(approach+self.scale*approach_delta, dim=1)
        approach=F.normalize(approach, dim=1)

        beta = self.beta_decoder(representation_2d,approach)
        beta=F.normalize(beta, dim=1)

        # print(approach.shape)
        # print(beta.shape)
        width = self.width_decoder( representation_2d,torch.cat([ approach, beta], dim=1))
        dist = self.dist_decoder( representation_2d,torch.cat([ approach, beta,width], dim=1))

        # width=self.sig(width)
        # dist=self.sig(dist)

        pose = torch.cat([approach, beta,dist,width], dim=1)

        # pose = reshape_for_layer_norm(pose, camera=camera, reverse=False)


        return pose

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.0,activation=silu).to('cuda')

        # self.sig = nn.Sigmoid()
        self.gripper_sampler = GripperGraspSampler(relu_negative_slope=0.0,activation=silu,decoder=att_conv_normalized)



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

        # features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        # approach= reshape_for_layer_norm(approach, camera=camera, reverse=False)
        # '''check exploded values'''
        # if self.training:
        #     max_ = features.max()
        #     if max_ > 100:
        #         print(Fore.RED, f'Warning: Res U net outputs high values up to {max_}', Fore.RESET)
        # print(depth.shape)
        # print(features.shape)
        # print(approach.shape)

        gripper_pose=self.gripper_sampler(features,approach)

        # gripper_pose[:,5:,...] = self.sig(gripper_pose[:,5:,...])
        # view_features(gripper_pose)
        # plt_features(gripper_pose)

        # '''reshape'''
        # gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=True)

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
                                  relu_negative_slope=0.2,activation=None).to('cuda')
        self.att_block_ = att_conv_normalize_free(in_c1=64, in_c2=7 , out_c=1,
                                           relu_negative_slope=0.2,activation=None,drop_out_ratio=0.,shallow=True).to(
            'cuda')

        # self.ln = nn.LayerNorm([64]).to('cuda')

        # self.sig=nn.Sigmoid()

        # self.drop_out=nn.Dropout(0.5)

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

        # features = reshape_for_layer_norm(features, camera=camera, reverse=False)
        # pose = reshape_for_layer_norm(pose, camera=camera, reverse=False)

        # features=self.drop_out(features)


        # normalized_features = self.ln(features)


        '''decode'''
        output= self.att_block_(features,pose)
        # output2= self.att_block_2(features,pose)
        # output=self.scale*output+output2*self.scale2

        # output = reshape_for_layer_norm(output, camera=camera, reverse=True)

        return output

