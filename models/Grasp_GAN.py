import torch
from colorama import Fore
from torch import nn
from torch.nn.utils import spectral_norm

# from torch.nn.utils.parametrizations import spectral_norm

from lib.cuda_utils import cuda_memory_report
from lib.custom_activations import LGRelu
from lib.models_utils import reshape_for_layer_norm
from models.decoders import LayerNorm2D, att_conv_LN, att_conv_norm_free, att_res_conv_normalized, \
    att_res_conv_normalized_free, att_conv_normalized_free, att_conv_LN_normalize, att_conv_normalized, att_conv_LN2, \
    film_conv_normalized, att_conv_LN3, att_conv_normalized128, film_conv_normalized_128, att_res_conv_normalized2, \
    att_conv_normalized2, att_conv_normalized_free2, sine, ParameterizedSine
from models.point_net_grasp_gan import DepthPointNetAdapter
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
import torch.nn.functional as F


def depth_standardization(depth,mask):
    # mean_ = depth[mask].mean()
    mean_=1265
    depth_ = (depth.clone() - mean_) / 30
    depth_[~mask] = 0.
    return depth_

from visualiztion import view_features

use_bn=False
use_in=True

silu=nn.SiLU()
tanh=nn.Tanh()
relu=nn.ReLU()

class Mish(nn.Module):
    def forward(self,x):
        return x* torch.tanh(F.softplus(x))

mish=Mish()


gripper_sampling_module_key = 'gripper_sampling_net'
N_gripper_sampling_module_key = 'N_gripper_sampling_net'
MH_gripper_sampling_module_key = 'MH_gripper_sampling_net'

class GripperGraspSampler(nn.Module):
    def __init__(self):
        super().__init__()
        # self.approach_decoder = att_res_mlp_IN2(in_c1=64, in_c2=3 , out_c=3,
        #                                    relu_negative_slope=0.,activation=activation).to(
        #     'cuda')
        # self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.approach_gate=SpatialGate(64)
        self.approach = nn.Sequential(
            # LayerNorm2D(64),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            LayerNorm2D(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            LayerNorm2D(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1),
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(0.15, dtype=torch.float32, device='cuda'), requires_grad=True)

        # self.approach = att_conv_LN(in_c1=64, in_c2=3 , out_c=2,
        #                                    relu_negative_slope=0.,activation=mish).to(
        #     'cuda')

        self.beta_decoder = att_conv_normalized(in_c1=64, in_c2=3, out_c=2,
                                                    relu_negative_slope=0., activation=None).to(
            'cuda')

        self.dist_width_decoder = att_conv_normalized(in_c1=64, in_c2=3 + 2, out_c=2,
                                              relu_negative_slope=0., activation=None).to(
            'cuda')

        # self.sig=nn.Sigmoid()

        # self.decoder_mlp = nn.Sequential(
        #     # LayerNorm2D(64),
        #     nn.Conv2d(64+3, 64, kernel_size=1, bias=False),
        #     LayerNorm2D(64),
        #     tanh,
        #     nn.Conv2d(64, 32, kernel_size=1, bias=False),
        #     LayerNorm2D(32),
        #     tanh,
        #     nn.Conv2d(32, 4, kernel_size=1),
        #     tanh
        # ).to('cuda')

        # self.sp=nn.Softplus()

    def forward(self, representation_2d,  approach=None  ):
        # representation_2d = reshape_for_layer_norm(representation_2d, camera=camera, reverse=True)
        # if approach is not None:approach = reshape_for_layer_norm(approach, camera=camera, reverse=True)

        # normalized_features=self.ln(representation_2d)
        #
        # approach=F.normalize(approach, dim=1).detach()
        # approach_delta = self.approach_decoder(representation_2d,approach)
        # print(self.scale)
        # exit()

        # print(approach)`
        # print(approach_delta)

        # approach=F.normalize(approach+self.scale*approach_delta, dim=1)
        verticle=torch.zeros_like(representation_2d[:,0:3])
        verticle[:,-1]+=1.
        # approach=verticle
        approach=self.approach(representation_2d)*self.scale+verticle

        approach=F.normalize(approach, dim=1)

        # ones_=torch.ones_like(approach[:,0:1])
        # approach=torch.concatenate([approach,ones_],dim=1)
        beta = self.beta_decoder(representation_2d,approach)
        # beta=beta__approach[:,0:2]
        # approach=beta__approach[:,2:]

        # beta_dist_width=self.decoder_mlp(torch.cat([representation_2d,approach],dim=1))

        beta = F.normalize(beta, dim=1)

        # beta=beta_dist_width[:,0:2]
        dist_width=self.dist_width_decoder(representation_2d,torch.cat([approach,beta],dim=1))
        # print(dist_width[:,0])

        # beta=self.tanh(beta)
        # dist=dist_width[:0:1]
        # width=dist_width[:1:2]


        # dist=self.sig(dist)

        pose = torch.cat([approach, beta,dist_width], dim=1)
        # pose = reshape_for_layer_norm(pose, camera=camera, reverse=False)

        return pose


class SpatialGate(nn.Module):
    """Spatial attention gate to weight backbone features."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),  # Reduce to 1 channel
        ).to('cuda')

    def forward(self, x):
        # x: Backbone feature map [B, C, H, W]
        attention =(self.conv(x)) # [B, 1, H, W]
        attention=torch.sigmoid(attention)
        # attention = attention.unflatten(1, (2, -1))
        # attention = F.softmax(attention, dim=1)
        # attention = attention.flatten(1, 2)
        return x * attention  # Gated features

class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (x.pow(2).mean(dim=1, keepdim=True) + self.eps).sqrt()
class norm_free(nn.Module):
    def __init__(self,w=None):
        super().__init__()
    def forward(self, x ):
        return x

class GripperGraspSampler3(nn.Module):
    def __init__(self):
        super().__init__()
        # self.approach_decoder = att_res_mlp_IN2(in_c1=64, in_c2=3 , out_c=3,
        #                                    relu_negative_slope=0.,activation=activation).to(
        #     'cuda')
        # self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.approach_gate=SpatialGate(64)
        self.beta_gate=SpatialGate(64)
        self.dist_width_gate=SpatialGate(64)

        self.approach = nn.Sequential(
            # LayerNorm2D(64),
            nn.Conv2d(64, 32, kernel_size=1),
            # LayerNorm2D(32),
            # nn.Dropout2d(0.),
            ParameterizedSine(),
            nn.Conv2d(32, 16, kernel_size=1),
            # LayerNorm2D(16),
            silu,
            nn.Conv2d(16, 3, kernel_size=1),
        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.beta_decoder = att_conv_normalized2(in_c1=64, in_c2=3+1, out_c=2,
                                                    relu_negative_slope=0., activation=silu,use_sin=True).to(
            'cuda')


        self.width = att_conv_normalized2(in_c1=64, in_c2=3 + 2+1, out_c=1*3,
                                              relu_negative_slope=0., activation=silu,normalization=norm_free).to(
            'cuda')

        self.dist = att_conv_normalized2(in_c1=64, in_c2=3 + 2+1+1, out_c=1*3,
                                              relu_negative_slope=0., activation=silu,normalization=norm_free).to(
            'cuda')
        self.sig=nn.Sigmoid()

        # add_spectral_norm_selective(self.approach)
        # add_spectral_norm_selective(self.beta_decoder)
        # add_spectral_norm_selective(self.width)
        # add_spectral_norm_selective(self.dist)


        # self.decoder_mlp = nn.Sequential(
        #     # LayerNorm2D(64),
        #     nn.Conv2d(64+3, 64, kernel_size=1, bias=False),
        #     LayerNorm2D(64),
        #     tanh,
        #     nn.Conv2d(64, 32, kernel_size=1, bias=False),
        #     LayerNorm2D(32),
        #     tanh,
        #     nn.Conv2d(32, 4, kernel_size=1),
        #     tanh
        # ).to('cuda')
        self.tanh=nn.Tanh()
        self.sp=nn.Softplus()
        self.softplus=nn.Softplus()

    def forward(self, representation_2d,  depth ):

        verticle=torch.zeros_like(representation_2d[:,0:3])
        verticle[:,-1]+=1.
        # approach=verticle

        approach=self.approach(representation_2d)
        approach=F.normalize(approach, dim=1)
        approach=approach*self.scale+verticle
        approach=F.normalize(approach, dim=1)
        # ones_=torch.ones_like(approach[:,0:1])
        # approach=torch.concatenate([approach,ones_],dim=1)
        beta = self.beta_decoder(representation_2d,torch.cat([approach,depth],dim=1))
        # beta=beta__approach[:,0:2]
        # approach=beta__approach[:,2:]

        # beta_dist_width=self.decoder_mlp(torch.cat([representation_2d,approach],dim=1))
        beta = F.normalize(beta, dim=1)

        # beta=beta_dist_width[:,0:2]
        width=self.width(representation_2d,torch.cat([approach,beta,depth],dim=1))
        width = F.normalize(width, p=2, dim=1).sum(dim=1,keepdim=True)

        dist=self.dist(representation_2d,torch.cat([approach,beta,width,depth],dim=1))
        dist = F.normalize(dist, p=2, dim=1).sum(dim=1,keepdim=True)

        # dist=self.softplus(dist)
        # width=1-self.softplus(-width)

        # beta=self.tanh(beta)
        # dist=dist_width[:0:1]
        # width=dist_width[:1:2]


        # dist_width=(self.tanh(dist_width)+1)/2

        pose = torch.cat([approach, beta,dist,width], dim=1)
        # pose = reshape_for_layer_norm(pose, camera=camera, reverse=False)

        return pose

def get_auto_groupnorm(num_channels, max_groups=8):
    # Find the largest number of groups <= max_groups that divides num_channels
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=num_channels, affine=True).to('cuda')
    # fallback to LayerNorm behavior
    return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=True).to('cuda')

def replace_instance_with_groupnorm(module, max_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.InstanceNorm2d):
            gn = get_auto_groupnorm(child.num_features, max_groups=max_groups)
            setattr(module, name, gn)
        else:
            replace_instance_with_groupnorm(child, max_groups=max_groups)

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=None, Instance_norm=True,
                                  relu_negative_slope=0.0,activation=silu,IN_affine=False).to('cuda')

        # add_spectral_norm_selective(self.back_bone)

        # add_spectral_norm_selective(self)

        # self.back_bone =DepthPointNetAdapter().to('cuda')
        # replace_relu_with_mish(self)

        # replace_instance_with_groupnorm(self.back_bone,max_groups=16)
        # self.sig = nn.Sigmoid()


        self.gripper_sampler = GripperGraspSampler3()
        # self.drop_out=nn.Dropout2d(0.1)

        # self.gripper_sampler2 = GripperGratt_conv_normalize_freeaspSampler(relu_negative_slope=0.0,activation=silu,decoder=att_conv_normalize_free)


    def forward(self, depth, detach_backbone=False):
        '''input standardization'''
        depth = standardize_depth(depth)
        # cuda_memory_report()

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        # print(features)
        print('G max_features_output=',features.max().item(), ', min=',features.min().item(),', mean=',features.mean().item())

        # features=self.drop_out(features)
        # view_features(features, hide_axis=True,reshape=False)

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

        # z=torch.randn_like(features[:,0:16])
        # features=torch.concatenate([features,z],dim=1)
        gripper_pose=self.gripper_sampler(features)
        # gripper_pose_plus=self.gripper_sampler2(features,approach)

        # gripper_pose[:,5:,...] = self.sig(gripper_pose[:,5:,...])
        # view_features(gripper_pose)
        # plt_features(gripper_pose)

        # '''reshape'''
        # gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=True)

        return gripper_pose
class G_PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
        #                           relu_negative_slope=0.,activation=mish,IN_affine=False).to('cuda')

        self.back_bone =DepthPointNetAdapter().to('cuda')
        # replace_relu_with_mish(self)

        # replace_instance_with_groupnorm(self,max_groups=8)
        # self.sig = nn.Sigmoid()


        self.gripper_sampler = GripperGraspSampler2()

        # self.gripper_sampler2 = GripperGratt_conv_normalize_freeaspSampler(relu_negative_slope=0.0,activation=silu,decoder=att_conv_normalize_free)


    def forward(self, depth, detach_backbone=False):
        '''input standardization'''
        depth = standardize_depth(depth)
        # cuda_memory_report()

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        # view_features(features, hide_axis=True,reshape=False)

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

        # z=torch.randn_like(features[:,0:16])
        # features=torch.concatenate([features,z],dim=1)
        gripper_pose=self.gripper_sampler(features)
        # gripper_pose_plus=self.gripper_sampler2(features,approach)

        # gripper_pose[:,5:,...] = self.sig(gripper_pose[:,5:,...])
        # view_features(gripper_pose)
        # plt_features(gripper_pose)

        # '''reshape'''
        # gripper_pose = reshape_for_layer_norm(gripper_pose, camera=camera, reverse=True)

        return gripper_pose


def add_spectral_norm_selective(model, layer_types=(nn.Conv2d, nn.Linear)):
    for name, layer in model.named_children():
        if isinstance(layer, layer_types):
            setattr(model, name, spectral_norm(layer, name='weight'))
        else:
            add_spectral_norm_selective(layer, layer_types)
    return model
def replace_relu_with_mish(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, Mish())
        else:
            replace_relu_with_mish(module)
def depth_standardization(depth,mask):
    mean_ = depth[mask].mean()

    depth_ = (depth.clone() - mean_) / 30
    depth_[~mask] = 0.

    return depth_
class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=2, Batch_norm=None, Instance_norm=True,
                                  relu_negative_slope=0.2,activation=None,IN_affine=False).to('cuda')

        replace_instance_with_groupnorm(self.back_bone, max_groups=16)
        # self.back_bone.SN_on_encoder()
        # self.back_bone.GN_on_decoder()

        self.att_block_ = att_conv_normalized_free2(in_c1=64, in_c2=7+1 , out_c=1,
                                           relu_negative_slope=0.1,activation=silu,drop_out_ratio=0.).to(
            'cuda')

        # add_spectral_norm_selective(self.back_bone)
        # add_spectral_norm_selective(self.att_block_)

    def get_features(self,depth):
        depth = standardize_depth(depth)
        features = self.back_bone(depth)
        return features
    def forward(self, depth, pose,mask, detach_backbone=False):

        '''input standardization'''
        depth = depth_standardization(depth,mask)
        # depth=depth.repeat(2,1,1,1)
        inputs=torch.cat([depth,mask],dim=1)

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(inputs)
        else:
            features = self.back_bone(inputs)

        features=features.repeat(2,1,1,1)
        depth=depth.repeat(2,1,1,1)



        print('D max_features_output=',features.max().item(), ', min=',features.min().item(),', std=',features.std().item(),', mean=',features.mean().item())



        '''decode'''
        output= self.att_block_(features,torch.cat([pose,depth],dim=1))


        return output, None

