
import torch.nn.functional as F

from GraspAgent_2.model.Backbones import PointNetEncoder
from GraspAgent_2.model.Decoders import ContextGate_1d, ContextGate_2d, res_ContextGate_2d, CSDecoder_2d, \
    ContextGate_2d_2, ContextGate_1d_2, Quality_Net_2d, ContextGate_1d_3, att_res_conv_normalized, multi_film_decoder
from GraspAgent_2.model.YPG_GAN import replace_activations
from GraspAgent_2.model.sparse_encoder import SparseEncoderIN, Encoder3D_IN, Encoder2D_IN
from GraspAgent_2.utils.NN_tools import replace_instance_with_groupnorm
from GraspAgent_2.model.utils import replace_activations, add_spectral_norm_selective

from GraspAgent_2.utils.model_init import init_weights_he_normal, gan_init_with_norms, init_orthogonal, \
    scaled_he_init_all, orthogonal_init_all
from GraspAgent_2.utils.positional_encoding import PositionalEncoding_2d, PositionalEncoding_1d,  \
    LearnableRBFEncoding2D, LearnableRBFEncoding1d

from models.resunet import res_unet
import torch
import torch.nn as nn

Allergo_model_key = 'Allergo_model'

class ALlergoPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.delta = ContextGate_2d(in_c1=64, in_c2= 1, out_c=3,
                                      relu_negative_slope=0., activation=nn.SiLU(), use_sin=False,normalize=True,bias=True).to(
            'cuda')

        self.alpha = ContextGate_2d(in_c1=64, in_c2= 1+3, out_c=3,
                                      relu_negative_slope=0., activation=nn.SiLU(), use_sin=False,normalize=True,bias=True).to(
            'cuda')
        self.beta = ContextGate_2d(in_c1=64, in_c2= 4+3, out_c=2,
                                      relu_negative_slope=0., activation=nn.SiLU(), use_sin=False,normalize=True,bias=True).to(
            'cuda')

        self.fingers=ContextGate_2d(in_c1=64, in_c2=9, out_c=16,
                                          relu_negative_slope=0., activation=nn.SiLU(),use_sin=False,normalize=True,bias=True).to(
            'cuda')

    def forward(self, features,depth,latent_vector):

        delta = self.delta(features,depth)

        alpha = self.alpha(features,torch.cat([depth,delta],dim=1))
        alpha = F.normalize(alpha, dim=1)

        beta = self.beta(features,torch.cat([depth,delta,alpha], dim=1))
        beta = F.normalize(beta, dim=1)

        fingers= self.fingers(features, torch.cat([depth,delta,alpha,beta], dim=1))

        pose = torch.cat([alpha,beta,delta,fingers], dim=1) #28

        return pose

def depth_standardization(depth,mask):
    mean_ = depth[mask].mean()
    depth_ = (depth.clone() - mean_)*10
    return depth_[None,None]

class Allergo_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0., activation=None, IN_affine=False,
                                  activate_skip=False).to('cuda')

        self.back_bone.apply(init_weights_he_normal)

        self.back_bone2_ = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                    relu_negative_slope=0., activation=None, IN_affine=False, activate_skip=False).to(
            'cuda')

        # self.back_bone3_ = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
        #                           relu_negative_slope=0., activation=None, IN_affine=False,activate_skip =False).to('cuda')
        self.back_bone2_.apply(orthogonal_init_all)


        self.AllergoPoseSampler_ = ALlergoPoseSampler()
        self.AllergoPoseSampler_.apply(init_weights_he_normal)

        self.grasp_quality_ =ContextGate_2d( 64, 25, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=True, bias=True, cyclic=False).to('cuda')

        self.grasp_collision_ = ContextGate_2d( 64, 25, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=True, bias=True, cyclic=False).to('cuda')


        self.grasp_collision2 =ContextGate_2d( 64, 25, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=True, bias=True, cyclic=False).to('cuda')


        self.grasp_collision3 = ContextGate_2d( 64, 25, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=True, bias=True, cyclic=False).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(-0.7, dtype=torch.float32, device='cuda'), requires_grad=True)


    def get_grasp_quality(self, depth, target_mask, model_B_poses):
        max_ = 1.3
        min_ = 1.15
        standarized_depth_ = (depth.clone() - min_) / (max_ - min_)

        standarized_depth_ = (standarized_depth_ - 0.5) / 0.5

        with torch.no_grad():
            features2 = self.back_bone2_(standarized_depth_)
            gripper_pose_B=torch.cat([model_B_poses,standarized_depth_],dim=1)

            B_grasp_quality = self.grasp_quality_(features2, gripper_pose_B)

        return B_grasp_quality

    def forward(self, depth, target_mask, latent_vector=None, backbone=None, model_B_poses=None, detach_backbone=False):
        # depth_= depth_standardization(depth[0,0],target_mask[0,0])
        # range=depth.max()-depth.min()
        # center=depth.min()+range/2
        max_ = 1.3
        min_ = 1.15
        standarized_depth_ = (depth.clone() - min_) / (max_ - min_)

        standarized_depth_ = (standarized_depth_ - 0.5) / 0.5
        print('Depth max=', standarized_depth_.max().item(), ', min=', standarized_depth_.min().item(), ', std=',
              standarized_depth_.std().item(), ', mean=', standarized_depth_.mean().item())

        # print(standarized_depth_.mean())
        # print(standarized_depth_.min())
        # print(standarized_depth_.max())
        # print('-----------------------------------------------------------------',standarized_depth_.std())
        # exit()
        # floor_elevation=standarized_depth_[0,0][~target_mask[0,0]].mean()
        # scale_=scale_*5*torch.ones_like(standarized_depth_)

        input = torch.cat([standarized_depth_, target_mask], dim=1)
        # standarized_depth_2=standarized_depth_.clone()
        # standarized_depth_2[latent_vector==0]*=0
        # latent_input = torch.cat([standarized_depth_2, target_mask], dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(input)  # if backbone is None else backbone(input)
                features2 = self.back_bone2_(standarized_depth_)  # *scale
                # features3 = self.back_bone3_(input)#*scale

        else:
            features = self.back_bone(input)  # if backbone is None else backbone(input)
            features2 = self.back_bone2_(standarized_depth_)  # *scale
            # features3 = self.back_bone3_(input)  # *scale

        # features2=torch.cat([features2,scaled_depth_,depth_],dim=1)
        # features=torch.cat([features,scaled_depth_,depth_],dim=1)
        print('G b1 max val= ', features.max().item(), 'mean:', features.mean().item(), ' std:',
              features.std(dim=1).mean().item())
        print('G b2 max val= ', features2.max().item(), 'mean:', features2.mean().item(), ' std:',
              features2.std(dim=1).mean().item())
        # print('G b3 max val= ',features3.max().item(), 'mean:',features3.mean().item(),' std:',features3.std(dim=1).mean().item())

        depth_data = standarized_depth_

        gripper_pose = self.AllergoPoseSampler_(features, depth_data, latent_vector)

        detached_gripper_pose = gripper_pose.detach().clone()


        detached_gripper_pose = torch.cat([detached_gripper_pose, depth_data], dim=1)


        grasp_collision = self.grasp_collision_(features2.detach(), detached_gripper_pose)
        grasp_collision = torch.cat(
            [grasp_collision, self.grasp_collision3(features2.detach(), detached_gripper_pose)], dim=1)
        grasp_collision = torch.cat(
            [grasp_collision, self.grasp_collision2(features2.detach(), detached_gripper_pose)], dim=1)



        grasp_quality = self.grasp_quality_(features2, detached_gripper_pose)


        if model_B_poses is not None:
            gripper_pose_B = torch.cat([model_B_poses, depth_data], dim=1)
            B_grasp_quality = self.grasp_quality_(features2.detach(), gripper_pose_B)
        else:
            B_grasp_quality = None


        return gripper_pose, grasp_quality, grasp_collision, features2.detach(), B_grasp_quality

class Allergo_D(nn.Module):
    def __init__(self):
        super().__init__()


        self.back_bone = SparseEncoderIN().to('cuda')

        self.att_block = ContextGate_1d(in_c1=512, in_c2=24  ).to('cuda')


        self.back_bone.apply(init_weights_he_normal)
        self.att_block.apply(init_weights_he_normal)



    def forward(self, depth, pose, pairs, target_mask, cropped_spheres, latent_vector=None, detach_backbone=False):
        max_ = 1.3
        min_ = 1.15
        standarized_depth_ = (depth.clone() - min_) / (max_ - min_)
        standarized_depth_ = (standarized_depth_ - 0.5) / 0.5


        if detach_backbone:
            with torch.no_grad():
                anchor = self.back_bone(cropped_spheres)
        else:
            anchor = self.back_bone(cropped_spheres)

        print('D max val= ', anchor.max().item(), 'mean:', anchor.mean().item(),
              ' std:',
              anchor.std(dim=1).mean().item())


        depth_data = standarized_depth_


        depth_data = depth_data.flatten(2, 3)
        depth_data_list = []
        xyz_list = []
        for pair in pairs:
            index = pair[0]
            xyz = pair[3].float()
            xyz_list.append(xyz)
            depth_data_list.append(depth_data[:, :, index])


        xyz_list = torch.stack(xyz_list)[:, None, :].repeat(1, 2, 1)  #
        scores = self.att_block(anchor[:, None], pose)

        return None, None, scores

