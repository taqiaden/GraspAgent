
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


SH_model_key = 'SH_model'


class siren(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(w0, dtype=torch.float32, device='cuda'), requires_grad=True)

    def forward(self,x):
        # print(x)
        # print('scale=',self.w0)
        x=torch.sin(x*self.w0)
        # print(x)
        return x

class SHPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()

        # self.quat = film_fusion_2d(in_c1=64, in_c2=10, out_c=4,
        #                                   relu_negative_slope=0., activation=None,use_sin=True).to(
        #     'cuda')

        # self.quat = nn.Sequential(
        #     nn.Conv2d(64+11, 64, kernel_size=1),
        #     Parameterizedsiren(),
        #     nn.Conv2d(64, 32, kernel_size=1),
        #     Parameterizedsiren(),
        # ).to('cuda')

        # self.quat = film_fusion_2d(in_c1=64, in_c2= 1, out_c=4,
        #                               relu_negative_slope=0.1, activation=None, use_sin=False,bias=False).to(
        #     'cuda')
        # self.transition=film_fusion_2d(in_c1=64, in_c2=1+10, out_c=1,
        #                                   relu_negative_slope=0.1, activation=nn.SiLU(),use_sin=False,bias=False).to(
        #     'cuda')
        # self.fingers_=film_fusion_2d(in_c1=64, in_c2=1+10+1, out_c=3,
        #                                   relu_negative_slope=0.1, activation=None,use_sin=False,bias=False).to(
        #     'cuda')

        # self.quat_ = ContextGate_2d(in_c1=64, in_c2= 1+8, out_c=4,
        #                               relu_negative_slope=0.1, activation=None, use_sin=False,normalize=False).to(
        #     'cuda')
        self.alpha = ContextGate_2d(in_c1=64, in_c2= 1, out_c=3,
                                      relu_negative_slope=0., activation=nn.SiLU(), use_sin=False,normalize=False,bias=True).to(
            'cuda')

        self.beta = ContextGate_2d(in_c1=64, in_c2= 1+3, out_c=2,
                                      relu_negative_slope=0., activation=nn.SiLU(), use_sin=False,normalize=False,bias=True).to(
            'cuda')
        self.transition_=ContextGate_2d(in_c1=64, in_c2=1+5, out_c=1,
                                          relu_negative_slope=0., activation=nn.SiLU(),use_sin=False,normalize=False,bias=True).to(
            'cuda')

        self.fingers=ContextGate_2d(in_c1=64, in_c2=1+5+1, out_c=3,
                                          relu_negative_slope=0., activation=nn.SiLU(),use_sin=False,normalize=False,bias=True).to(
            'cuda')

        self.biases = nn.Parameter(torch.tensor([0.,0.,0.,0.], dtype=torch.float32, device='cuda'), requires_grad=True).reshape(1,-1,1,1)



    def forward(self, features,depth,latent_vector):
        # encoded_depth=self.depth_encoding(depth)
        # quat = self.quat_(features,torch.cat([depth,latent_vector], dim=1))
        # # quat=expmap_to_quat_map_2d(quat)
        # quat = torch.where(quat[:,0:1] >= 0, quat, -quat)
        # quat = F.normalize(quat, dim=1)

        alpha = self.alpha(features,depth)
        alpha = F.normalize(alpha, dim=1)

        beta = self.beta(features,torch.cat([depth,alpha], dim=1).detach())
        beta = F.normalize(beta, dim=1)

        # encoded_quat=sign_invariant_quat_encoding_2d(quat)
        # encoded_quat=self.quat_encoding(encoded_quat)
        # quat_embedding=self.normalize_quat_embedding(quat_embedding)

        # fingers_embedding=self.normalize_fingers_embedding(fingers_embedding)
        transition=self.transition_(features,torch.cat([alpha,beta,depth], dim=1).detach())
        transition=F.tanh(transition)+self.biases[:,0:1]
        # transition=(F.normalize(transition,p=2,dim=1).sum(dim=1,keepdim=True)+math.sqrt(3))/(2*math.sqrt(3))
        # transition=(transition-0.5)*3
        # encoded_transition=self.transition_encoding(transition)
        # transition=F.tanh(transition)
        # encoded_transition=self.pos_encoder(transition)
        fingers= self.fingers(features, torch.cat([alpha,beta,transition,depth], dim=1).detach())
        fingers=F.tanh(fingers)+self.biases[:,1:]

        # fingers=F.tanh(fingers)/2
        # fingers=F.normalize(fingers.unflatten(1,(3,3)),p=2,dim=1).sum(dim=1)/(2*math.sqrt(3))
        # fingers = F.normalize(fingers, dim=1)
        # finger1=fingers[:,0:3]
        # finger2=fingers[:,2:5]
        # finger3=fingers[:,[4,5,0]]
        # finger1=F.normalize(finger1,p=2, dim=1,eps=1e-8).sum(dim=1,keepdim=True)
        # finger2=F.normalize(finger2,p=2, dim=1,eps=1e-8).sum(dim=1,keepdim=True)
        # finger3=F.normalize(finger3,p=2, dim=1,eps=1e-8).sum(dim=1,keepdim=True)
        # fingers=torch.cat([finger1,finger2,finger3],dim=1)/math.sqrt(3)

        # fingers_scale= self.fingers_scale(features, torch.cat([alpha,beta,transition,fingers,depth], dim=1))
        # fingers_scale=F.normalize(fingers_scale,p=2,dim=1).sum(dim=1,keepdim=True)
        #
        # fingers=fingers*(1+fingers_scale)
        # fingers=F.sigmoid(fingers)

        # quat=self.proj_quat_(quat_embedding)
        # fingers=self.proj_fingers(fingers_embedding)
        # transition=self.proj_transition(transition_embedding)

        pose = torch.cat([alpha,beta,fingers,transition], dim=1)
        # print(pose.shape)
        # print(fingers.shape)

        # exit()
        return pose

def depth_standardization(depth,mask):
    mean_ = depth[mask].mean()
    depth_ = (depth.clone() - mean_)*10
    return depth_[None,None]

class SH_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0., activation=nn.ReLU(), IN_affine=False,
                                  activate_skip=False).to('cuda')

        # gain = torch.nn.init.calculate_gain('leaky_relu', 0.1)
        self.back_bone.apply(init_weights_he_normal)
        # add_spectral_norm_selective(self.back_bone)

        # gan_init_with_norms(self.back_bone)

        self.back_bone2_ = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                    relu_negative_slope=0., activation=None, IN_affine=False, activate_skip=False).to(
            'cuda')
        # replace_instance_with_groupnorm(self.back_bone2_, max_groups=16)


        self.SH_PoseSampler = SHPoseSampler()

        self.back_bone2_.apply(init_weights_he_normal)


        self.grasp_quality_=ContextGate_2d( 64, 10, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=False, bias=True, cyclic=False).to('cuda')



        self.grasp_collision_ =ContextGate_2d( 64, 10, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=False, bias=True, cyclic=False).to('cuda')

        self.grasp_collision2 = ContextGate_2d( 64, 10, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=False, bias=True, cyclic=False).to('cuda')

        self.grasp_collision3 = ContextGate_2d( 64, 10, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=False, bias=True, cyclic=False).to('cuda')

        self.grasp_quality_.apply(init_weights_he_normal)
        self.grasp_collision_.apply(init_weights_he_normal)
        self.grasp_collision2.apply(init_weights_he_normal)
        self.grasp_collision3.apply(init_weights_he_normal)


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
        max_ = 1.3
        min_ = 1.15
        standarized_depth_ = (depth.clone() - min_) / (max_ - min_)

        standarized_depth_ = (standarized_depth_ - 0.5) / 0.5
        print('Depth max=', standarized_depth_.max().item(), ', min=', standarized_depth_.min().item(), ', std=',
              standarized_depth_.std().item(), ', mean=', standarized_depth_.mean().item())


        input = torch.cat([standarized_depth_, target_mask], dim=1)
        # standarized_depth_2=standarized_depth_.clone()
        # standarized_depth_2[latent_vector==0]*=0
        # latent_input = torch.cat([standarized_depth_2, target_mask], dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(standarized_depth_)  # if backbone is None else backbone(input)
                features2 = self.back_bone2_(standarized_depth_)  # *scale
                # features3 = self.back_bone3_(input)#*scale

        else:
            features = self.back_bone(standarized_depth_)  # if backbone is None else backbone(input)
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

        gripper_pose = self.SH_PoseSampler(features, depth_data, latent_vector)

        detached_gripper_pose = gripper_pose.detach().clone()


        detached_gripper_pose = torch.cat([detached_gripper_pose, depth_data], dim=1)


        grasp_collision = self.grasp_collision_(features2.detach(), detached_gripper_pose)
        grasp_collision = torch.cat(
            [grasp_collision, self.grasp_collision3(features2.detach(), detached_gripper_pose)], dim=1)
        grasp_collision = torch.cat(
            [grasp_collision, self.grasp_collision2(features2.detach(), detached_gripper_pose)], dim=1)

        # grasp_collision=self.grasp_collision_(features2,rotation=encoded_quat,transition=torch.cat([transition,depth_data],dim=1),fingers=None)

        # grasp_collision=self.sig(grasp_collision)

        grasp_quality = self.grasp_quality_(features2, detached_gripper_pose)

        # grasp_quality=grasp_quality-grasp_quality[~target_mask].mean()+self.bias
        # grasp_quality=grasp_quality*self.scale
        #
        # print(f'bias: {self.bias.item()}, scale: {self.scale.item()}')


        if model_B_poses is not None:
            gripper_pose_B = torch.cat([model_B_poses, depth_data], dim=1)
            B_grasp_quality = self.grasp_quality_(features2.detach(), gripper_pose_B)
        else:
            B_grasp_quality = None
        # grasp_quality=self.sig(grasp_quality)

        # grasp_quality=torch.rand_like(gripper_pose[:,0:1])
        # grasp_collision=torch.rand_like(gripper_pose[:,0:2])

        return gripper_pose, grasp_quality, grasp_collision, features2.detach(), B_grasp_quality

class ScalerEncoding_1d(nn.Module):
    def __init__(self,in_c,out_per_channel=10,shared=True,out=10):
        super().__init__()
        self.shared=shared
        self.mlp = nn.Sequential(
            nn.Linear(1, out_per_channel),
            nn.SiLU()
        ).to('cuda') if shared else nn.Sequential(
            nn.Linear(in_c, out),
            nn.SiLU()
        ).to('cuda')


    def forward(self, x):
        if self.shared:
            x=self.mlp(x[...,None])
            x=F.normalize(x,p=2,dim=-1,eps=1e-8)
            return x.flatten(start_dim=-2)
        else:
            x = self.mlp(x)
            x=F.normalize(x,p=2,dim=-1,eps=1e-8)
            return x

class ScalerEncoding_2d(nn.Module):
    def __init__(self,in_c,out_per_channel=10,shared=True,out=10):
        super().__init__()
        self.mlp = ScalerEncoding_1d(in_c,out_per_channel,shared,out)

    def forward(self, x):
        x=x.permute(0,2,3,1)
        x=self.mlp(x).permute(0,3,1,2)
        return x

class SH_D(nn.Module):
    def __init__(self):
        super().__init__()


        self.back_bone = SparseEncoderIN().to('cuda')

        # self.scaler_encoder=ScalerEncoding_1d(4,out_per_channel=10)

        self.att_block = ContextGate_1d(in_c1=512 , in_c2=9  ).to('cuda')

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
        feature_list = []
        depth_data_list = []
        xyz_list = []
        for pair in pairs:
            index = pair[0]
            xyz = pair[3].float()
            xyz_list.append(xyz)
            # feature_list.append(features[:, :, index])
            depth_data_list.append(depth_data[:, :, index])

        depth_data_list = torch.cat(depth_data_list, dim=0)[:, None, :].repeat(1, 2, 1)  #
        xyz_list = torch.stack(xyz_list)[:, None, :].repeat(1, 2, 1)  #

        # pose=torch.cat([pose[:,:,0:5],self.scaler_encoder(pose[:,:,5:])],dim=-1)

        scores = self.att_block(anchor[:,None], pose)

        return None, None, scores

class SH_D2(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=2, Batch_norm=None, Instance_norm=True,
                                  relu_negative_slope=0., activation=None, IN_affine=False,activate_skip=False).to('cuda')

        replace_instance_with_groupnorm(self.back_bone, max_groups=16)

        # add_spectral_norm_selective(self.back_bone)

        self.back_bone.apply(init_weights_he_normal)

        # self.att_block = ContextGate_1d_2(in_c1=64, in_c2=10).to('cuda')

        self.att_block = ContextGate_1d_3(64, 10, out_c=1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=False, bias=False, cyclic=False)

        # add_spectral_norm_selective(self.att_block)


    def forward(self, depth, pose, pairs, target_mask, cropped_spheres, latent_vector=None, detach_backbone=False):
        max_ = 1.3
        min_ = 1.15
        standarized_depth_ = (depth.clone() - min_) / (max_ - min_)
        standarized_depth_ = (standarized_depth_ - 0.5) / 0.5


        input = torch.cat([standarized_depth_, target_mask], dim=1)

        depth_data = standarized_depth_

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(input)  # *scale
        else:
            features = self.back_bone(input)  # *scale

        print('D max val= ',features.max().item(), 'mean:',features.mean().item(),' std:',features.std(dim=1).mean().item())
        # features=self.ln(features)
        features = features.flatten(2, 3)
        depth_data = depth_data.flatten(2, 3)
        feature_list = []
        depth_data_list = []
        xyz_list=[]
        for pair in pairs:
            index = pair[0]
            xyz=pair[3].float()
            xyz_list.append(xyz)
            feature_list.append(features[:, :, index])
            depth_data_list.append(depth_data[:, :, index])

        anchor = torch.cat(feature_list, dim=0)  # n,64
        depth_data_list=torch.cat(depth_data_list,dim=0)[:,None,:].repeat(1,2,1) #
        xyz_list=torch.stack(xyz_list)[:,None,:].repeat(1,2,1) #

        pose=torch.cat([pose,depth_data_list],dim=-1)


        scores = self.att_block(anchor[:, None], pose)

        return None, None, scores
