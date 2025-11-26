import numpy as np
import torch.nn.functional as F
from torch import utils

from GraspAgent_2.model.Decoders import film_fusion_2d, film_fusion_1d, ParameterizedSine, att_1d, att_2d, \
    Linear_modulation_2d, res_att_2d, Grasp_att_2d
from GraspAgent_2.model.utils import  add_spectral_norm_selective
from GraspAgent_2.utils.NN_tools import replace_instance_with_groupnorm
from GraspAgent_2.utils.depth_processing import sobel_gradients
from GraspAgent_2.utils.model_init import orthogonal_init_all, init_orthogonal, init_weights_normal, \
    init_weights_xavier_normal, init_weights_he_normal, gan_init_with_norms
from GraspAgent_2.utils.positional_encoding import PositionalEncoding_2d, PositionalEncoding_1d, EncodedScaler, \
    LearnableRBFEncoding2D, LearnableRBFEncoding1d, depth_sin_cos_encoding
from GraspAgent_2.utils.quat_operations import sign_invariant_quat_encoding_1d, sign_invariant_quat_encoding_2d
from GraspAgent_2.utils.weigts_normalization import scale_all_weights, scale_module_weights
from lib.image_utils import view_image
from models.decoders import  LayerNorm2D
from models.resunet import res_unet
import torch
import torch.nn as nn
CH_model_key = 'CH_model'

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

class ParallelGripperPoseSampler(nn.Module):
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
        #                               relu_negative_slope=0.2, activation=None, use_sin=True,bias=False).to(
        #     'cuda')
        # self.transition=film_fusion_2d(in_c1=64, in_c2=10+1, out_c=1,
        #                                   relu_negative_slope=0.2, activation=None,use_sin=False,bias=False).to(
        #     'cuda')
        # self.fingers_=film_fusion_2d(in_c1=64, in_c2=10+10+1, out_c=3,
        #                                   relu_negative_slope=0.2, activation=None,use_sin=False,bias=False).to(
        #     'cuda')

        self.quat = att_2d(in_c1=64, in_c2= 10, out_c=4,
                                      relu_negative_slope=0.2, activation=None, use_sin=True).to(
            'cuda')
        self.transition=att_2d(in_c1=64, in_c2=10+10, out_c=1,
                                          relu_negative_slope=0.2, activation=nn.SiLU(),use_sin=False).to(
            'cuda')
        self.fingers_=att_2d(in_c1=64, in_c2=10+10+10, out_c=3,
                                          relu_negative_slope=0.2, activation=None,use_sin=False).to(
            'cuda')

        self.depth_encoding=ScalerEncoding_2d(in_c=1,out_c=10)
        self.transition_encoding=ScalerEncoding_2d(in_c=1,out_c=10)

        # self.fingers_scale = film_fusion_2d(in_c1=64, in_c2=32+32+11, out_c=32,
        #                                   relu_negative_slope=0.2, activation=None,normalize=True,final_linear=False).to(
        #     'cuda')



        # self.proj_quat_=nn.Conv2d(32, 4, kernel_size=1).to('cuda')
        # self.proj_fingers=nn.Conv2d(32, 3, kernel_size=1).to('cuda')
        # self.proj_transition=nn.Conv2d(32, 1, kernel_size=1).to('cuda')

        # self.normalize_quat_embedding= nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=1),
        #     nn.Tanh()
        # ).to('cuda')
        #
        # self.normalize_fingers_embedding= nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=1),
        #     nn.Tanh()
        # ).to('cuda')

        # self.finger_encoder=EncodedScaler(min_val=0.1,max_val=1.7).to('cuda')
        # self.transition_encoder=EncodedScaler(min_val=-1,max_val=1).to('cuda')

        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d(num_freqs=4)  # for 2D/3D viewing direction

    #     self.initialize_film_gen()
    #
    # def initialize_film_gen(self):
    #     self.proj_quat_.apply(lambda m: init_orthogonal(m, scale=1))
    #     self.proj_fingers.apply(lambda m: init_orthogonal(m, scale=1/4))
    #     self.proj_transition.apply(lambda m: init_orthogonal(m, scale=1/6))

    def forward(self, features,depth):
        encoded_depth=self.depth_encoding(depth)
        quat = self.quat(features,encoded_depth)
        # quat = torch.where(quat[:, 0:1] < 0, -quat, quat)
        quat = F.normalize(quat, dim=1)
        encoded_quat=sign_invariant_quat_encoding_2d(quat)
        # quat_embedding=self.normalize_quat_embedding(quat_embedding)

        # fingers_embedding=self.normalize_fingers_embedding(fingers_embedding)
        transition=self.transition(features,torch.cat([encoded_quat,encoded_depth], dim=1))
        encoded_transition=self.transition_encoding(transition)
        # transition=F.tanh(transition)
        # encoded_transition=self.pos_encoder(transition)
        fingers= self.fingers_(features, torch.cat([encoded_quat,encoded_transition,encoded_depth], dim=1))


        # fingers=F.sigmoid(fingers)

        # quat=self.proj_quat_(quat_embedding)
        # fingers=self.proj_fingers(fingers_embedding)
        # transition=self.proj_transition(transition_embedding)

        pose = torch.cat([quat,fingers,transition], dim=1)
        # exit()
        return pose
def depth_standardization(depth,mask):
    mean_ = depth[mask].mean()
    depth_ = (depth.clone() - mean_)*10
    return depth_[None,None]

class CH_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone_ = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.2,activation=None,IN_affine=False).to('cuda')
        # gain = torch.nn.init.calculate_gain('leaky_relu', 0.2)
        # self.back_bone_.apply(init_weights_he_normal)
        replace_instance_with_groupnorm(self.back_bone_, max_groups=16)
        gan_init_with_norms(self.back_bone_)


        self.back_bone_2 = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.2, activation=nn.SiLU(), IN_affine=False).to('cuda')
        replace_instance_with_groupnorm(self.back_bone_2, max_groups=16)
        # replace_instance_with_groupnorm(self.back_bone2, max_groups=32)
        # orthogonal_init_all(self.back_bone_2,gain=gain)

        self.CH_PoseSampler_ = ParallelGripperPoseSampler()

        gan_init_with_norms(self.CH_PoseSampler_)

        # self.grasp_quality_=film_fusion_2d(in_c1=64, in_c2=51, out_c=1,in_c3=2,
        #                                       relu_negative_slope=0.2,activation=None,normalize=False,use_sin=False,bias=False,gate=True).to(
        #     'cuda')
        # self.grasp_collision_ = film_fusion_2d(in_c1=64, in_c2=51, out_c=2,
        #                                       relu_negative_slope=0.2,activation=None,normalize=False,use_sin=False,bias=False,gate=True).to(
        #     'cuda')

        self.grasp_quality_=res_att_2d(in_c1=64, in_c2=60, out_c=1,in_c3=3,
                                              relu_negative_slope=0.2,activation=None,normalize=True,use_sin=False).to(
            'cuda')
        self.grasp_collision_ = res_att_2d(in_c1=64, in_c2=30, out_c=3,
                                              relu_negative_slope=0.2,activation=None,normalize=True,use_sin=False).to(
            'cuda')

        self.depth_encoding=ScalerEncoding_2d(in_c=1,out_c=10)
        self.transition_encoding=ScalerEncoding_2d(in_c=1,out_c=10)
        self.fingers_encoding=ScalerEncoding_2d(in_c=3,out_c=30)

        # self.grasp_collision_ = Grasp_att_2d(in_c1=64, rotation_size=10, transition_size=2,fingers_size=0, out_c=2,
        #                                       relu_negative_slope=0.2,activation=None,normalize=True,use_sin=False).to(
        #     'cuda')

        # self.grasp_collision_=nn.Sequential(
        #     nn.Conv2d(64 + 51, 64, kernel_size=1),
        #     siren(30),
        #     nn.Conv2d(64, 32, kernel_size=1),
        #     siren(10),
        #     nn.Conv2d(32, 2, kernel_size=1)
        # ).to('cuda')
        # self.res = nn.Sequential(
        #     nn.Conv2d(64+51, 64, kernel_size=1),
        # ).to('cuda')

        # self.condition_encoder = nn.Sequential(
        #     nn.Conv2d(10+40+1, 128,kernel_size=1),
        #     nn.SiLU()
        # ).to('cuda')

        # add_spectral_norm_selective(self.back_bone_2)
        # add_spectral_norm_selective(self.grasp_quality_)
        # add_spectral_norm_selective(self.grasp_collision_)

        self.sig=nn.Sigmoid()

        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction


    def forward(self, depth,target_mask,latent_mask, detach_backbone=False):
        # depth_= depth_standardization(depth[0,0],target_mask[0,0])
        # range=depth.max()-depth.min()
        # center=depth.min()+range/2
        standarized_depth_ = (depth.clone() - depth.mean())*10
        # floor_elevation=standarized_depth_[0,0][~target_mask[0,0]].mean()
        # scale_=scale_*5*torch.ones_like(standarized_depth_)


        # local_diff3 = depth_ - F.avg_pool2d(depth_, kernel_size=3, stride=1, padding=1)
        # local_diff5 = depth_ - F.avg_pool2d(depth_, kernel_size=5, stride=1, padding=2)
        # local_diff7 = depth_ - F.avg_pool2d(depth_, kernel_size=7, stride=1, padding=3)
        #
        # Gx, Gy = sobel_gradients(depth_)
        #
        # encoded_depth = torch.cat([depth_, Gx, Gy, local_diff3, local_diff5, local_diff7], dim=1)

        input = torch.cat([standarized_depth_, target_mask], dim=1)
        # standarized_depth_2=standarized_depth_.clone()
        # standarized_depth_2[latent_mask==0]*=0
        # latent_input = torch.cat([standarized_depth_2, target_mask], dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone_(input)#*scale
                features2 = self.back_bone_2(input)#*scale
        else:
            features = self.back_bone_(input)#*scale
            features2 = self.back_bone_2(input)#*scale

        # features2=torch.cat([features2,scaled_depth_,depth_],dim=1)
        # features=torch.cat([features,scaled_depth_,depth_],dim=1)
        print('G b1 max val= ',features.max().item(), 'mean:',features.mean().item(),' std:',features.std(dim=1).mean().item())
        print('G b2 max val= ',features2.max().item(), 'mean:',features2.mean().item(),' std:',features2.std(dim=1).mean().item())


        depth_data=standarized_depth_

        gripper_pose=self.CH_PoseSampler_(features,depth_data)

        detached_gripper_pose=gripper_pose.detach().clone()
        quat = detached_gripper_pose[:, :4]
        fingers = detached_gripper_pose[:,  4:4 + 3]
        fingers=torch.clip(fingers,0,1)
        transition = detached_gripper_pose[:, 4 + 3:4+3+1]

        encoded_quat = sign_invariant_quat_encoding_2d(quat)  # 10
        fingers=self.fingers_encoding(fingers)
        transition=self.depth_encoding(transition)
        depth_data=self.depth_encoding(depth_data)
        # encoded_fingers = self.pos_encoder(fingers)  # 30
        # encoded_transition = self.pos_encoder((transition + 1) / 2)  # 10
        # encoded_depth = self.pos_encoder(depth_)  # 10


        detached_gripper_pose=torch.cat([encoded_quat,fingers,transition,depth_data],dim=1)
        detached_gripper_pose_without_fingers=torch.cat([encoded_quat,transition,depth_data],dim=1)

        # detached_gripper_pose=self.condition_encoder(detached_gripper_pose)

        grasp_collision=self.grasp_collision_(features2,detached_gripper_pose_without_fingers)
        # grasp_collision=self.grasp_collision_(features2,rotation=encoded_quat,transition=torch.cat([transition,depth_data],dim=1),fingers=None)

        # grasp_collision=self.sig(grasp_collision)

        grasp_quality=self.grasp_quality_(features2,detached_gripper_pose,additional_features=F.sigmoid(grasp_collision.detach()))
        # grasp_quality=self.sig(grasp_quality)

        # grasp_quality=torch.rand_like(gripper_pose[:,0:1])
        # grasp_collision=torch.rand_like(gripper_pose[:,0:2])

        return gripper_pose,grasp_quality,grasp_collision
class ScalerEncoding_1d(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_c, out_c),
        ).to('cuda')


    def forward(self, x):
        x=self.mlp(x)
        x = F.normalize(x, p=2, dim=2, eps=1e-8)
        return x

class ScalerEncoding_2d(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1),
        ).to('cuda')


    def forward(self, x):
        x=self.mlp(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        return x
class CH_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=3, Batch_norm=None, Instance_norm=False,
                                  relu_negative_slope=0.2, activation=None, IN_affine=False).to('cuda')
        # add_spectral_norm_selective(self.back_bone)

        # self.back_bone.apply(init_weights_he_normal)

        gan_init_with_norms(self.back_bone)


        # self.att_block = normalize_free_att_2d(in_c1=64, in_c2=7+1, out_c=1,
        #                                relu_negative_slope=0.2, activation=None,softmax_att=True).to('cuda')
        # self.condition_encoder = nn.Sequential(
        #     nn.Linear(10+40+1, 128),
        #     nn.LayerNorm(128),
        #     nn.SiLU()
        # ).to('cuda')
        # self.att_block = film_fusion_1d(in_c1=64, in_c2=51, out_c=1,
        #                                relu_negative_slope=0.2, activation=nn.SiLU(),normalize=False,with_gate=True,bias=False).to('cuda')
        self.att_block = att_1d(in_c1=64, in_c2=60, out_c=1,
                                       relu_negative_slope=0.2, activation=nn.SiLU(),normalize=False,use_sin=False).to('cuda')
        gan_init_with_norms(self.att_block)

        self.depth_encoding =ScalerEncoding_1d(in_c=1,out_c=10)
        self.transition_encoding =ScalerEncoding_1d(in_c=1,out_c=10)
        self.fingers_encoding =ScalerEncoding_1d(in_c=3,out_c=30)

        # self.f1 = film_fusion_1d(in_c1=64, in_c2=10, out_c=1,
        #                                relu_negative_slope=0.2, activation=nn.SiLU(),normalize=False,with_gate=True,bias=False,decode=False).to('cuda')
        # self.f2 = film_fusion_1d(in_c1=64, in_c2=10+1, out_c=1,
        #                                relu_negative_slope=0.2, activation=nn.SiLU(),normalize=False,with_gate=True,bias=False,decode=False).to('cuda')
        # self.f3 = film_fusion_1d(in_c1=64, in_c2=30, out_c=1,
        #                                relu_negative_slope=0.2, activation=nn.SiLU(),normalize=False,with_gate=True,bias=False,decode=True).to('cuda')

        # self.att_block = nn.Sequential(
        #     nn.Linear(64+10+40+1+8, 128),
        #     nn.SiLU(),
        #     nn.Linear(128, 64),
        #     nn.SiLU(),
        #     nn.Linear(64, 1),
        # ).to('cuda')

        # self.att_block2 = film_fusion_1d(in_c1=64, in_c2=40+1, out_c=1,
        #                                relu_negative_slope=0.2, activation=nn.SiLU()).to('cuda')
        # add_spectral_norm_selective(self.att_block)
        # add_spectral_norm_selective(self.f1)
        # add_spectral_norm_selective(self.f2)
        # add_spectral_norm_selective(self.f3)

        # add_spectral_norm_selective(self.att_block2)


        # self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        # self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction

        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

        # self.ln=LayerNorm2D(64).to('cuda')

    def forward(self, depth, pose,pairs, target_mask, detach_backbone=False):
        attention_mask = torch.zeros_like(target_mask, dtype=torch.bool)
        for pair in pairs:
            index = pair[0]
            h = index // 600
            w = index % 600
            attention_mask[0, 0, h, w] = True

        # depth_= depth_standardization(depth[0,0],target_mask[0,0])
        # range = depth.max() - depth.min()
        # center = depth.min() + range / 2
        standarized_depth_ = (depth.clone() - depth.mean()) * 10
        # local_diff3 = depth_ - F.avg_pool2d(depth_, kernel_size=3, stride=1, padding=1)
        # local_diff5 = depth_ - F.avg_pool2d(depth_, kernel_size=5, stride=1, padding=2)
        # local_diff7 = depth_ - F.avg_pool2d(depth_, kernel_size=7, stride=1, padding=3)
        #
        # Gx, Gy = sobel_gradients(depth_)
        #
        # encoded_depth = torch.cat([depth_, Gx, Gy, local_diff3, local_diff5, local_diff7], dim=1)

        input = torch.cat([standarized_depth_, target_mask,attention_mask], dim=1)

        depth_data=standarized_depth_


        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(input)#*scale
        else:
            features = self.back_bone(input)#*scale

        # features = torch.cat([features, scaled_depth_, depth_], dim=1)

        print('D max val= ',features.max().item(), 'mean:',features.mean().item(),' std:',features.std(dim=1).mean().item())
        # features=self.ln(features)
        features=features.view(1,64,-1)
        depth_data=depth_data.view(1,1,-1)
        feature_list=[]
        depth_data_list=[]
        for pair in pairs:
            index=pair[0]
            feature_list.append(features[:,:,index])
            depth_data_list.append(depth_data[:,:,index])

        feature_list=torch.cat(feature_list,dim=0)[:,None,:].repeat(1,2,1) # n,2,64
        depth_data_list=torch.cat(depth_data_list,dim=0)[:,None,:].repeat(1,2,1) # n,2,64

        quat = pose[:,:, :4]
        fingers = pose[:,:, 4:4+3]
        transition = pose[:,:, 4+3:4+3+1]


        encoded_quat = sign_invariant_quat_encoding_1d(quat)  # 10
        # encoded_transition = self.pos_encoder((transition+1)/2)  # 10
        # # encoded_depth=self.pos_encoder(encoded_depth_list) # 10
        # encoded_fingers=self.pos_encoder(fingers) # 30

        transition=self.transition_encoding(transition)
        depth_data_list=self.depth_encoding(depth_data_list)
        fingers=self.fingers_encoding(fingers)

        condition=(torch.cat([encoded_quat,fingers,transition,depth_data_list], dim=-1))


        # output = self.att_block(feature_list, torch.cat([encoded_quat,fingers,transition,depth_data_list], dim=-1))
        output = self.att_block( feature_list,condition)

        # output1 = self.f1( feature_list,encoded_quat)+feature_list
        # output2 = self.f2( output1,torch.cat([encoded_transition,depth_data_list], dim=-1))+output1
        # output = self.f3( output2,encoded_fingers)

        # output = self.att_block2(output, torch.cat([fingers,transition,depth_data_list], dim=-1))

        return output

