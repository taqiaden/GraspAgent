import torch.nn.functional as F
from GraspAgent_2.model.Decoders import ContextGate_1d,  res_ContextGate_2d, ContextGate_2d_2,  Quality_Net_2d
from GraspAgent_2.model.sparse_encoder import SparseEncoderIN
from GraspAgent_2.utils.NN_tools import replace_instance_with_groupnorm
from GraspAgent_2.utils.model_init import init_weights_he_normal
from models.resunet import res_unet
import torch
import torch.nn as nn

SH_model_key = 'SH_model'

class SHPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = ContextGate_2d_2(in_c1=64, in_c2= 1, out_c=3,
                                      relu_negative_slope=0.1, activation=None, use_sin=False,normalize=False,bias=False).to(
            'cuda')
        self.beta = ContextGate_2d_2(in_c1=64, in_c2= 1+3, out_c=2,
                                      relu_negative_slope=0.1, activation=None, use_sin=False,normalize=False,bias=False).to(
            'cuda')
        self.transition_=ContextGate_2d_2(in_c1=64, in_c2=1+5, out_c=1,
                                          relu_negative_slope=0.1, activation=None,use_sin=False,normalize=False).to(
            'cuda')

        self.fingers_abduction=ContextGate_2d_2(in_c1=64, in_c2=1+5+1, out_c=3,
                                          relu_negative_slope=0.1, activation=None,use_sin=False,normalize=False).to(
            'cuda')

        self.fingers=ContextGate_2d_2(in_c1=64, in_c2=1+5+1+3, out_c=6,
                                          relu_negative_slope=0.1, activation=None,use_sin=False,normalize=False).to(
            'cuda')

    def forward(self, features,depth,latent_vector):

        alpha = self.alpha(features,depth)
        alpha = F.normalize(alpha, dim=1)

        beta = self.beta(features,torch.cat([depth,alpha], dim=1))
        beta = F.normalize(beta, dim=1)


        transition=self.transition_(features,torch.cat([alpha,beta,depth], dim=1))

        fingers_abduction= self.fingers_abduction(features, torch.cat([alpha,beta,transition,depth], dim=1))

        fingers= self.fingers(features, torch.cat([alpha,beta,transition,fingers_abduction,depth], dim=1))

        pose = torch.cat([alpha,beta,fingers_abduction,fingers,transition], dim=1)

        return pose

def depth_standardization(depth,mask):
    mean_ = depth[mask].mean()
    depth_ = (depth.clone() - mean_)*10
    return depth_[None,None]

class SH_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.2,activation=nn.SiLU(),IN_affine=False,activate_skip=False).to('cuda')

        self.back_bone.apply(init_weights_he_normal)

        self.back_bone2 = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.2, activation=None, IN_affine=False,activate_skip =True).to('cuda')
        self.back_bone3_ = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.2, activation=nn.SiLU(), IN_affine=False,activate_skip =True).to('cuda')
        self.back_bone2.apply(init_weights_he_normal)
        self.back_bone3_.apply(init_weights_he_normal)

        replace_instance_with_groupnorm(self.back_bone2, max_groups=16)
        replace_instance_with_groupnorm(self.back_bone3_, max_groups=16)

        self.CH_PoseSampler = SHPoseSampler()

        self.query = nn.Sequential(
            nn.Conv2d(14, 5, kernel_size=1),
        ).to('cuda')

        self.grasp_quality_=Quality_Net_2d(in_c1=64, in_c2=16, out_c=1,
                                              relu_negative_slope=0.1,activation=None).to(
            'cuda')



        self.grasp_collision_ = res_ContextGate_2d(in_c1=64, in_c2=7, out_c=1,
                                              relu_negative_slope=0.1,activation=None).to(
            'cuda')
        self.grasp_collision2 = res_ContextGate_2d(in_c1=64, in_c2=7, out_c=1,
                                              relu_negative_slope=0.1,activation=None).to(
            'cuda')
        self.grasp_collision3= res_ContextGate_2d(in_c1=64, in_c2=7, out_c=1,
                                              relu_negative_slope=0.1,activation=None).to(
            'cuda')

        self.sig=nn.Sigmoid()

    def forward(self, depth,target_mask,latent_vector=None,backbone=None, detach_backbone=False):

        max_=1.3
        min_=1.15
        standarized_depth_ = (depth.clone() - min_)/(max_-min_)

        standarized_depth_ = (standarized_depth_ - 0.5) / 0.5
        print('Depth max=',standarized_depth_.max().item(), ', min=',standarized_depth_.min().item(),', std=',standarized_depth_.std().item(),', mean=',standarized_depth_.mean().item())

        input = torch.cat([standarized_depth_, target_mask], dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(input) #if backbone is None else backbone(input)
                # features2 = self.back_bone2(input)#*scale
                # features3 = self.back_bone3_(input)#*scale

        else:
            features = self.back_bone(input) #if backbone is None else backbone(input)
            features2 = self.back_bone2(input)#*scale
            features3 = self.back_bone3_(input)  # *scale

        print('G b1 max val= ',features.max().item(), 'mean:',features.mean().item(),' std:',features.std(dim=1).mean().item())
        print('G b2 max val= ',features2.max().item(), 'mean:',features2.mean().item(),' std:',features2.std(dim=1).mean().item())
        print('G b3 max val= ',features3.max().item(), 'mean:',features3.mean().item(),' std:',features3.std(dim=1).mean().item())


        depth_data=standarized_depth_

        gripper_pose=self.CH_PoseSampler(features,depth_data,latent_vector)

        detached_gripper_pose=gripper_pose.detach().clone()

        detached_gripper_pose=torch.cat([detached_gripper_pose,depth_data],dim=1)
        detached_gripper_pose_without_fingers=torch.cat([detached_gripper_pose[:,0:5],detached_gripper_pose[:,5+9:]],dim=1)


        grasp_collision=self.grasp_collision_(features2,detached_gripper_pose_without_fingers)
        grasp_collision=torch.cat([grasp_collision,self.grasp_collision3(features2,detached_gripper_pose_without_fingers)],dim=1)
        grasp_collision=torch.cat([grasp_collision,self.grasp_collision2(features2,detached_gripper_pose_without_fingers)],dim=1)


        grasp_quality=self.grasp_quality_(features3,detached_gripper_pose)


        return gripper_pose,grasp_quality,grasp_collision



class SH_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=2, Batch_norm=None, Instance_norm=True,
                                  relu_negative_slope=0.2, activation=nn.SiLU(), IN_affine=False).to('cuda')

        self.back_bone.apply(init_weights_he_normal)

        self.sparse_encoder=SparseEncoderIN().to('cuda')
        self.sparse_encoder.apply(init_weights_he_normal)


        self.att_block = ContextGate_1d(in_c1=512, in_c2=15, out_c=1).to('cuda')


    def forward(self, depth, pose,pairs, target_mask,cropped_spheres,latent_vector=None, detach_backbone=False):
        max_ = 1.3
        min_ = 1.15
        standarized_depth_ = (depth.clone() - min_) / (max_ - min_)
        standarized_depth_ = (standarized_depth_ - 0.5) / 0.5


        if detach_backbone:
            with torch.no_grad():
                point_global_features=self.sparse_encoder(cropped_spheres)
        else:
            point_global_features=self.sparse_encoder(cropped_spheres)

        print('D max val= ', point_global_features.max().item(), 'mean:', point_global_features.mean().item(), ' std:',
              point_global_features.std(dim=1).mean().item())


        input = torch.cat([standarized_depth_, target_mask], dim=1)

        depth_data=standarized_depth_


        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(input)#*scale
        else:
            features = self.back_bone(input)#*scale


        features=features.flatten(2,3)
        depth_data=depth_data.flatten(2,3)
        feature_list=[]
        depth_data_list=[]
        for pair in pairs:
            index=pair[0]
            feature_list.append(features[:,:,index])
            depth_data_list.append(depth_data[:,:,index])


        scores = self.att_block( point_global_features[:,None],pose)


        return None,None,scores
