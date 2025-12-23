import torch.nn.functional as F

from GraspAgent_2.model.Backbones import PointNetA
from GraspAgent_2.model.Decoders import  ContextGate_2d, ContextGate_1d, res_ContextGate_2d
from GraspAgent_2.utils.NN_tools import replace_instance_with_groupnorm
from GraspAgent_2.utils.model_init import  init_weights_he_normal
from GraspAgent_2.utils.positional_encoding import PositionalEncoding_2d, LearnableRBFEncoding1d, PositionalEncoding_1d, \
    LearnableRBFEncoding2D

from models.resunet import res_unet
import torch
import torch.nn as nn
YPG_point_base_model_key = 'YPG_model'


class ParallelGripperPoseSampler(nn.Module):
    def  __init__(self):
        super().__init__()

        self.approach = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 3)
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.beta_decoder = ContextGate_2d(in_c1=64, in_c2= 1+3, out_c=2,
                                      relu_negative_slope=0.1, activation=None, use_sin=False,normalize=False).to(
            'cuda')
        self.width_ = ContextGate_2d(in_c1=64, in_c2= 1+3+2, out_c=3,
                                      relu_negative_slope=0.1, activation=None, use_sin=False,normalize=False).to(
            'cuda')


        self.dist_ = ContextGate_2d(in_c1=64, in_c2= 1+3+2+1, out_c=3,
                                      relu_negative_slope=0.1, activation=None, use_sin=False,normalize=False).to(
            'cuda')


        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, features,depth_):
        vertical=torch.zeros_like(features[:,0:3])
        vertical[:,-1]+=1.

        approach = self.approach(features)
        approach = F.normalize(approach, dim=1)
        approach = approach * self.scale + vertical * (1-self.scale)
        approach = F.normalize(approach, dim=1)

        beta = self.beta_decoder(features, torch.cat([approach, depth_], dim=1))

        beta = F.normalize(beta, dim=1)

        width = self.width_(features, torch.cat([approach, beta,depth_], dim=1))
        width = F.normalize(width, dim=1).sum(dim=1,keepdim=True)

        dist = self.dist_(features, torch.cat([approach, beta, width,depth_], dim=1))
        dist = F.normalize(dist, dim=1).sum(dim=1,keepdim=True)


        pose = torch.cat([approach, beta, dist, width], dim=1)

        return pose

class YPG_G_point_base(nn.Module):
    def __init__(self):
        super().__init__()

        self.back_bone_ = PointNetA(use_instance_norm=True).to('cuda')
        self.back_bone_.apply(init_weights_he_normal)
        self.back_bone2 = PointNetA(use_instance_norm=True).to('cuda')
        self.back_bone2.apply(init_weights_he_normal)

        self.PoseSampler_ = ParallelGripperPoseSampler()

        self.grasp_quality=res_ContextGate_2d(in_c1=64, in_c2=8, out_c=1,
                                              relu_negative_slope=0.1,activation=None).to(
            'cuda')
        self.grasp_collision = res_ContextGate_2d(in_c1=64, in_c2=8, out_c=2,
                                              relu_negative_slope=0.1,activation=None).to(
            'cuda')

        self.background_detector = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            # LayerNorm2D(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            # LayerNorm2D(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=1),
        ).to('cuda')
        # add_spectral_norm_selective(self.grasp_collision_)

        self.sig=nn.Sigmoid()

        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction

        self.scale_gamma=nn.Sequential(nn.Linear(1,128)).to('cuda')
        self.scale_beta=nn.Sequential(nn.Linear(1,128)).to('cuda')

    def forward(self, pc_,backbone=None,pairs=None, detach_backbone=False):
        pc=pc_.clone()
        pc-=pc.mean(dim=1,keepdim=True)
        scale = torch.norm(pc, dim=-1).max(dim=1)[0][:,None]  # [B,1]
        pc/=scale

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone_(pc) #if backbone is None else backbone(input)
                features2 = self.back_bone2(pc)#*scale

        else:
            features = self.back_bone_(pc)#if backbone is None else backbone(input)
            features2 = self.back_bone2(pc)#*scale # b,128,N

        gamma=self.scale_gamma(scale)[:,:,None]
        beta=self.scale_beta(scale)[:,:,None]

        features=features*gamma+beta
        features2=features2*gamma+beta

        print('G max val= ',features.max().item(),' f2: ',features2.max().item())

        gripper_pose=self.PoseSampler_(features,pc)

        detached_gripper_pose=gripper_pose.detach().clone()

        detached_gripper_pose=torch.cat([detached_gripper_pose,depth_],dim=1)


        grasp_quality=self.grasp_quality(features2,detached_gripper_pose)

        grasp_quality=self.sig(grasp_quality)

        grasp_collision=self.grasp_collision(features2,detached_gripper_pose)
        grasp_collision=self.sig(grasp_collision)

        background_detection=self.background_detector(features2)
        background_detection=self.sig(background_detection)

        return gripper_pose,grasp_quality,background_detection,grasp_collision


def depth_standardization(depth):
    # mean_ = depth[mask].mean()
    mask=depth!=0
    mean_ = 1265
    # print(f'mean_: {mean_}')
    depth_ = (depth.clone() - mean_) / 30
    depth_[~mask] = 0.

    return depth_
class YPG_D_point_base(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=2, Batch_norm=None, Instance_norm=True,
                                  relu_negative_slope=0.2, activation=None, IN_affine=False).to('cuda')
        replace_instance_with_groupnorm(self.back_bone, max_groups=16)


        self.back_bone.apply(init_weights_he_normal)


        self.att_block_ = ContextGate_1d(in_c1=64, in_c2=8, out_c=1  ).to('cuda')
        self.att_block_.apply(init_weights_he_normal)



        self.cond_proj = nn.Sequential(
            nn.Linear(8, 128),
            nn.Softmax(dim=-1),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
        ).to('cuda')



        self.contx_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
        ).to('cuda')

        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, pc, pose,pairs,mask, backbone=None, detach_backbone=False):


        depth_ = depth_standardization(depth)

        input = torch.cat([depth_, mask], dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(input)#if backbone is None else backbone(input)
        else:
            features = self.back_bone(input)#if backbone is None else backbone(input)

        # features=features.repeat(2,1,1,1)#.permute(0,2,3,1)
        # features=torch.cat([features,scaled_depth_,depth_],dim=1)
        print('D max val= ',features.max().item(), 'mean:',features.mean().item(),' std:',features.std(dim=1).mean().item())

        features = features.view(1, 64, -1)
        depth_ = depth_.view(1, 1, -1)
        feature_list = []
        depth_list = []
        for pair in pairs:
            index = pair[0]
            feature_list.append(features[:, :, index])
            depth_list.append(depth_[:, :, index])

        # feature_list = torch.cat(feature_list, dim=0)[:, None, :].repeat(1, 2, 1)  # n,2,64
        anchor = torch.cat(feature_list, dim=0)  # n,64

        depth_list = torch.cat(depth_list, dim=0)[:, None, :].repeat(1, 2, 1)  # n,2,64

        # output = self.att_block_( feature_list,torch.cat([pose,depth_list],dim=-1))
        positive_negative = self.cond_proj( torch.cat([pose,depth_list],dim=-1))
        positive_negative=F.normalize(positive_negative,p=2,dim=-1,eps=1e-8)
        anchor=self.contx_proj(anchor)
        anchor=F.normalize(anchor,p=2,dim=-1,eps=1e-8)

        return anchor,positive_negative




