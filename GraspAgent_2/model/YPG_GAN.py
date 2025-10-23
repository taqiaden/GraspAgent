import torch
from torch import nn
import torch.nn.functional as F
from GraspAgent_2.model.Backbones import PointNetA
from GraspAgent_2.model.Decoders import normalize_free_att_sins, normalized_att_1d, \
    normalize_free_att_2d, custom_att_1d, ParameterizedSine, normalize_free_att_1d, \
    custom_att_2d, normalized_att_2d, normalize_free_res_att_2d, normalized_res_att_2d, film_fusion_2d, film_fusion_1d
from GraspAgent_2.model.conv_140 import ConvFeatureExtractor
from GraspAgent_2.model.utils import replace_activations, add_spectral_norm_selective
from GraspAgent_2.model.Grasp_Discriminator import D, SubD
from models.decoders import sine, LayerNorm2D, att_res_conv_normalized
from models.resunet import res_unet
from registration import  standardize_depth
from lib.cuda_utils import cuda_memory_report

YPG_model_key = 'YPG_model'
YPG_model_key2 = 'YPG_model2'
YPG_model_key3 = 'YPG_model3'

class NormalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(128, 32, bias=False),
            nn.LayerNorm([32]),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 3),
        ).to('cuda')

    def forward(self, features):
        x = self.decoder(features)
        x = F.normalize(x, dim=-1)
        return x

class ParallelGripperPoseSampler1d(nn.Module):
    def  __init__(self):
        super().__init__()

        self.approach = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            ParameterizedSine(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            ParameterizedSine(),
            nn.Linear(16, 3)
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)

        # self.beta_decoder = normalize_free_att_sins(in_c1=128, in_c2=3, out_c=2).to(
        #     'cuda')
        # self.beta_decoder = normalize_free_att_1d(in_c1=128, in_c2=3+3, out_c=2,
        #                                   relu_negative_slope=0., activation=nn.SiLU(),use_sin=True).to(
        #     'cuda')
        self.beta_decoder = film_fusion_1d(in_c1=128, in_c2=3+3, out_c=2,
                                          relu_negative_slope=0., activation=nn.SiLU(),use_sin=True,normalize=True).to(
            'cuda')
        self.width_ = film_fusion_1d(in_c1=128, in_c2=3+3+2, out_c=1,
                                          relu_negative_slope=0., activation=nn.SiLU(),normalize=True).to(
            'cuda')

        self.dist_ = film_fusion_1d(in_c1=128, in_c2=3+3+2+1, out_c=1,
                                         relu_negative_slope=0., activation=nn.SiLU(),normalize=True).to(
            'cuda')


        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, features,pc):
        encoded_pc=(pc)#.detach()


        vertical=torch.zeros_like(features[:,:,0:3])
        vertical[:,:,-1]+=1.

        # approach=vertical

        approach = self.approach(features)
        approach = F.normalize(approach, dim=-1)
        approach = approach * self.scale + vertical * (1-self.scale)
        approach = F.normalize(approach, dim=-1)

        encoded_approach=(approach)#.detach()


        beta = self.beta_decoder(features, torch.cat([encoded_approach, encoded_pc], dim=-1))

        beta = F.normalize(beta, dim=-1)

        encoded_beta=(beta)#.detach()


        width = self.width_(features, torch.cat([encoded_approach, encoded_beta,encoded_pc], dim=-1))

        encoded_width=(width)#.detach()


        dist = self.dist_(features, torch.cat([encoded_approach, encoded_beta, encoded_width,encoded_pc], dim=-1))

        # dist=F.sigmoid(dist)
        # width=F.sigmoid(width)

        pose = torch.cat([approach, beta, dist, width], dim=-1)

        return pose


class ParallelGripperPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.approach = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            ParameterizedSine(),
            nn.Conv2d(32, 16, kernel_size=1),
            ParameterizedSine(),
            nn.Conv2d(16, 3, kernel_size=1)
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)

        # self.beta_decoder = normalize_free_att_sins(in_c1=128, in_c2=3, out_c=2).to(
        #     'cuda')
        # self.beta_decoder = normalize_free_att_2d(in_c1=64, in_c2=3+1, out_c=2,
        #                                   relu_negative_slope=0., activation=nn.SiLU(),softmax_att=False,use_sin=True).to(
        #     'cuda')
        self.beta_decoder = film_fusion_2d(in_c1=64, in_c2=3+1, out_c=2,
                                          relu_negative_slope=0., activation=None,use_sin=True).to(
            'cuda')
        self.width_ = film_fusion_2d(in_c1=64, in_c2=3+2+1, out_c=1,
                                          relu_negative_slope=0.1, activation=nn.SiLU()).to(
            'cuda')

        self.dist_ = film_fusion_2d(in_c1=64, in_c2=3+2+1+1, out_c=1,
                                         relu_negative_slope=0.1, activation=nn.SiLU()).to(
            'cuda')


        self.pos_encoder = PositionalEncoding_2d(num_freqs=10)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d(num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, features,depth_):
        encoded_depth=(depth_)#.detach()


        vertical=torch.zeros_like(features[:,0:3])
        vertical[:,-1]+=1.

        # approach=vertical

        approach = self.approach(features)
        approach = F.normalize(approach, dim=1)
        approach = approach * self.scale + vertical * (1-self.scale)
        approach = F.normalize(approach, dim=1)

        encoded_approach=(approach)#.detach()

        beta = self.beta_decoder(features, torch.cat([encoded_approach, encoded_depth], dim=1))

        beta = F.normalize(beta, dim=1)

        encoded_beta=(beta)#.detach()



        width = self.width_(features, torch.cat([encoded_approach, encoded_beta,encoded_depth], dim=1))

        encoded_width=(width)#.detach()


        dist = self.dist_(features, torch.cat([encoded_approach, encoded_beta, encoded_width,encoded_depth], dim=1))

        # dist=F.sigmoid(dist)
        # width=F.sigmoid(width)

        pose = torch.cat([approach, beta, dist, width], dim=1)

        return pose

class YPG_G_Point_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = PointNetA(use_instance_norm=True)

        replace_activations(self.back_bone, nn.ReLU, nn.LeakyReLU(0.01))
        # add_spectral_norm_selective(self.back_bone)

        self.PoseSampler = ParallelGripperPoseSampler1d()

        self.grasp_quality=film_fusion_1d(in_c1=128, in_c2=95+2, out_c=1,
                                              relu_negative_slope=0.1,activation=None,normalize=True).to('cuda')
        self.grasp_collision = film_fusion_1d(in_c1=128, in_c2=95+2, out_c=2,
                                            relu_negative_slope=0.1, activation=None,normalize=True).to('cuda')

        self.background_detector_ = nn.Sequential(
            nn.Linear(128, 32, bias=False),
            nn.LayerNorm([32]),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
        ).to('cuda')

        # self.ln=nn.LayerNorm(128).to('cuda')
        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

        self.sig=nn.Sigmoid()

    def forward(self, pc, detach_backbone=False):
        pc_=pc.clone()
        # pc_[:,:,0]-=0.43
        pc_=pc_-pc_.mean(dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(pc_)
        else:
            features = self.back_bone(pc_)

        print('G max val= ',features.max().item())

        features=features.transpose(1,2)

        # features=self.ln(features)

        gripper_pose=self.PoseSampler(features,pc_)

        detached_gripper_pose=gripper_pose.detach().clone()

        dir=detached_gripper_pose[:,:,:5]
        pos=detached_gripper_pose[:,:,5:]
        # print(pos[0,0:2])
        pos=self.pos_encoder(pos) # 20
        # print(pos[0,0:2])
        # exit()
        dir=self.dir_encoder(dir) # 45
        pc_e=self.pos_encoder(pc_) # 30


        # pos2=self.pos_encoder(depth_).repeat(2,1,1,1) # 10
        detached_gripper_pose=torch.cat([dir,pos,pc_e,detached_gripper_pose[:,:,5:]],dim=-1)

        grasp_quality=self.grasp_quality(features,detached_gripper_pose)

        grasp_collision=self.grasp_collision(features,detached_gripper_pose)
        grasp_collision=self.sig(grasp_collision)

        background_detection=self.background_detector_(features)
        background_detection=self.sig(background_detection)

        return gripper_pose,grasp_quality,background_detection,grasp_collision
def get_auto_groupnorm(num_channels, max_groups=8,affine=True):
    # Find the largest number of groups <= max_groups that divides num_channels
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=num_channels, affine=affine).to('cuda')
    # fallback to LayerNorm behavior
    return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=affine).to('cuda')

def replace_instance_with_groupnorm(module, max_groups=8,affine=True):
    for name, child in module.named_children():
        if isinstance(child, nn.InstanceNorm2d):
            gn = get_auto_groupnorm(child.num_features, max_groups=max_groups,affine=affine)
            setattr(module, name, gn)
        else:
            replace_instance_with_groupnorm(child, max_groups=max_groups)




class YPG_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.1,activation=None,IN_affine=False).to('cuda')
        # add_spectral_norm_selective(self.back_bone)
        replace_instance_with_groupnorm(self.back_bone, max_groups=16)


        self.PoseSampler = ParallelGripperPoseSampler()

        self.grasp_quality_=film_fusion_2d(in_c1=64, in_c2=75, out_c=1,
                                              relu_negative_slope=0.1,activation=None,normalize=False).to(
            'cuda')
        self.grasp_collision_ = film_fusion_2d(in_c1=64, in_c2=75, out_c=2,
                                              relu_negative_slope=0.1,activation=None,normalize=False).to(
            'cuda')
        add_spectral_norm_selective(self.grasp_quality_)
        add_spectral_norm_selective(self.grasp_collision_)


        self.background_detector_ = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            LayerNorm2D(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 1, kernel_size=1),
        ).to('cuda')



        self.sig=nn.Sigmoid()

        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction


    def forward(self, depth, detach_backbone=False):
        depth_ = standardize_depth(depth)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth_)
        else:
            features = self.back_bone(depth_)


        print('G max val= ',features.max().item())

        gripper_pose=self.PoseSampler(features,depth_)

        detached_gripper_pose=gripper_pose.detach().clone()
        dir=detached_gripper_pose[:,:5,...]
        pos=detached_gripper_pose[:,5:,...]
        dir=self.dir_encoder(dir) # 45
        pos=self.pos_encoder(pos) # 20
        encoded_depth=self.pos_encoder(depth_) # 20

        detached_gripper_pose_encoded=torch.cat([dir,pos,encoded_depth],dim=1)
        print('test---')
        # cuda_memory_report()

        grasp_quality=self.grasp_quality_(features,detached_gripper_pose_encoded)
        # grasp_quality=self.sig(grasp_quality)
        # cuda_memory_report()

        grasp_collision=self.grasp_collision_(features,detached_gripper_pose_encoded)
        grasp_collision=self.sig(grasp_collision)

        background_detection=self.background_detector_(features)
        background_detection=self.sig(background_detection)

        return gripper_pose,grasp_quality,background_detection,grasp_collision

class PositionalEncoding_2d(nn.Module):
    def __init__(self, num_freqs=10):
        """
        num_freqs: number of frequency bands for Fourier features
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.arange(num_freqs) * torch.pi

    def forward(self, x):
        """
        x: [b, in_dim, h, w]
        returns: [b, in_dim * (2*num_freqs+1), h, w]
        """
        b, in_dim, h, w = x.shape
        # [b, in_dim, h, w] -> [b, h, w, in_dim]
        x_perm = x.permute(0, 2, 3, 1)

        out = [x_perm]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x_perm))
            out.append(torch.cos(freq * x_perm))

        encoded = torch.cat(out, dim=-1)  # concat on channel-like axis
        # [b, h, w, in_dim*(2*num_freqs+1)] -> [b, in_dim*(2*num_freqs+1), h, w]
        encoded = encoded.permute(0, 3, 1, 2).contiguous()
        return encoded

class PositionalEncoding_1d(nn.Module):
    def __init__(self, num_freqs=10):
        """
        num_freqs: number of frequency bands for Fourier features
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.arange(num_freqs) * torch.pi

    def forward(self, x):
        """
        x: [..., dim]  (any number of leading dimensions, last dim = coordinate/features)
        returns: [..., dim * (2*num_freqs + 1)]
        """
        orig_shape = x.shape
        dim = x.shape[-1]

        # Flatten leading dimensions
        x_flat = x.reshape(-1, dim)  # [*, dim]

        out = [x_flat]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x_flat))
            out.append(torch.cos(freq * x_flat))

        encoded = torch.cat(out, dim=-1)
        # Restore leading dimensions
        final_shape = orig_shape[:-1] + (dim * (2 * self.num_freqs + 1),)
        return encoded.view(final_shape)

import torch
import torch.nn as nn

class LearnableRBFEncoding2D(nn.Module):
    def __init__(self, num_centers=16, init_sigma=0.1):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(0, 1, num_centers)).to('cuda')
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma))).to('cuda')

    def forward(self, x):
        B, C, W, H = x.shape

        x_exp = x.unsqueeze(2)
        diff = x_exp - self.centers.view(1, 1, -1, 1, 1)
        sigma = torch.exp(self.log_sigma)
        rbf = torch.exp(-0.5 * (diff / sigma) ** 2)
        rbf = rbf.view(B, C * len(self.centers), W, H)
        return rbf


class LearnableRBFEncoding1d(nn.Module):
    def __init__(self, num_centers=16, init_sigma=0.1, device='cuda'):
        super().__init__()
        self.num_centers = num_centers
        self.register_parameter(
            "centers",
            nn.Parameter(torch.linspace(0, 1, num_centers, device=device))
        )
        self.register_parameter(
            "log_sigma",
            nn.Parameter(torch.log(torch.tensor(init_sigma, device=device)))
        )

    def forward(self, x):
        # x shape: [*, D] (any number of leading dims)
        *prefix, D = x.shape

        # Expand for broadcasting
        x_exp = x.unsqueeze(-1)  # [*, D, 1]
        centers = self.centers.view(*([1] * (len(prefix) + 1)), -1)  # [1,...,1, num_centers]

        # Compute Gaussian RBF encoding
        sigma = torch.exp(self.log_sigma)
        # print(sigma)
        # print(self.centers.squeeze())
        # exit()

        diff = x_exp - centers  # [*, D, num_centers]
        rbf = torch.exp(-0.5 * (diff / sigma) ** 2)

        # Flatten last two dims: each feature D expands to D*num_centers
        rbf = rbf.reshape(*prefix, D * self.num_centers)
        return rbf

class YPG_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=None, Instance_norm=False,
                                  relu_negative_slope=0.01, activation=None, IN_affine=False).to('cuda')
        add_spectral_norm_selective(self.back_bone)

        # self.att_block = normalize_free_att_2d(in_c1=64, in_c2=7+1, out_c=1,
        #                                relu_negative_slope=0.2, activation=None,softmax_att=True).to('cuda')
        self.att_block = film_fusion_2d(in_c1=64, in_c2=75, out_c=1,
                                       relu_negative_slope=0.2, activation=None).to('cuda')
        add_spectral_norm_selective(self.att_block)

        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, depth, pose,  detach_backbone=False):
        depth_ = standardize_depth(depth)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth_)
        else:
            features = self.back_bone(depth_)

        features=features.repeat(2,1,1,1)#.permute(0,2,3,1)

        print('D max val= ',features.max().item())
        dir=pose[:,:5,...]
        pos=pose[:,5:,...]
        dir=self.dir_encoder(dir) # 45
        pos=self.pos_encoder(pos) # 20
        pos2=self.pos_encoder(depth_).repeat(2,1,1,1) # 10

        pose=torch.cat([dir,pos,pos2],dim=1)

        # print(dir.shape)
        # print(pos.shape)
        # exit()

        # selective_features=[]
        # for target_pixel in target_pixels_list:
        #
        #     sub_features=features[:,:,target_pixel[0],target_pixel[1]]
        #     selective_features.append(sub_features)
        # selective_features=torch.stack(selective_features)
        # selective_features=selective_features.repeat(1,2,1)

        output=self.att_block(features,pose)

        return output


class YPG_D2(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = ConvFeatureExtractor(
            in_channels=1,
            activation=nn.LeakyReLU(0.01),  # Try different activations
            use_norm=False
        ).to('cuda')
        # self.back_bone = res_unet(in_c=1, Batch_norm=None, Instance_norm=False,
        #                           relu_negative_slope=0.01, activation=None, IN_affine=False).to('cuda')
        add_spectral_norm_selective(self.back_bone)

        # self.att_block = normalize_free_att_2d(in_c1=64, in_c2=7+1, out_c=1,
        #                                relu_negative_slope=0.2, activation=None,softmax_att=True).to('cuda')
        self.att_block = film_fusion_1d(in_c1=64, in_c2=65, out_c=1,
                                       relu_negative_slope=0., activation=nn.SiLU()).to('cuda')
        add_spectral_norm_selective(self.att_block)

        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, depth, pose,  detach_backbone=False):

        # print(depth[0].max())
        # print(depth[0].min())
        # exit()
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
        else:
            features = self.back_bone(depth)

        features=features.squeeze()[:,None,:].repeat(1,2,1)

        print('D max val= ',features.max().item())
        # pose=pose.permute(0,2,1)[:,:,:,None]
        dir=pose[:,:,:5]
        pos=pose[:,:,5:]
        # print(pos)

        pos=self.pos_encoder(pos) # 20
        # print(pos)
        # exit()
        dir=self.dir_encoder(dir) # 45

        # pos2=self.pos_encoder(depth_).repeat(2,1,1,1) # 10
        pose=torch.cat([dir,pos],dim=-1)
        # pose=pose.squeeze().permute(0,2,1)

        # print(features.shape)
        # print(pose.shape)
        # print(dir.shape)
        # print(pos.shape)
        # exit()

        # selective_features=[]
        # for target_pixel in target_pixels_list:
        #
        #     sub_features=features[:,:,target_pixel[0],target_pixel[1]]
        #     selective_features.append(sub_features)
        # selective_features=torch.stack(selective_features)
        # selective_features=selective_features.repeat(1,2,1)

        # print(features.shape)
        # print(pose.shape)

        output=self.att_block(features,pose)

        return output

class YPG_D_Point_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = PointNetA(use_instance_norm=True)

        replace_activations(self.back_bone, nn.ReLU, nn.LeakyReLU(0.01))
        # add_spectral_norm_selective(self.back_bone)

        # self.ln=nn.LayerNorm(128).to('cuda')
        # self.relu=nn.ReLU()

        self.att_block=film_fusion_1d(in_c1=128, in_c2=95+2, out_c=1,
                                              relu_negative_slope=0.,activation=nn.SiLU(),normalize=True).to('cuda')
        # add_spectral_norm_selective(self.att_block)

        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, pc,pose, detach_backbone=False):
        # print(pc.max(dim=1))
        # print(pc.min(dim=1))
        # exit()
        pc_=pc.clone()
        pc_=pc_-pc_.mean(dim=1)
        # pc_[:,:,0]-=0.43

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(pc_)
        else:
            features = self.back_bone(pc_)

        features=features.transpose(1,2).repeat(2,1,1)

        print('D max val= ',features.max().item())
        # pose=pose.permute(0,2,1)[:,:,:,None]
        dir=pose[:,:,:5]
        pos=pose[:,:,5:]
        # print(pos[0,0:2])
        pos=self.pos_encoder(pos) # 20
        # print(pos[0,0:2])
        # exit()
        dir=self.dir_encoder(dir) # 45
        pc_e=self.pos_encoder(pc_).repeat(2,1,1) # 30


        # pos2=self.pos_encoder(depth_).repeat(2,1,1,1) # 10
        pose=torch.cat([dir,pos,pc_e,pose[:,:,5:]],dim=-1)


        output=self.att_block(features,pose)


        return output
