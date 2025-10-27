from torch import nn
import torch.nn.functional as F
from GraspAgent_2.model.Backbones import PointNetA
from GraspAgent_2.model.Decoders import normalize_free_att_sins, normalized_att_1d, \
    normalize_free_att_2d, custom_att_1d, ParameterizedSine, normalize_free_att_1d, \
    custom_att_2d, normalized_att_2d, normalize_free_res_att_2d, normalized_res_att_2d, film_fusion_2d, film_fusion_1d
from GraspAgent_2.model.conv_140 import ConvFeatureExtractor
from GraspAgent_2.model.utils import replace_activations, add_spectral_norm_selective
from GraspAgent_2.utils.NN_tools import replace_instance_with_groupnorm
from GraspAgent_2.utils.model_init import init_norm_free_resunet, kaiming_init_all
from GraspAgent_2.utils.positional_encoding import PositionalEncoding_2d, LearnableRBFEncoding1d, PositionalEncoding_1d, \
    LearnableRBFEncoding2D
from models.decoders import sine, LayerNorm2D, att_res_conv_normalized
from models.resunet import res_unet,res_unet_encoder
from registration import  standardize_depth
import torch
import torch.nn as nn
YPG_model_key = 'YPG_model'
YPG_model_key2 = 'YPG_model2'
YPG_model_key3 = 'YPG_model3'

class LearnableBins(nn.Module):
    def __init__(self, min_val, max_val, N):
        super().__init__()
        # initialize N+1 positive widths
        widths = torch.ones(N)
        self.width_params = nn.Parameter(widths)

        self.min_val = min_val
        self.max_val = max_val
        self.N=N

        self.ini_bin_size=(max_val-min_val)/N


    def forward(self):
        # ensure positivity
        widths = F.softplus(self.width_params)

        # normalize total width to span desired range
        widths = widths / widths.sum() * (self.max_val - self.min_val)
        # convert widths to sorted bin edges
        bin_edges = self.min_val + torch.cumsum(widths, dim=0)

        return bin_edges

class WSConv2d(nn.Conv2d):
    """
    Weight-Standardized Conv2d
    """
    def forward(self, x):
        # Get weight
        w = self.weight
        # Compute per-output-channel mean and std
        mean = w.mean(dim=(1,2,3), keepdim=True)
        std = w.std(dim=(1,2,3), keepdim=True) + 1e-5
        # Standardize
        w_hat = (w - mean) / std
        # Perform convolution with standardized weight
        return F.conv2d(x, w_hat, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def replace_conv_with_wsconv(module):
    """
    Recursively replace all nn.Conv2d layers with WSConv2d
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Keep the same initialization parameters
            ws_conv = WSConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None)
            ).to('cuda')
            # Copy the original weights and bias
            ws_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                ws_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, ws_conv)
        else:
            # Recursively replace in child modules
            replace_conv_with_wsconv(child)

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

        self.dist_ = film_fusion_1d(in_c1=128, in_c2=3+3+2+10, out_c=1,
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

        encoded_width=self.pos_encoder(width)#.detach()


        dist = self.dist_(features, torch.cat([encoded_approach, encoded_beta, encoded_width,encoded_pc], dim=-1))

        # dist=F.sigmoid(dist)
        # width=F.sigmoid(width)

        pose = torch.cat([approach, beta, dist, width], dim=-1)

        return pose


class ParallelGripperPoseSampler(nn.Module):
    def  __init__(self):
        super().__init__()

        self.approach = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            ParameterizedSine(),
            nn.Conv2d(32, 16, kernel_size=1),
            ParameterizedSine(),
            nn.Conv2d(16, 3, kernel_size=1)
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.width_scale = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)
        self.dist_scale = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)

        # self.beta_decoder = normalize_free_att_sins(in_c1=128, in_c2=3, out_c=2).to(
        #     'cuda')
        # self.beta_decoder = normalize_free_att_2d(in_c1=64, in_c2=3+1, out_c=2,
        #                                   relu_negative_slope=0., activation=nn.SiLU(),softmax_att=False,use_sin=True).to(
        #     'cuda')
        self.beta_decoder = film_fusion_2d(in_c1=64, in_c2=3+1, out_c=2,
                                          relu_negative_slope=0., activation=None,use_sin=True).to(
            'cuda')
        self.width_ = film_fusion_2d(in_c1=64, in_c2=3+2+1, out_c=10,
                                          relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')

        self.dist_ = film_fusion_2d(in_c1=64, in_c2=3+2+1+10, out_c=10,
                                         relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')

        self.width_bin_centers=LearnableBins(min_val=0.,max_val=1.0,N=10).to(
            'cuda')
        self.dist_bin_centers=LearnableBins(min_val=0.0,max_val=1.0,N=10).to(
            'cuda')

        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction

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



        width_logits = self.width_(features, torch.cat([encoded_approach, encoded_beta,encoded_depth], dim=1))

        width_p=F.softmax(width_logits,dim=1)*torch.exp(self.width_scale)
        width_bin_centers=self.width_bin_centers().view(1, 10, 1, 1)

        width=(width_p*width_bin_centers).sum(dim=1,keepdim=True)
        encoded_width=self.pos_encoder(width)#.detach()


        dist_logits = self.dist_(features, torch.cat([encoded_approach, encoded_beta, encoded_width,encoded_depth], dim=1))
        dist_p=F.softmax(dist_logits,dim=1)*torch.exp(self.dist_scale)
        dist_bin_centers=self.dist_bin_centers().view(1, 10, 1, 1)
        dist=(dist_p*dist_bin_centers).sum(dim=1,keepdim=True)

        # dist=F.sigmoid(dist)
        # width=F.sigmoid(width)

        pose = torch.cat([approach, beta, dist, width], dim=1)
        
        # exit()

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


class YPG_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=False,
                                  relu_negative_slope=0.2,activation=None,IN_affine=False,scale=1.).to('cuda')



        # init_norm_free_resunet(self.back_bone)
        # add_spectral_norm_selective(self.back_bone)
        # replace_instance_with_groupnorm(self.back_bone, max_groups=16)


        self.PoseSampler = ParallelGripperPoseSampler()

        self.grasp_quality_=film_fusion_2d(in_c1=64, in_c2=20+10+5, out_c=1,
                                              relu_negative_slope=0.2,activation=None,normalize=False).to(
            'cuda')
        self.grasp_collision_ = film_fusion_2d(in_c1=64, in_c2=20+10+5, out_c=2,
                                              relu_negative_slope=0.2,activation=None,normalize=False).to(
            'cuda')
        add_spectral_norm_selective(self.grasp_quality_)
        add_spectral_norm_selective(self.grasp_collision_)

        self.background_detector_ = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            LayerNorm2D(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=1),
        ).to('cuda')
        # add_spectral_norm_selective(self.grasp_collision_)

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
        # dir=self.dir_encoder(dir) # 45
        pos=self.pos_encoder(pos) # 20
        encoded_depth=self.pos_encoder(depth_) # 10

        detached_gripper_pose_encoded=torch.cat([dir,pos,encoded_depth],dim=1)
        # print('test---')
        # cuda_memory_report()

        grasp_quality=self.grasp_quality_(features,detached_gripper_pose_encoded)
        # grasp_quality=self.sig(grasp_quality)
        # cuda_memory_report()

        grasp_collision=self.grasp_collision_(features,detached_gripper_pose_encoded)
        grasp_collision=self.sig(grasp_collision)

        background_detection=self.background_detector_(features)
        background_detection=self.sig(background_detection)

        return gripper_pose,grasp_quality,background_detection,grasp_collision

class YPG_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=None, Instance_norm=False,
                                  relu_negative_slope=0.01, activation=None, IN_affine=False).to('cuda')
        add_spectral_norm_selective(self.back_bone)

        # self.att_block = normalize_free_att_2d(in_c1=64, in_c2=7+1, out_c=1,
        #                                relu_negative_slope=0.2, activation=None,softmax_att=True).to('cuda')
        self.att_block = film_fusion_1d(in_c1=64, in_c2=5+20+10, out_c=1,
                                       relu_negative_slope=0.2, activation=None).to('cuda')
        add_spectral_norm_selective(self.att_block)

        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, depth, pose,pairs,  detach_backbone=False):
        depth_ = standardize_depth(depth)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth_)
        else:
            features = self.back_bone(depth_)

        # features=features.repeat(2,1,1,1)#.permute(0,2,3,1)

        print('D max val= ',features.max().item())
        features = features.view(1, 64, -1)
        depth_ = depth_.view(1, 1, -1)
        feature_list = []
        depth_list = []
        for pair in pairs:
            index = pair[0]
            feature_list.append(features[:, :, index])
            depth_list.append(depth_[:, :, index])

        feature_list = torch.cat(feature_list, dim=0)[:, None, :].repeat(1, 2, 1)  # n,2,64
        depth_list = torch.cat(depth_list, dim=0)[:, None, :].repeat(1, 2, 1)  # n,2,64

        approach_and_beta = pose[:, :, :5]
        dist_and_width = pose[:, :, 5:]

        dist_and_width = self.pos_encoder((dist_and_width + 1) / 2)  # 20
        encoded_depth = self.pos_encoder(depth_list)  # 10

        pose_ = torch.cat([approach_and_beta, dist_and_width, encoded_depth], dim=-1)

        output = self.att_block(feature_list, pose_)
        # dir=pose[:,:5,...]
        # pos=pose[:,5:,...]
        # dir=self.dir_encoder(dir) # 45
        # pos=self.pos_encoder(pos) # 20
        # pos2=self.pos_encoder(depth_).repeat(2,1,1,1) # 10
        #
        # pose=torch.cat([dir,pos,pos2],dim=1)
        #
        # # print(dir.shape)
        # # print(pos.shape)
        # # exit()
        #
        # # selective_features=[]
        # # for target_pixel in target_pixels_list:
        # #
        # #     sub_features=features[:,:,target_pixel[0],target_pixel[1]]
        # #     selective_features.append(sub_features)
        # # selective_features=torch.stack(selective_features)
        # # selective_features=selective_features.repeat(1,2,1)
        #
        # output=self.att_block(features,pose)

        return output



class YPG_D2(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = ConvFeatureExtractor(
            in_channels=1,
            activation=nn.LeakyReLU(0.2),  # Try different activations
            use_norm=False
        ).to('cuda')
        kaiming_init_all(self.back_bone, negative_slope=0.2)
        # # self.back_bone = res_unet(in_c=1, Batch_norm=None, Instance_norm=False,
        # #                           relu_negative_slope=0.01, activation=None, IN_affine=False).to('cuda')
        # add_spectral_norm_selective(self.back_bone)

        # self.back_bone = res_unet_encoder(in_c=1, Batch_norm=False, Instance_norm=False,
        #                           relu_negative_slope=0.2, activation=None, IN_affine=False,scale=0.1).to('cuda')
        # init_norm_free_resunet(self.back_bone)

        # add_spectral_norm_selective(self.back_bone)

        # self.att_block = normalize_free_att_2d(in_c1=64, in_c2=7+1, out_c=1,
        #                                relu_negative_slope=0.2, activation=None,softmax_att=True).to('cuda')
        self.att_block = film_fusion_1d(in_c1=64, in_c2=20+5, out_c=1,
                                       relu_negative_slope=0.2, activation=None).to('cuda')
        add_spectral_norm_selective(self.att_block)

        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, depth, pose,  detach_backbone=False):


        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth)
                # features2 = self.back_bone2(depth)

        else:
            features = self.back_bone(depth)
            # features2 = self.back_bone2(depth)


        features=features.squeeze()[:,None,:].repeat(1,2,1)

        print('D max val= ',features.max().item())
        # pose=pose.permute(0,2,1)[:,:,:,None]
        dir=pose[:,:,:5]
        pos=pose[:,:,5:]
        # print(pos)

        pos=self.pos_encoder(pos) # 20
        # print(pos)
        # exit()
        # dir=self.dir_encoder(dir) # 45

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
