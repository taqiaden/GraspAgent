import math

from torch import nn
import torch.nn.functional as F
from GraspAgent_2.model.Decoders import ParameterizedSine, \
    film_fusion_2d, film_fusion_1d, ContextGate_2d, ContextGate_1d, res_ContextGate_2d, siren
from GraspAgent_2.model.sparse_encoder import SparseEncoderIN
from GraspAgent_2.model.utils import replace_activations, add_spectral_norm_selective
from GraspAgent_2.utils.NN_tools import replace_instance_with_groupnorm
from GraspAgent_2.utils.depth_processing import masked_sobel_gradients
from GraspAgent_2.utils.model_init import init_norm_free_resunet, kaiming_init_all, orthogonal_init_all, \
    init_orthogonal, init_weights_xavier, init_weights_he_normal, init_weights_normal, init_weights_xavier_normal, \
    init_simplex_diversity, scaled_he_init_all
from GraspAgent_2.utils.positional_encoding import PositionalEncoding_2d, LearnableRBFEncoding1d, PositionalEncoding_1d, \
    LearnableRBFEncoding2D, EncodedScaler, depth_sin_cos_encoding
from models.decoders import LayerNorm2D
from models.resunet import res_unet, res_unet_encoder
from registration import standardize_depth
import torch
import torch.nn as nn

YPG_model_key = 'YPG_model'
YPG_model_key2 = 'YPG_model2'
YPG_model_key3 = 'YPG_model3'


class ParallelGripperPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.approach = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            LayerNorm2D(32),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=1)
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.width_scale = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.dist_scale = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)

        # self.beta_decoder = normalize_free_att_sins(in_c1=128, in_c2=3, out_c=2).to(
        #     'cuda')
        # self.beta_decoder = normalize_free_ContextGate_2d(in_c1=64, in_c2=3+1, out_c=2,
        #                                   relu_negative_slope=0., activation=nn.SiLU(),softmax_att=False,use_sin=True).to(
        #     'cuda')
        self.beta_decoder = ContextGate_2d(in_c1=64, in_c2=1 + 3, out_c=2,
                                           relu_negative_slope=0.1, activation=None, use_sin=False, bias=False,cyclic=False).to(
            'cuda')
        self.width_ = ContextGate_2d(in_c1=64, in_c2=1 + 3 + 2, out_c=1*3,
                                     relu_negative_slope=0.1, activation=None, use_sin=False).to(
            'cuda')

        self.dist_ = ContextGate_2d(in_c1=64, in_c2=1 + 3 + 2 + 1, out_c=1*3,
                                    relu_negative_slope=0.1, activation=None, use_sin=False).to(
            'cuda')

        self.pos_encoder = LearnableRBFEncoding2D(num_centers=10, init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d(num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, features, depth_):
        # encoded_depth=self.pos_encoder(depth_)#.detach()

        vertical = torch.zeros_like(features[:, 0:3])
        vertical[:, -1] += 1.

        # approach=vertical

        approach = self.approach(features)
        approach = F.normalize(approach, p=2,dim=1,eps=1e-8)
        approach = approach * self.scale + vertical * (1 - self.scale)
        approach = F.normalize(approach, p=2,dim=1,eps=1e-8)

        # encoded_approach=self.dir_encoder(approach)#.detach()

        beta = self.beta_decoder(features, torch.cat([approach, depth_], dim=1))

        beta = F.normalize(beta, p=2,dim=1,eps=1e-8)

        # encoded_beta=self.dir_encoder(beta)#.detach()

        width = self.width_(features, torch.cat([approach, beta, depth_], dim=1))
        width = F.normalize(width, p=2,dim=1,eps=1e-8)[:,0:1]#.sum(dim=1,keepdim=True)
        # width += math.sqrt(3)
        # width /= 2 * math.sqrt(3)
        # encoded_width=self.pos_encoder(width)#.detach()

        dist = self.dist_(features, torch.cat([approach, beta, width, depth_], dim=1))
        dist = F.normalize(dist, p=2,dim=1,eps=1e-8)[:,0:1]#.sum(dim=1,keepdim=True)
        # dist += math.sqrt(3)
        # dist /= 2 * math.sqrt(3)

        # dist=F.sigmoid(dist)
        # width=F.sigmoid(width)

        pose = torch.cat([approach, beta, dist, width], dim=1)

        # print(torch.exp(self.width_scale))
        # print(torch.exp(self.dist_scale))
        #
        # exit()

        return pose


class YPG_G(nn.Module):
    def __init__(self):
        super().__init__()
        gain = torch.nn.init.calculate_gain('leaky_relu', 0.2)

        self.back_bone_ = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                   relu_negative_slope=0.2, activation=nn.SiLU(), IN_affine=True,activate_skip=False).to('cuda')

        self.back_bone_.apply(init_weights_he_normal)
        # orthogonal_init_all(self.back_bone, gain=gain)

        # init_norm_free_resunet(self.back_bone)
        # add_spectral_norm_selective(self.back_bone_)
        # replace_instance_with_groupnorm(self.back_bone_, max_groups=16)

        self.back_bone2 = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                   relu_negative_slope=0.2, activation=None, IN_affine=False,activate_skip =False).to('cuda')
        replace_instance_with_groupnorm(self.back_bone2, max_groups=16)
        # add_spectral_norm_selective(self.back_bone2)
        # orthogonal_init_all(self.back_bone2, gain=gain)
        self.back_bone2.apply(init_weights_he_normal)

        self.PoseSampler_ = ParallelGripperPoseSampler()
        self.PoseSampler_2 = ParallelGripperPoseSampler()
        # scaled_he_init_all( self.PoseSampler_ )
        # scaled_he_init_all( self.PoseSampler_2 )


        self.grasp_quality = res_ContextGate_2d(in_c1=64, in_c2=8, out_c=1,
                                                relu_negative_slope=0.1, activation=None).to(
            'cuda')
        self.grasp_collision = res_ContextGate_2d(in_c1=64, in_c2=8, out_c=2,
                                                  relu_negative_slope=0.1, activation=None).to(
            'cuda')

        # self.grasp_quality = nn.Sequential(
        #     nn.Conv2d(64+65+1, 64, kernel_size=1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(64, 32, kernel_size=1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, kernel_size=1),
        # ).to('cuda')

        # add_spectral_norm_selective(self.grasp_quality)
        # add_spectral_norm_selective(self.grasp_collision)

        # self.background_detector =res_ContextGate_2d(in_c1=64, in_c2=1, out_c=1,
        #               relu_negative_slope=0.2, activation=None, normalize=False).to(
        #     'cuda')
        self.background_detector = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            # LayerNorm2D(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            # LayerNorm2D(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=1),
        ).to('cuda')
        # add_spectral_norm_selective(self.grasp_collision_)

        self.sig = nn.Sigmoid()

        self.pos_encoder = LearnableRBFEncoding2D(num_centers=10, init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d(num_freqs=4)  # for 2D/3D viewing direction

        # self.D=YPG_D_Decoder()

    def forward(self, depth, mask, backbone=None, pairs=None, detach_backbone=False):

        depth_ = depth_standardization(depth)

        # local_diff3 = depth_ - F.avg_pool2d(depth_, kernel_size=3, stride=1, padding=1)
        # local_diff5 = depth_ - F.avg_pool2d(depth_, kernel_size=5, stride=1, padding=2)
        # local_diff7 = depth_ - F.avg_pool2d(depth_, kernel_size=7, stride=1, padding=3)
        #
        # Gx, Gy = masked_sobel_gradients(depth_,mask)

        # encoded_depth=torch.cat([depth_,Gx, Gy,local_diff3,local_diff5,local_diff7],dim=1)
        input = torch.cat([depth_, mask], dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone_(input)  #if backbone is None else backbone(input)
                features2 = self.back_bone2(input)#*scale

        else:
            features = self.back_bone_(input)  #if backbone is None else backbone(input)
            features2 = self.back_bone2(input)#*scale

        print('G max val= ', features.max().item(), 'mean:', features.mean().item(), ' std:',
              features.std(dim=1).mean().item())
        # features2=torch.cat([features2,scaled_depth_,depth_],dim=1)
        # features=torch.cat([features,scaled_depth_,depth_],dim=1)

        gripper_pose = self.PoseSampler_(features, depth_)
        gripper_pose2 = self.PoseSampler_2(features, depth_)

        detached_gripper_pose = gripper_pose.detach().clone()
        # dir=detached_gripper_pose[:,:5,...]
        # pos=detached_gripper_pose[:,5:,...]
        # pos=torch.clip(pos,0,1)
        # encoded_dir=self.dir_encoder(dir) # 45
        # encoded_pos=self.pos_encoder(pos) # 20
        # encoded_depth=self.pos_encoder(depth_) # 10
        detached_gripper_pose = torch.cat([detached_gripper_pose, depth_], dim=1)

        # print(depth_.mean())
        # print(depth_.max())
        # print(depth_.min())
        # print(encoded_depth[0,:,100,100])
        # print(encoded_depth[0,:,200,200])
        #
        # exit()

        # detached_gripper_pose_encoded=torch.cat([dir,pos,encoded_depth],dim=1)
        # print('test---')
        # cuda_memory_report()

        # pose_embedding=torch.cat([pose_embedding,encoded_depth,depth_],dim=1)

        grasp_quality = self.grasp_quality(features2, detached_gripper_pose)
        # grasp_quality=self.grasp_quality(torch.cat([features2,detached_gripper_pose],dim=1))

        grasp_quality = self.sig(grasp_quality)
        # cuda_memory_report()

        grasp_collision = self.grasp_collision(features2, detached_gripper_pose)
        grasp_collision = self.sig(grasp_collision)

        background_detection = self.background_detector(features2)
        background_detection = self.sig(background_detection)

        # grasp_quality=torch.rand_like(gripper_pose[:,0:1])
        # background_detection=torch.rand_like(gripper_pose[:,0:1])
        # grasp_collision=torch.rand_like(gripper_pose[:,0:2])

        # self.D(pairs, features,depth_)

        return gripper_pose, grasp_quality, background_detection, grasp_collision,gripper_pose2


def depth_standardization(depth):
    # mean_ = depth[mask].mean()
    mask = depth != 0
    mean_ = 1265
    # print(f'mean_: {mean_}')
    depth_ = (depth.clone() - mean_) / 30
    depth_[~mask] = 0.

    return depth_

class MahalanobisDistance(nn.Module):
    def __init__(self, dim=64, out_dim=None, normalize=True):
        """
        dim: input feature dimension (64)
        out_dim: projected dimension (default = dim)
        normalize: whether to L2-normalize inputs before distance
        """
        super().__init__()
        out_dim =  dim if out_dim is None else out_dim
        self.normalize = normalize

        # W defines M = W^T W
        self.W = nn.Linear(dim, out_dim, bias=False)

        # Small-gain initialization for stability
        nn.init.kaiming_normal_(self.W.weight, nonlinearity="linear")
        self.W.weight.data *= 0.5

    def forward(self, main, others):
        """
        main:   [B, 1, 64]
        others: [B, N, 64]

        returns:
            dist: [B, N]
        """
        if self.normalize:
            main = F.normalize(main, dim=-1)
            others = F.normalize(others, dim=-1)

        # Broadcast main to [B, N, 64]
        diff = main - others          # [B, N, 64]

        # Apply learned transform
        z = self.W(diff)              # [B, N, out_dim]

        # Squared Mahalanobis distance
        dist = (z * z).sum(dim=-1)    # [B, N]

        return dist

class NaivePointBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Linear(3, 64),
            # nn.InstanceNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            # nn.InstanceNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            # nn.InstanceNorm1d(256),
        ).to('cuda')

    def forward(self, pc):
        x=self.enc1(pc).max(dim=1)[0]
        return x

def replace_activations(module, new_act_fn):
    """
    Recursively replace activation layers in a module.

    Args:
        module (nn.Module): root module
        new_act_fn (callable): function returning a new activation instance
                              e.g. lambda: nn.LeakyReLU(0.1, inplace=True)
    """
    for name, child in module.named_children():
        # If child is an activation, replace it
        if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.GELU, nn.ELU)):
            setattr(module, name, new_act_fn)
        else:
            replace_activations(child, new_act_fn)
class YPG_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=2, Batch_norm=None, Instance_norm=False,
                                  relu_negative_slope=0.2, activation=None, IN_affine=False).to('cuda')
        # add_spectral_norm_selective(self.back_bone)
        # replace_instance_with_groupnorm(self.back_bone, max_groups=16)
        # gan_init_with_norms(self.back_bone)


        # add_spectral_norm_selective(self.back_bone)

        self.back_bone.apply(init_weights_he_normal)

        # self.point_net_backbone=PointNetEncoder(use_instance_norm=True).to('cuda')
        # replace_activations(self.point_net_backbone,nn.LeakyReLU(0.2))
        # self.point_net_backbone.apply(init_weights_he_normal)

        self.sparse_encoder=SparseEncoderIN().to('cuda')
        self.sparse_encoder.apply(init_weights_he_normal)
        # add_spectral_norm_selective(self.point_net_backbone)

        # self.dist=MahalanobisDistance(dim=64,normalize=True).to('cuda')

        # self.point_net_backbone=NaivePointBackbone().to('cuda')

        # self.att_block = film_fusion_1d(in_c1=64, in_c2=45+20+1, out_c=1,
        #                                relu_negative_slope=0.2, activation=nn.SiLU(),normalize=False,with_gate=False,bias=False).to('cuda')
        self.att_block_ = ContextGate_1d(in_c1=512, in_c2=7, out_c=1).to('cuda')
        # self.att_block_.apply(init_weights_he_normal)

        # self.condition_projection= nn.Sequential(
        #     nn.Linear(45+20, 256),
        #     nn.SiLU()
        # ).to('cuda')

        # self.att_block = nn.Sequential(
        #     nn.Linear(64 + 45 +20+ 1+7, 128),
        #     nn.SiLU(),
        #     nn.Linear(128, 64),
        #     nn.SiLU(),
        #     nn.Linear(64, 1),
        # ).to('cuda')

        # add_spectral_norm_selective(self.att_block)

        # self.cond_proj1 = nn.Sequential(
        #     nn.Linear(7, 128),
        #     nn.SiLU(),
        #     nn.Linear(128, 128),
        #     nn.SiLU(),
        #     nn.Linear(128, 64),
        # ).to('cuda')
        #
        # self.cond_proj2 = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.SiLU(),
        #     nn.Linear(128, 64),
        # ).to('cuda')
        #
        # self.contx_proj1 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.SiLU(),
        #     # nn.Linear(128, 5),
        #     # nn.SiLU(),
        #     nn.Linear(128, 64),
        # ).to('cuda')
        #
        # self.contx_proj2 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.SiLU(),
        #     # nn.Linear(128, 5),
        #     # nn.SiLU(),
        #     nn.Linear(128, 64),
        # ).to('cuda')



        # scaled_he_init_all( self.cond_proj1 )
        # scaled_he_init_all( self.cond_proj2 )
        # scaled_he_init_all( self.contx_proj1 )
        # scaled_he_init_all( self.contx_proj2 )


        self.pos_encoder = LearnableRBFEncoding1d(num_centers=10, init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d(num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, depth, pose, pairs, mask, cropped_spheres,backbone=None, detach_backbone=False):
        # coords = torch.nonzero(mask[0,0], as_tuple=False)
        # attention_mask=torch.zeros_like(mask, dtype=torch.bool)
        # for pair in pairs:
        #     index = pair[0]
        #     pixel_index=coords[index]
        #     attention_mask[0,0,pixel_index[0],pixel_index[1]]=True
        # point_global_features=[]
        # for sub_pc in cropped_spheres:
        if detach_backbone:
            with torch.no_grad():
                point_global_features=self.sparse_encoder(cropped_spheres)
        else:
            point_global_features=self.sparse_encoder(cropped_spheres)

        # print(point_global_features.shape)
        # print(point_global_features)

            # print(sub_pc.shape) [n,3]
            # point_global_features.append(point_global_features)
        # point_global_features=torch.cat(point_global_features,dim=0)

        print('D max val= ', point_global_features.max().item(), 'mean:', point_global_features.mean().item(), ' std:',
                    point_global_features.std(dim=1).mean().item())


        depth_ = depth_standardization(depth)

        # local_diff3 = depth_ - F.avg_pool2d(depth_, kernel_size=3, stride=1, padding=1)
        # local_diff5 = depth_ - F.avg_pool2d(depth_, kernel_size=5, stride=1, padding=2)
        # local_diff7 = depth_ - F.avg_pool2d(depth_, kernel_size=7, stride=1, padding=3)
        #
        # Gx, Gy = masked_sobel_gradients(depth_,mask)
        #
        # encoded_depth=torch.cat([depth_,Gx, Gy,local_diff3,local_diff5,local_diff7],dim=1)

        # input = torch.cat([depth_, mask], dim=1)

        # if detach_backbone:
        #     with torch.no_grad():
        #         features = self.back_bone(input)  #if backbone is None else backbone(input)
        # else:
        #     features = self.back_bone(input)  #if backbone is None else backbone(input)

        # features=features.repeat(2,1,1,1)#.permute(0,2,3,1)
        # features=torch.cat([features,scaled_depth_,depth_],dim=1)
        # print('D max val= ', features.max().item(), 'mean:', features.mean().item(), ' std:',
        #       features.std(dim=1).mean().item())

        # features = features.view(1, 64, -1)
        depth_ = depth_.view(1, 1, -1)
        # feature_list = []
        depth_list = []
        for pair in pairs:
            index = pair[0]
            # feature_list.append(features[:, :, index])
            depth_list.append(depth_[:, :, index])

        # feature_list = torch.cat(feature_list, dim=0)[:, None, :].repeat(1, 2, 1)  # n,2,64
        # anchor = torch.cat(feature_list, dim=0)  # n,64

        # depth_list = torch.cat(depth_list, dim=0)[:, None, :].repeat(1, 2, 1)  # n,2,64
        # anchor=torch.cat([anchor,depth_list[:,0]],dim=-1)

        output = self.att_block_( point_global_features[:,None],pose)
        # positive_negative = self.cond_proj1(pose)
        # positive_negative = F.normalize(positive_negative, p=2, dim=-1, eps=1e-8)
        # positive_negative=torch.softmax(positive_negative,dim=-1)
        # positive_negative = self.cond_proj2(positive_negative)
        # positive_negative = F.normalize(positive_negative, p=2, dim=-1, eps=1e-8)
        # point_global_features = F.normalize(point_global_features, p=2, dim=-1, eps=1e-8)

        # anchor = self.contx_proj1(point_global_features)
        # anchor = F.normalize(anchor, p=2, dim=-1, eps=1e-8)
        # print(anchor.shape)
        # print(point_global_features.shape)
        # print(positive_negative.shape)

        # print(anchor[0])
        # print(anchor[1])
        # exit()

        # scores = ((anchor[:, None, :] - positive_negative) ** 2).sum(dim=-1)

        # scores=(anchor[:,None]*positive_negative).sum(dim=-1)

        # anchor=torch.softmax(anchor,dim=-1)
        # anchor = self.contx_proj2(anchor)
        # anchor = F.normalize(anchor, p=2, dim=-1, eps=1e-8)

        return None, None,output
