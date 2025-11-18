from torch import nn
import torch.nn.functional as F
from GraspAgent_2.model.Backbones import PointNetA
from GraspAgent_2.model.Decoders import ParameterizedSine, \
    film_fusion_2d, film_fusion_1d, att_2d, att_1d, res_att_2d
from GraspAgent_2.model.utils import replace_activations, add_spectral_norm_selective
from GraspAgent_2.utils.NN_tools import replace_instance_with_groupnorm
from GraspAgent_2.utils.depth_processing import masked_sobel_gradients
from GraspAgent_2.utils.model_init import init_norm_free_resunet, kaiming_init_all, orthogonal_init_all, \
    init_orthogonal, init_weights_xavier, init_weights_he_normal, init_weights_normal, init_weights_xavier_normal
from GraspAgent_2.utils.positional_encoding import PositionalEncoding_2d, LearnableRBFEncoding1d, PositionalEncoding_1d, \
    LearnableRBFEncoding2D, EncodedScaler, depth_sin_cos_encoding
from lib.models_utils import view_parameters_value
from models.decoders import LayerNorm2D
from models.resunet import res_unet,res_unet_encoder
from registration import  standardize_depth
import torch
import torch.nn as nn
YPG_model_key = 'YPG_model'
YPG_model_key2 = 'YPG_model2'
YPG_model_key3 = 'YPG_model3'

class ParallelGripperPoseSampler2(nn.Module):
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
        # self.width_scale = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.dist_scale = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)

        # self.beta_decoder = normalize_free_att_sins(in_c1=128, in_c2=3, out_c=2).to(
        #     'cuda')
        # self.beta_decoder = normalize_free_att_2d(in_c1=64, in_c2=3+1, out_c=2,
        #                                   relu_negative_slope=0., activation=nn.SiLU(),softmax_att=False,use_sin=True).to(
        #     'cuda')
        self.beta_decoder = film_fusion_2d(in_c1=64, in_c2=27+10, out_c=2,
                                          relu_negative_slope=0., activation=None,use_sin=True).to(
            'cuda')
        self.width_ = film_fusion_2d(in_c1=64, in_c2=27+18+10, out_c=1,
                                          relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')


        self.dist_ = film_fusion_2d(in_c1=64, in_c2=27+18+10+10, out_c=1,
                                         relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')


        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, features,depth_):
        encoded_depth=self.pos_encoder(depth_)#.detach()

        vertical=torch.zeros_like(features[:,0:3])
        vertical[:,-1]+=1.

        # approach=vertical

        approach = self.approach(features)
        approach = F.normalize(approach, dim=1)
        approach = approach * self.scale + vertical * (1-self.scale)
        approach = F.normalize(approach, dim=1)

        encoded_approach=self.dir_encoder(approach)#.detach()

        beta = self.beta_decoder(features, torch.cat([encoded_approach, encoded_depth], dim=1))

        beta = F.normalize(beta, dim=1)

        encoded_beta=self.dir_encoder(beta)#.detach()


        width = self.width_(features, torch.cat([encoded_approach, encoded_beta,encoded_depth], dim=1))

        encoded_width=self.pos_encoder(width)#.detach()


        dist = self.dist_(features, torch.cat([encoded_approach, encoded_beta, encoded_width,encoded_depth], dim=1))


        dist=F.sigmoid(dist)
        width=F.sigmoid(width)

        pose = torch.cat([approach, beta, dist, width], dim=1)

        # print(torch.exp(self.width_scale))
        # print(torch.exp(self.dist_scale))
        #
        # exit()

        return pose

class ParallelGripperPoseSampler(nn.Module):
    def  __init__(self):
        super().__init__()

        self.approach = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            ParameterizedSine(),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=1),
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.width_scale = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.dist_scale = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)

        # self.beta_decoder = normalize_free_att_sins(in_c1=128, in_c2=3, out_c=2).to(
        #     'cuda')
        # self.beta_decoder = normalize_free_att_2d(in_c1=64, in_c2=3+1, out_c=2,
        #                                   relu_negative_slope=0., activation=nn.SiLU(),softmax_att=False,use_sin=True).to(
        #     'cuda')
        self.beta_decoder_ = film_fusion_2d(in_c1=64, in_c2=1, out_c=2+3,
                                          relu_negative_slope=0.2, activation=None,use_sin=True).to(
            'cuda')
        self.width_ = film_fusion_2d(in_c1=64, in_c2=27+18+1, out_c=1,
                                          relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')

        self.dist_ = film_fusion_2d(in_c1=64, in_c2=27+18+10+1, out_c=1,
                                         relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')

        # self.proj_approach_=nn.Conv2d(32, 3, kernel_size=1).to('cuda')
        # self.proj_beta_=nn.Conv2d(32, 2, kernel_size=1).to('cuda')
        # self.proj_width_=nn.Conv2d(32, 1, kernel_size=1).to('cuda')
        # self.proj_dist_=nn.Conv2d(32, 1, kernel_size=1).to('cuda')

        # self.normalize_approach_embedding= nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=1),
        #     nn.Softmax(dim=1)
        # ).to('cuda')
        # self.normalize_beta_embedding= nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=1),
        #     nn.Softmax(dim=1)
        # ).to('cuda')
        # self.normalize_width_embedding= nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=1),
        #     nn.Softmax(dim=1)
        # ).to('cuda')

        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction

    #     self.initialize_film_gen()
    #
    # def initialize_film_gen(self):
    #     self.proj_approach_.apply(lambda m: init_orthogonal(m, scale=1.0))
    #     self.proj_beta_.apply(lambda m: init_orthogonal(m, scale=1/4))
    #     self.proj_width_.apply(lambda m: init_orthogonal(m, scale=1/6))
    #     self.proj_dist_.apply(lambda m: init_orthogonal(m, scale=1/10))


    def forward(self, features,encoded_depth):
        # encoded_depth=self.pos_encoder(depth_) # 10
        # print(features.max())
        # print(features.min())
        # print(features.mean())
        # print(features.std())
        # exit()


        # approach = self.approach(features)
        # vertical = torch.zeros_like(features[:, 0:3])
        # vertical[:, -1] += 1.
        # approach = F.normalize(approach, dim=1)
        # approach = approach * self.scale + vertical * (1-self.scale)
        # approach = F.normalize(approach, dim=1)
        #
        # approach_encoding=self.dir_encoder(approach)
        # approach_embedding=self.normalize_approach_embedding(approach_embedding)

        beta_approach = self.beta_decoder_(features, encoded_depth)
        beta=beta_approach[:,0:2]
        approach=beta_approach[:,2:]


        approach = F.normalize(approach, dim=1)

        approach_encoding=self.dir_encoder(approach)
        # print(beta.max())
        # print(beta.min())
        # print(beta.std())
        # exit()

        beta = F.normalize(beta, dim=1)

        beta_encoding=self.dir_encoder(beta)
        # beta_embedding=self.normalize_beta_embedding(beta_embedding)

        width = self.width_(features, torch.cat([approach_encoding, beta_encoding,encoded_depth], dim=1))
        width_encoding=self.pos_encoder(width)
        # width_embedding=self.normalize_width_embedding(width_embedding)

        dist = self.dist_(features, torch.cat([approach_encoding, beta_encoding, width_encoding,encoded_depth], dim=1))

        # print(width.max())
        # print(width.min())
        # print(width.mean())
        # print(width.std())
        # print()
        #
        # print(dist.max())
        # print(dist.min())
        # print(dist.mean())
        # print(dist.std())
        # exit()

        # print(self.beta_decoder.film_gen_[0].bias[:64])
        # print(self.width_.film_gen_[0].bias[:64])
        # print(self.dist_.film_gen_[0].bias[:64])
        # exit()
        # approach=self.proj_approach_(approach_embedding)
        # beta=self.proj_beta_(beta_embedding)
        # width=self.proj_width_(width_embedding)
        # dist=self.proj_dist_(dist_embedding)

        dist=F.sigmoid(dist)
        width=F.sigmoid(width)

        pose = torch.cat([approach, beta, dist, width], dim=1)
        # pose_embedding=torch.cat([approach_embedding, beta_embedding, dist_embedding, width_embedding], dim=1)
        # exit()
        return pose#,pose_embedding.detach()


def depth_standardization(depth,mask):
    mean_ = depth[mask].mean()

    depth_ = (depth.clone() - mean_) / 30
    depth_[~mask] = 0.

    # print(depth.max())
    # print(depth.min())
    # print(depth.mean())
    # print(depth.std())
    #
    # print()
    # print(depth_.max())
    # print(depth_.min())
    # print(depth_.mean())
    # print(depth_.std())
    # exit()

    return depth_[None,None]


class YPG_G(nn.Module):
    def __init__(self):
        super().__init__()
        gain = torch.nn.init.calculate_gain('leaky_relu', 0.2)

        self.back_bone_ = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0.01,activation=nn.SiLU(),IN_affine=False).to('cuda')

        self.back_bone_.apply(init_weights_he_normal)
        # orthogonal_init_all(self.back_bone, gain=gain)

        # init_norm_free_resunet(self.back_bone)
        # add_spectral_norm_selective(self.back_bone)
        replace_instance_with_groupnorm(self.back_bone_, max_groups=16)

        self.back_bone2 = res_unet(in_c=2, Batch_norm=False, Instance_norm=False,
                                  relu_negative_slope=0.2, activation=nn.SiLU(), IN_affine=False).to('cuda')
        # replace_instance_with_groupnorm(self.back_bone2, max_groups=32)
        # add_spectral_norm_selective(self.back_bone2)
        orthogonal_init_all(self.back_bone2, gain=gain)

        self.PoseSampler_ = ParallelGripperPoseSampler()

        self.grasp_quality=film_fusion_2d(in_c1=64, in_c2=65+1+7, out_c=1,
                                              relu_negative_slope=0.2,activation=None,normalize=False).to(
            'cuda')
        self.grasp_collision = film_fusion_2d(in_c1=64, in_c2=65+1+7, out_c=2,
                                              relu_negative_slope=0.2,activation=None,normalize=False).to(
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

        # self.background_detector =res_att_2d(in_c1=64, in_c2=1, out_c=1,
        #               relu_negative_slope=0.2, activation=None, normalize=False).to(
        #     'cuda')
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

    def forward(self, depth,mask, detach_backbone=False):


        depth_ = depth_standardization(depth[0,0],mask[0,0])

        # local_diff3 = depth_ - F.avg_pool2d(depth_, kernel_size=3, stride=1, padding=1)
        # local_diff5 = depth_ - F.avg_pool2d(depth_, kernel_size=5, stride=1, padding=2)
        # local_diff7 = depth_ - F.avg_pool2d(depth_, kernel_size=7, stride=1, padding=3)
        #
        # Gx, Gy = masked_sobel_gradients(depth_,mask)

        # encoded_depth=torch.cat([depth_,Gx, Gy,local_diff3,local_diff5,local_diff7],dim=1)
        input=torch.cat([depth_,mask.float()],dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone_(input)#*scale
                features2 = self.back_bone2(input)#*scale

        else:
            features = self.back_bone_(input)#*scale
            features2 = self.back_bone2(input)#*scale

        print('G max val= ',features.max().item(),' f2: ',features2.max().item())
        # features2=torch.cat([features2,scaled_depth_,depth_],dim=1)
        # features=torch.cat([features,scaled_depth_,depth_],dim=1)

        gripper_pose=self.PoseSampler_(features,depth_)

        detached_gripper_pose=gripper_pose.detach().clone()
        dir=detached_gripper_pose[:,:5,...]
        pos=detached_gripper_pose[:,5:,...]
        pos=torch.clip(pos,0,1)
        encoded_dir=self.dir_encoder(dir) # 45
        encoded_pos=self.pos_encoder(pos) # 20
        # encoded_depth=self.pos_encoder(depth_) # 10
        detached_gripper_pose=torch.cat([encoded_dir,encoded_pos,dir,pos,depth_],dim=1)

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


        grasp_quality=self.grasp_quality(features2,detached_gripper_pose)
        # grasp_quality=self.grasp_quality(torch.cat([features2,detached_gripper_pose],dim=1))

        grasp_quality=self.sig(grasp_quality)
        # cuda_memory_report()

        grasp_collision=self.grasp_collision(features2,detached_gripper_pose)
        grasp_collision=self.sig(grasp_collision)

        background_detection=self.background_detector(features2)
        background_detection=self.sig(background_detection)

        # grasp_quality=torch.rand_like(gripper_pose[:,0:1])
        # background_detection=torch.rand_like(gripper_pose[:,0:1])
        # grasp_collision=torch.rand_like(gripper_pose[:,0:2])

        return gripper_pose,grasp_quality,background_detection,grasp_collision



class YPG_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=2, Batch_norm=None, Instance_norm=False,
                                  relu_negative_slope=0.2, activation=None, IN_affine=False).to('cuda')
        add_spectral_norm_selective(self.back_bone)
        self.back_bone.apply(init_weights_he_normal)


        self.att_block = film_fusion_1d(in_c1=64, in_c2=45+20+1, out_c=1,
                                       relu_negative_slope=0.2, activation=nn.SiLU(),normalize=False,with_gate=False,bias=False).to('cuda')

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

        add_spectral_norm_selective(self.att_block)

        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

    def forward(self, depth, pose,pairs,mask,  detach_backbone=False):
        # coords = torch.nonzero(mask[0,0], as_tuple=False)
        # attention_mask=torch.zeros_like(mask, dtype=torch.bool)
        # for pair in pairs:
        #     index = pair[0]
        #     pixel_index=coords[index]
        #     attention_mask[0,0,pixel_index[0],pixel_index[1]]=True

        depth_ = depth_standardization(depth[0,0],mask[0,0])

        # local_diff3 = depth_ - F.avg_pool2d(depth_, kernel_size=3, stride=1, padding=1)
        # local_diff5 = depth_ - F.avg_pool2d(depth_, kernel_size=5, stride=1, padding=2)
        # local_diff7 = depth_ - F.avg_pool2d(depth_, kernel_size=7, stride=1, padding=3)
        #
        # Gx, Gy = masked_sobel_gradients(depth_,mask)
        #
        # encoded_depth=torch.cat([depth_,Gx, Gy,local_diff3,local_diff5,local_diff7],dim=1)

        input = torch.cat([depth_, mask.float()], dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(input)#*scale
        else:
            features = self.back_bone(input)#*scale

        # features=features.repeat(2,1,1,1)#.permute(0,2,3,1)
        # features=torch.cat([features,scaled_depth_,depth_],dim=1)

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

        encoded_dist_and_width = self.pos_encoder(dist_and_width )  # 20
        # encoded_depth = self.pos_encoder(depth_list)  # 10
        encoded_approach_and_beta = self.dir_encoder(approach_and_beta)  # 45


        # condition=self.condition_projection(pose_)

        output = self.att_block( feature_list,torch.cat([encoded_approach_and_beta,encoded_dist_and_width,depth_list], dim=-1))
        # output = self.att_block2(output, torch.cat([ dist_and_width,depth_list], dim=-1))

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


