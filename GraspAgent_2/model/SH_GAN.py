import torch.nn.functional as F
from GraspAgent_2.model.Decoders import normalize_free_att_sins, normalized_att_1d, \
    normalize_free_att_2d, custom_att_1d, ParameterizedSine, normalize_free_att_1d, \
    custom_att_2d, normalized_att_2d, normalize_free_res_att_2d, normalized_res_att_2d, film_fusion_2d, film_fusion_1d
from GraspAgent_2.model.YPG_GAN import LearnableRBFEncoding2D, LearnableRBFEncoding1d
from GraspAgent_2.model.conv_140 import ConvFeatureExtractor
from GraspAgent_2.model.utils import replace_activations, add_spectral_norm_selective
from GraspAgent_2.model.Grasp_Discriminator import D, SubD
from GraspAgent_2.utils.NN_tools import replace_instance_with_groupnorm
from GraspAgent_2.utils.positional_encoding import PositionalEncoding_2d, PositionalEncoding_1d
from GraspAgent_2.utils.quat_encoding import sign_invariant_quat_encoding_2d
from models.decoders import sine, LayerNorm2D, att_res_conv_normalized
from models.resunet import res_unet
import torch
import torch.nn as nn
SH_model_key = 'SH_model'



def sign_invariant_quat_encoding_1d(q):
    """
    Sign-invariant encoding for a batch of quaternion sets.

    Args:
        q: torch.Tensor of shape [B, N, 4]
           Each quaternion is (w, x, y, z).

    Returns:
        encoded: torch.Tensor of shape [B, N, 10]
                 Sign-invariant quadratic features per quaternion.
    """
    # Normalize quaternions along the last dimension
    q = F.normalize(q, dim=-1)

    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute quadratic, sign-invariant terms
    encoded = torch.stack([
        qw * qw,      # 1
        qx * qx,      # 2
        qy * qy,      # 3
        qz * qz,      # 4
        2 * qw * qx,  # 5
        2 * qw * qy,  # 6
        2 * qw * qz,  # 7
        2 * qx * qy,  # 8
        2 * qy * qz,  # 9
        2 * qz * qx,  # 10
    ], dim=-1)  # stack along feature dimension → [B, N, 10]

    return encoded

class ParallelGripperPoseSampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device='cuda'), requires_grad=True)

        # self.beta_decoder = normalize_free_att_sins(in_c1=128, in_c2=3, out_c=2).to(
        #     'cuda')
        # self.beta_decoder = normalize_free_att_2d(in_c1=64, in_c2=3+1, out_c=2,
        #                                   relu_negative_slope=0., activation=nn.SiLU(),softmax_att=False,use_sin=True).to(
        #     'cuda')
        # self.quat_mean = nn.Parameter(torch.tensor([.0,1.,0.,0.], dtype=torch.float32, device='cuda'), requires_grad=True).view(1,-1,1,1)
        # self.quat_log_std = nn.Parameter(torch.tensor([.0,0.,0.,0.], dtype=torch.float32, device='cuda'), requires_grad=True).view(1,-1,1,1)

        self.quat = film_fusion_2d(in_c1=64, in_c2=1, out_c=4,
                                          relu_negative_slope=0., activation=None,use_sin=True).to(
            'cuda')

        self.beta=film_fusion_2d(in_c1=64, in_c2=10+10, out_c=3,
                                          relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')
        self.fingers_main_joints = film_fusion_2d(in_c1=64, in_c2=10+10+30, out_c=3,
                                          relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')

        self.fingers_tip_joints = film_fusion_2d(in_c1=64, in_c2=10+10+30+30, out_c=3,
                                          relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')

        self.transition=film_fusion_2d(in_c1=64, in_c2=10+10+90, out_c=1,
                                          relu_negative_slope=0.2, activation=None,normalize=False).to(
            'cuda')

        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d(num_freqs=4)  # for 2D/3D viewing direction

        # self.ln=LayerNorm2D(64).to('cuda')

    def forward(self, features,depth_):
        encoded_depth=self.pos_encoder(depth_) #10
        # features=self.ln(features)

        quat = self.quat(features, depth_)

        quat = F.normalize(quat, dim=1)

        encoded_quat=sign_invariant_quat_encoding_2d(quat) #10

        beta = self.beta(features, torch.cat([encoded_depth,encoded_quat], dim=1))
        beta=F.tanh(beta)
        encoded_beta=self.pos_encoder((beta+1)/2)


        fingers_main_joints = self.fingers_main_joints(features, torch.cat([encoded_depth,encoded_quat,encoded_beta], dim=1))
        fingers_main_joints=F.tanh(fingers_main_joints)
        encoded_fingers_main_joints=self.pos_encoder((fingers_main_joints+1)/2)


        fingers_tip_joints = self.fingers_tip_joints(features, torch.cat([encoded_depth,encoded_quat,encoded_beta,encoded_fingers_main_joints], dim=1))
        fingers_tip_joints=F.tanh(fingers_tip_joints)
        encoded_fingers_tip_joints=self.pos_encoder((fingers_tip_joints+1)/2)


        transition=self.transition(features,torch.cat([encoded_depth,encoded_quat,encoded_beta,encoded_fingers_main_joints,encoded_fingers_tip_joints], dim=1))
        transition=F.tanh(transition)

        pose = torch.cat([quat,beta,fingers_main_joints,fingers_tip_joints,transition], dim=1)



        return pose

class SH_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=False, Instance_norm=False,
                                  relu_negative_slope=0.2,activation=None,IN_affine=False).to('cuda')
        add_spectral_norm_selective(self.back_bone)

        self.back_bone2 = res_unet(in_c=1, Batch_norm=False, Instance_norm=False,
                                  relu_negative_slope=0.2, activation=None, IN_affine=False).to('cuda')
        replace_instance_with_groupnorm(self.back_bone2, max_groups=16)
        # replace_instance_with_groupnorm(self.back_bone2, max_groups=16)


        self.PoseSampler = ParallelGripperPoseSampler()

        self.grasp_quality_=film_fusion_2d(in_c1=64, in_c2=10+100+10, out_c=1,
                                              relu_negative_slope=0.2,activation=None,normalize=True).to(
            'cuda')
        self.grasp_collision_ = film_fusion_2d(in_c1=64, in_c2=10+100+10, out_c=2,
                                              relu_negative_slope=0.2,activation=None,normalize=True).to(
            'cuda')
        # add_spectral_norm_selective(self.grasp_quality_)
        # add_spectral_norm_selective(self.grasp_collision_)

        self.sig=nn.Sigmoid()

        self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction


    def forward(self, depth, detach_backbone=False):

        depth_= (depth-1.2)*10

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth_)
                features2 = self.back_bone2(depth_)

        else:
            features = self.back_bone(depth_)
            features2 = self.back_bone2(depth_)

        # print(features[0,:,0,0])
        # # print(features2[0,:,0,0])
        # print(features[0,:,0,10])
        # # print(features2[0,:,0,10])
        # exit()
        # features2=torch.cat([features2,features.detach()],dim=1)


        print('G max val= ',features.max().item())

        gripper_pose=self.PoseSampler(features,depth_)

        detached_gripper_pose=gripper_pose.detach().clone()
        quat=detached_gripper_pose[:,:4,...]
        fingers_and_transition=detached_gripper_pose[:,4:,...]
        quat=sign_invariant_quat_encoding_2d(quat) # 10
        fingers_and_transition=self.pos_encoder((fingers_and_transition+1)/2) # 100
        encoded_depth=self.pos_encoder(depth_) # 10

        detached_gripper_pose_encoded=torch.cat([quat,fingers_and_transition,encoded_depth],dim=1)

        grasp_quality=self.grasp_quality_(features2,detached_gripper_pose_encoded)
        grasp_quality=self.sig(grasp_quality)

        grasp_collision=self.grasp_collision_(features2,detached_gripper_pose_encoded)
        grasp_collision=self.sig(grasp_collision)

        return gripper_pose,grasp_quality,grasp_collision


class SH_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=None, Instance_norm=False,
                                  relu_negative_slope=0.2, activation=None, IN_affine=False).to('cuda')
        add_spectral_norm_selective(self.back_bone)

        # self.att_block = normalize_free_att_2d(in_c1=64, in_c2=7+1, out_c=1,
        #                                relu_negative_slope=0.2, activation=None,softmax_att=True).to('cuda')
        self.att_block = film_fusion_1d(in_c1=64, in_c2=10+100+10, out_c=1,
                                       relu_negative_slope=0.2, activation=None).to('cuda')
        add_spectral_norm_selective(self.att_block)

        # self.pos_encoder = LearnableRBFEncoding2D( num_centers=10,init_sigma=0.1)  # for 3D position
        # self.dir_encoder = PositionalEncoding_2d( num_freqs=4)  # for 2D/3D viewing direction

        self.pos_encoder = LearnableRBFEncoding1d( num_centers=10,init_sigma=0.1)  # for 3D position
        self.dir_encoder = PositionalEncoding_1d( num_freqs=4)  # for 2D/3D viewing direction

        self.ln=LayerNorm2D(64).to('cuda')

    def forward(self, depth, pose,pairs,  detach_backbone=False):
        # print(depth.shape)
        # print(pose.shape)
        # exit()
        depth_= (depth-1.2)*10

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(depth_)
        else:
            features = self.back_bone(depth_)
        print('D max val= ',features.max().item())
        # features=self.ln(features)
        features=features.view(1,64,-1)
        depth_=depth_.view(1,1,-1)
        feature_list=[]
        depth_list=[]
        for pair in pairs:
            index=pair[0]
            feature_list.append(features[:,:,index])
            depth_list.append(depth_[:,:,index])

        feature_list=torch.cat(feature_list,dim=0)[:,None,:].repeat(1,2,1) # n,2,64
        depth_list=torch.cat(depth_list,dim=0)[:,None,:].repeat(1,2,1) # n,2,64

        quat = pose[:,:, :4]
        fingers_and_transition = pose[:,:, 4:]

        encoded_quat = sign_invariant_quat_encoding_1d(quat)  # 36
        encoded_fingers_and_transition = self.pos_encoder((fingers_and_transition+1)/2)  # 100
        encoded_depth=self.pos_encoder(depth_list) # 10

        pose_ = torch.cat([encoded_quat,encoded_fingers_and_transition,  encoded_depth], dim=-1)

        output = self.att_block(feature_list, pose_)

        # features=features.repeat(2,1,1,1)#.permute(0,2,3,1)
        #
        # quat=pose[:,:4,...]
        # fingers=pose[:,4:4+12,...]
        # transition=pose[:,4+12:,...]
        #
        # quat=self.dir_encoder(quat) # 36
        # fingers=self.pos_encoder(fingers) # 120
        # transition=self.pos_encoder(transition) # 30
        # depth_=self.pos_encoder(depth).repeat(2,1,1,1) # 10
        #
        # pose=torch.cat([quat,fingers,transition,depth_],dim=1)
        #
        # output=self.att_block(features,pose)

        return output

