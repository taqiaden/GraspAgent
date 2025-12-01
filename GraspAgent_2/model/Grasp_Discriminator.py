import torch
from torch import nn
from GraspAgent_2.model.Backbones import PointNetA
from GraspAgent_2.model.Decoders import normalized_ContextGate_1d, normalize_free_ContextGate_1d, custom_ContextGate_1d
from GraspAgent_2.model.conv_140 import ConvFeatureExtractor
from GraspAgent_2.model.utils import add_spectral_norm_selective, replace_activations
from GraspAgent_2.model.voxel_3d_conv import VoxelBackbone
from models.resunet import ResNet


class D(nn.Module):
    def __init__(self,in_c2):
        super().__init__()
        self.back_bone = PointNetA(use_instance_norm=False)

        add_spectral_norm_selective(self.back_bone)
        replace_activations(self.back_bone, nn.ReLU, nn.LeakyReLU(0.01))

        self.att_block = normalize_free_ContextGate_1d(in_c1=128, in_c2=in_c2 , out_c=1,med_c=128,
                                           relu_negative_slope=0.,activation=nn.SiLU()).to(
            'cuda')
        add_spectral_norm_selective(self.att_block)

        # self.att_block = normalized_att(in_c1=128, in_c2=in_c2, out_c=1,med_c=128,
        #                                     relu_negative_slope=0., activation=nn.SiLU()).to(
        #     'cuda')

    def forward(self, pc, pose, detach_backbone=False):

        pc_=pc.clone()
        pc_[:,:,0]-=0.43

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(pc_)
        else:
            features = self.back_bone(pc_)

        # [b,128,n]
        features=features.repeat(2,1,1).transpose(1,2)

        '''decode'''
        output= self.att_block(features,pose)

        return output


class SubD(nn.Module):
    def __init__(self,in_c2):
        super().__init__()
        # self.back_bone = ResNet(in_c=1, Batch_norm=None, Instance_norm=False,
        #                           relu_negative_slope=0.01,activation=None,IN_affine=False).to('cuda')

        self.back_bone = ConvFeatureExtractor(
            in_channels=1,
            activation=nn.LeakyReLU(0.01),  # Try different activations
            use_norm=False
        ).to('cuda')
        add_spectral_norm_selective(self.back_bone)

        self.att_block = custom_ContextGate_1d(in_c1=64, in_c2=in_c2 , out_c=1,med_c=64,
                                           relu_negative_slope=0.,activation=nn.SiLU()).to('cuda')
        add_spectral_norm_selective(self.att_block)

        # self.att_block = normalized_att(in_c1=128, in_c2=in_c2, out_c=1,med_c=128,
        #                                     relu_negative_slope=0., activation=nn.SiLU()).to(
        #     'cuda')

    def forward(self, occupancy_grids, pose, detach_backbone=False):

        '''backbone'''
        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(occupancy_grids)
        else:
            features = self.back_bone(occupancy_grids)

        # print(features.shape)
        # print(pose.shape)

        # [b,256]
        features=features.squeeze()[:,None,:].repeat(1,2,1)

        '''decode'''
        output= self.att_block(features,pose)
        # print(output.shape)
        return output
