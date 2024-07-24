import torch
from torch import nn
import torch.nn.functional as F
from Configurations.config import ref_pc_center
from lib.models_utils import reshape_for_layer_norm
from models.pointnet2_backbone import PointNetbackbone

opening_generator_path=r'opening_generator'
opening_critic_path=r'opening_critic'

class G_decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32),
        ).to('cuda')
        self.query = nn.Sequential(
            nn.Linear(6, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 32),
        ).to('cuda')
        self.value = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32),
        ).to('cuda')

        self.get_opening= nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 1)
        ).to('cuda')

        self.residuals = nn.Sequential(
            nn.Linear(128 + 6 , 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 16),
        ).to('cuda')

        self.attention = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 16),
        ).to('cuda')

    def forward(self, representation_128,parameters, **kwargs):
        residuals=torch.cat([representation_128,parameters],dim=-1)
        residuals=self.residuals(residuals)
        key = self.key(representation_128)
        query = self.query(parameters)
        value = self.value(representation_128)
        att = query * key
        att = F.softmax(att, dim=-1)
        att = att * value
        att=self.attention(att)
        features = torch.cat([att,residuals],dim=-1)
        opening = self.get_opening(features)
        return opening

class opening_generator(nn.Module):
    def __init__(self):
        super().__init__()

        r = 1.
        radii_list_new = [[0.0025 * r, 0.004 * r, 0.008 * r],
                          [0.015 * r, 0.03 * r, 0.06 * r],
                          [0.08 * r, 0.13 * r],
                          [0.18 * r, 0.25 * r]]
        n = 2
        npoint = [6144 * n, 1536 * n, 384 * n, 96 * n]

        self.back_bone = PointNetbackbone(points_feature_dim=0, radii_list=radii_list_new,npoint=npoint,use_bn=False,use_instance_norm=True).to('cuda')
        self.get_opening=G_decoder().to('cuda')


    def forward(self, pc_data,grasp_pose,**kwargs):

        shifted_pc_data = pc_data - ref_pc_center
        representation_128 = self.back_bone(shifted_pc_data)#.detach()
        representation_128 = reshape_for_layer_norm(representation_128)

        reshaped_grasp_pose=reshape_for_layer_norm(grasp_pose)

        opening=self.get_opening(representation_128,reshaped_grasp_pose)
        opening=reshape_for_layer_norm(opening,reverse=True)

        return opening

class C_decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32),
        ).to('cuda')
        self.query = nn.Sequential(
            nn.Linear(6, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 32),
        ).to('cuda')
        self.value = nn.Sequential(
            nn.Linear(1, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 32),
        ).to('cuda')

        self.get_value= nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 1)
        ).to('cuda')

        self.residuals = nn.Sequential(
            nn.Linear(128 + 7 , 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 16),
        ).to('cuda')

        self.attention = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 16),
        ).to('cuda')

    def forward(self, representation_128,grasp_pose,opening, **kwargs):
        residuals=torch.cat([representation_128,grasp_pose,opening],dim=-1)
        residuals=self.residuals(residuals)
        key = self.key(representation_128)
        query = self.query(grasp_pose)
        value = self.value(opening)
        att = query * key
        att = F.softmax(att, dim=-1)
        att = att * value
        att=self.attention(att)
        features = torch.cat([att,residuals],dim=-1)
        score = self.get_value(features)
        return score

class opening_critic(nn.Module):
    def __init__(self):
        super().__init__()
        r = 1.
        radii_list_new = [[0.0025 * r, 0.004 * r, 0.008 * r],
                          [0.015 * r, 0.03 * r, 0.06 * r],
                          [0.08 * r, 0.13 * r],
                          [0.18 * r, 0.25 * r]]
        n = 2
        npoint = [6144 * n, 1536 * n, 384 * n, 96 * n]
        self.back_bone = PointNetbackbone(points_feature_dim=0, radii_list=radii_list_new, npoint=npoint, use_bn=False,
                                          use_instance_norm=True).to('cuda')
        self.decoder=C_decoder().to('cuda')
    def forward(self, pc_data_,grasp_pose,opening ,**kwargs):
        shifted_pc_data = pc_data_ - ref_pc_center
        representation = self.back_bone(shifted_pc_data[:, :, 0:3])
        representation_128 = reshape_for_layer_norm(representation)
        reshaped_grasp_pose = reshape_for_layer_norm(grasp_pose)
        reshaped_opening = reshape_for_layer_norm(opening)

        score=self.decoder(representation_128,reshaped_grasp_pose,reshaped_opening)
        score=reshape_for_layer_norm(score,reverse=True)
        return score

# opening_generator_net_=initialize_model(opening_generator,opening_generator_path)
# opening_critic_net_=initialize_model(opening_critic,opening_critic_path)