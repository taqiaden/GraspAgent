import torch
from torch import nn
import torch.nn.functional as F
from Configurations.config import ref_pc_center
from lib.models_utils import reshape_for_layer_norm, initialize_model
from models.pointnet2_backbone import PointNetbackbone
from Configurations.config import ip_address

dropout_p=0.0
dense_gripper_discriminator_path=r'dense_gripper_discriminator.pth.tar'

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.pose_encoder = nn.Sequential(
            nn.Linear(7, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(),
            nn.Linear(64, 128, bias=False),
            nn.LayerNorm([128]),
            nn.ReLU(),
            # nn.Dropout(dropout_p),
            nn.Linear(128, 128),
        ).to('cuda')

        self.point_encoder = nn.Sequential(
            nn.Linear(128, 128,bias=False),
            nn.LayerNorm([128]),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 128),
        ).to('cuda')


        self.decoderB = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 1),
        ).to('cuda')

    def forward(self, representation, dense_pose, **kwargs):
        assert dense_pose.shape[1] == 7
        reshaped_representation = reshape_for_layer_norm(representation)

        encoded_point = self.point_encoder(reshaped_representation)

        reshaped_pose = reshape_for_layer_norm(dense_pose)
        encoded_pose = self.pose_encoder(reshaped_pose)
        encoded_pose = F.softmax(encoded_pose, dim=-1)

        attention_map = encoded_point * encoded_pose

        output = self.decoderB(attention_map)  # [b,1,n_points]
        output = reshape_for_layer_norm(output, reverse=True)
        return output

class discriminator3(nn.Module):
    def __init__(self):
        super().__init__()

        self.pose_encoder = nn.Sequential(
            nn.Linear(7, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(),
            nn.Linear(32, 64),
        ).to('cuda')

        self.point_encoder = nn.Sequential(
            nn.Linear(128, 64),
        ).to('cuda')


        self.decoderB = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 1),
        ).to('cuda')

    def forward(self, representation, dense_pose, **kwargs):
        assert dense_pose.shape[1] == 7
        reshaped_representation = reshape_for_layer_norm(representation)
        reshaped_pose = reshape_for_layer_norm(dense_pose)

        encoded_pose=self.pose_encoder(reshaped_pose)
        encoded_rep=self.point_encoder(reshaped_representation)
        features=torch.cat([encoded_rep,encoded_pose],dim=-1)

        output = self.decoderB(features)  # [b,1,n_points]
        output = reshape_for_layer_norm(output, reverse=True)
        return output

class grasp_ability(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoderB = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 1),
        ).to('cuda')

    def forward(self, representation, **kwargs):
        reshaped_representation = reshape_for_layer_norm(representation)
        output = self.decoderB(reshaped_representation)  # [b,1,n_points]
        output = reshape_for_layer_norm(output, reverse=True)
        return output
class gripper_discriminator(nn.Module):
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

        self.dis=discriminator().to('cuda')
        self.dis3=discriminator3().to('cuda')
        self.dis3_plus=discriminator3().to('cuda')
        self.grasp_ability_=grasp_ability().to('cuda')

    def forward(self, pc_data_,dense_pose_ ,**kwargs):
        shifted_pc_data = pc_data_[:,:,0:3] - ref_pc_center

        representation=self.back_bone(shifted_pc_data)
        quality_score= self.dis(representation,dense_pose_) * self.dis3(representation,dense_pose_)
        quality_score=quality_score+self.dis3_plus(representation,dense_pose_)

        grasp_ability_score=self.grasp_ability_(representation.detach().clone())


        return quality_score,grasp_ability_score

# dense_gripper_discriminator_net_=initialize_model(gripper_discriminator,dense_gripper_discriminator_path)
