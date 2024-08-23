import torch
from torch import nn
import torch.nn.functional as F
from Configurations.ENV_boundaries import ref_pc_center
from lib.models_utils import reshape_for_layer_norm
from models.point_net_base.pointnet2_backbone import PointNetbackbone

dropout_p=0.0
dense_gripper_generator_path=r'dense_gripper_generator'
contrastive_discriminator_path=r'contrastive_discriminator'


class approach_module(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32),
        ).to('cuda')
        self.query = nn.Sequential(
            nn.Linear(3, 16, bias=False),
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

        self.get_approach= nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 3)
        ).to('cuda')

        self.residuals = nn.Sequential(
            nn.Linear(128 + 3 , 64, bias=False),
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
        approach = self.get_approach(features)
        return approach

class beta_module(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32),
        ).to('cuda')

        self.value = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Linear(3+3, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 32),
        ).to('cuda')

        self.get_beta= nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 2)
        ).to('cuda')

        self.residuals = nn.Sequential(
            nn.Linear(128 + 3 + 3, 64, bias=False),
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

    def forward(self, representation_128, parameters, **kwargs):
        residuals = torch.cat([representation_128, parameters], dim=-1)
        residuals = self.residuals(residuals)
        key = self.key(representation_128)
        value = self.value(representation_128)
        query = self.query(parameters)
        att = query * key
        att = F.softmax(att, dim=-1)
        att = att * value
        att = self.attention(att)
        # print(key[0])
        # print(query[0])
        # print(value[0])
        # print(att[0])
        # print(residuals[0])
        # exit()
        features = torch.cat([att, residuals], dim=-1)
        beta = self.get_beta(features)
        return beta

class dist_module(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Linear(5+3, 16, bias=False),
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

        self.get_dist= nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 1)
        ).to('cuda')

        self.residuals = nn.Sequential(
            nn.Linear(128+3+5, 64, bias=False),
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

    def forward(self, representation_128, parameters, **kwargs):
        residuals = torch.cat([representation_128, parameters], dim=-1)
        residuals = self.residuals(residuals)
        key = self.key(representation_128)
        value = self.value(representation_128)
        query=self.query(parameters)
        att = query * key
        att= F.softmax(att, dim=-1)
        att = att * value
        att=self.attention(att)
        features = torch.cat([att,residuals],dim=-1)
        dist = self.get_dist(features)
        return dist
class width_module(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Linear(6+3, 16, bias=False),
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

        self.get_width= nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 1)
        ).to('cuda')

        self.residuals = nn.Sequential(
            nn.Linear(128+3+6, 64, bias=False),
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

    def forward(self, representation_128, parameters, **kwargs):
        residuals = torch.cat([representation_128, parameters], dim=-1)
        residuals = self.residuals(residuals)
        key = self.key(representation_128)
        value = self.value(representation_128)
        query=self.query(parameters)
        att = query * key
        att= F.softmax(att, dim=-1)
        att = att * value
        att=self.attention(att)
        features = torch.cat([att,residuals],dim=-1)
        width = self.get_width(features)
        return width
class gripper_generator(nn.Module):
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
        self.get_approach=approach_module().to('cuda')
        self.get_beta=beta_module().to('cuda')
        self.get_dist=dist_module().to('cuda')
        self.get_width=width_module().to('cuda')

    def custom_activation(self,output,output_theta_phi=True):
        # print(output.shape)
        # exit()
        approach = output[:, 0:3, :]
        approach=F.normalize(approach,dim=1)
        #if output_theta_phi:
        #    theta_ratio,phi_sin, phi_cos=self.approach_vec_to_theta_phi(approach)

            # print(theta.shape)
            # print(phi.shape)
            # print(approach.sa we am am am

        # print(approach.shape)
        beta_sin_cos = output[:, 3:5, :]

        beta_sin_cos=F.normalize(beta_sin_cos,dim=1)

        dist = output[:, 5:6, :]
        # dist=F.sigmoid(dist)
        # dist=torch.clamp(dist,0,1.)
        width = output[:, 6:7, :]
        # width=F.sigmoid(width)
        # width=torch.clamp(width,0,1.)

        output= torch.cat([approach,beta_sin_cos,dist,width],dim=1)
        return output

    def forward(self, pc_data=None,pose=None,**kwargs):

        if pc_data is not None:
            b = pc_data.shape[0]

            shifted_pc_data = pc_data - ref_pc_center
            representation_128 = self.back_bone(shifted_pc_data)#.detach()
            representation_128 = reshape_for_layer_norm(representation_128)

            reshaped_pc=reshape_for_layer_norm(shifted_pc_data.permute(0,2,1))
            approach=self.get_approach(representation_128,reshaped_pc)
            params2=torch.cat([approach,reshaped_pc],dim=-1)#.detach().clone()
            beta=self.get_beta(representation_128,params2)
            params3=torch.cat([approach,beta,reshaped_pc],dim=-1)#.detach().clone()
            dist=self.get_dist(representation_128,params3)
            params4=torch.cat([approach,beta,dist,reshaped_pc],dim=-1)#.detach().clone()
            width=self.get_width(representation_128,params4)
            final_pose = torch.cat([approach, beta, dist, width], dim=-1)

            final_pose=final_pose.reshape(b, -1, 7).transpose(1, 2)
            final_pose = self.custom_activation(final_pose, output_theta_phi=False)

            return final_pose

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
class contrastive_discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pc_center_3 = torch.tensor([0.4364, -0.0091, 0.0767]).to('cuda')
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



    def forward(self, pc_data_,dense_pose_ ,**kwargs):
        shifted_pc_data = pc_data_ - self.pc_center_3
        representation = self.back_bone(shifted_pc_data[:, :, 0:3])

        contrastive_output= self.dis(representation,dense_pose_) * self.dis3(representation,dense_pose_)
        contrastive_output=contrastive_output+self.dis3_plus(representation,dense_pose_)

        return contrastive_output

