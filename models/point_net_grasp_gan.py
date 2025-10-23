import torch
from torch import nn
import torch.nn.functional as F

from models.pointnet2_backbone import PointNetbackbone

ref_pc_center=torch.tensor([0.4,-0.26,1.3]).to('cuda')
num_points= 50000*1 #16384


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
def custom_reshape(tensor,reverse=False,n_points=num_points):
    if reverse==False:
        channels=tensor.shape[1]
        tensor=tensor.transpose(1,2).reshape(-1,channels)
        return tensor
    else:
        batch_size=int(tensor.shape[0]/n_points)
        channels=tensor.shape[-1]
        tensor=tensor.reshape(batch_size,-1,channels).transpose(1,2)
        return tensor
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
class G_point_net(nn.Module):
    def __init__(self,downsample_size=None):
        super().__init__()

        r = 1.
        radii_list_new = [[0.0025 * r, 0.004 * r, 0.008 * r],
                          [0.015 * r, 0.03 * r, 0.06 * r],
                          [0.08 * r, 0.13 * r],
                          [0.18 * r, 0.25 * r]]
        n = 2
        npoint = [6144 * n, 1536 * n, 384 * n, 96 * n]
        self.downsample_size=downsample_size

        self.back_bone = PointNetbackbone(points_feature_dim=0, radii_list=radii_list_new,npoint=npoint,use_bn=False,use_instance_norm=True).to('cuda')
        self.get_approach=approach_module().to('cuda')
        self.get_beta=beta_module().to('cuda')
        self.get_dist=dist_module().to('cuda')
        self.get_width=width_module().to('cuda')

    def custom_activation(self,output):
        # print(output.shape)
        # exit()
        approach = output[:, 0:3, :]
        approach=F.normalize(approach,dim=1)

        beta = output[:, 3:5, :]

        beta=F.normalize(beta,dim=1)

        dist = output[:, 5:6, :]
        # dist=F.sigmoid(dist)
        # dist=torch.clamp(dist,0,1.)
        width = output[:, 6:7, :]
        # width=F.sigmoid(width)
        # width=torch.clamp(width,0,1.)

        output= torch.cat([approach,beta,dist,width],dim=1)
        return output

    def forward(self, depth):
        b = depth.shape[0]

        pc_data=[]
        for i in range(b):
            pc, mask = depth_to_point_clouds_torch(depth[i, 0], camera)
            pc = transform_to_camera_frame_torch(pc, reverse=True)
            print(pc.shape)
            if self.downsample_size is not None:
                assert pc.shape[0]>self.downsample_size,f'{pc.shape[0]}'
                indices = torch.randperm(pc.shape[0])[:self.downsample_size]
                pc=pc[indices]
            pc_data.append(pc)

        pc_data=torch.stack(pc_data)


        shifted_pc_data = pc_data - ref_pc_center
        representation_128 = self.back_bone(shifted_pc_data)#.detach()
        representation_128 = custom_reshape(representation_128)

        reshaped_pc=custom_reshape(shifted_pc_data.permute(0,2,1))
        approach=self.get_approach(representation_128,reshaped_pc)
        params2=torch.cat([approach,reshaped_pc],dim=-1)#.detach().clone()
        beta=self.get_beta(representation_128,params2)
        params3=torch.cat([approach,beta,reshaped_pc],dim=-1)#.detach().clone()
        dist=self.get_dist(representation_128,params3)
        params4=torch.cat([approach,beta,dist,reshaped_pc],dim=-1)#.detach().clone()
        width=self.get_width(representation_128,params4)
        final_pose = torch.cat([approach, beta, dist, width], dim=-1)

        final_pose=final_pose.reshape(b, -1, 7).transpose(1, 2)
        final_pose = self.custom_activation(final_pose)

        return final_pose

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
        reshaped_representation = custom_reshape(representation)
        reshaped_pose = custom_reshape(dense_pose)

        encoded_pose=self.pose_encoder(reshaped_pose)
        encoded_rep=self.point_encoder(reshaped_representation)
        features=torch.cat([encoded_rep,encoded_pose],dim=-1)

        output = self.decoderB(features)  # [b,1,n_points]
        output = custom_reshape(output, reverse=True)
        return output

class D_point_net(nn.Module):
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

        self.dis3=discriminator3().to('cuda')

    def forward(self, pc_data_,dense_pose_ ,**kwargs):
        shifted_pc_data = pc_data_ - self.pc_center_3
        # with torch.no_grad():
        representation = self.back_bone(shifted_pc_data[:, :, 0:3])

        # representation=self.dropout(representation)
        contrastive_output= self.dis3(representation,dense_pose_)

        return contrastive_output

class DepthPointNetAdapter(nn.Module):
    def __init__(self,downsample_size=50000,use_bn=False,use_instance_norm=True):
        super().__init__()

        r = 1.
        radii_list_new = [[0.0025 * r, 0.004 * r, 0.008 * r],
                          [0.015 * r, 0.03 * r, 0.06 * r],
                          [0.08 * r, 0.13 * r],
                          [0.18 * r, 0.25 * r]]
        n = 2
        npoint = [6144 * n, 1536 * n, 384 * n, 96 * n]
        self.downsample_size=downsample_size

        self.back_bone = PointNetbackbone(points_feature_dim=0, radii_list=radii_list_new,npoint=npoint,use_bn=use_bn,use_instance_norm=use_instance_norm).to('cuda')

        from registration import camera
        self.camera=camera

    def custom_activation(self,output):
        # print(output.shape)
        # exit()
        approach = output[:, 0:3, :]
        approach=F.normalize(approach,dim=1)

        beta = output[:, 3:5, :]

        beta=F.normalize(beta,dim=1)

        dist = output[:, 5:6, :]
        # dist=F.sigmoid(dist)
        # dist=torch.clamp(dist,0,1.)
        width = output[:, 6:7, :]
        # width=F.sigmoid(width)
        # width=torch.clamp(width,0,1.)

        output= torch.cat([approach,beta,dist,width],dim=1)
        return output

    def forward(self, depth):
        # print(depth.shape)
        b = depth.shape[0]
        h=depth.shape[2]
        w=depth.shape[3]

        pc_data=[]
        original_point_size=[]
        selected_indicses=[]
        masks_list=[]
        from lib.depth_map import depth_to_point_clouds_torch, transform_to_camera_frame_torch

        for i in range(b):
            pc, mask = depth_to_point_clouds_torch(depth[i, 0], self.camera)
            masks_list.append(mask)
            pc = transform_to_camera_frame_torch(pc, reverse=True)

            pc = pc - ref_pc_center

            original_point_size.append(pc.shape[0])

            if self.downsample_size is not None:
                assert pc.shape[0]>self.downsample_size,f'{pc.shape[0]}'
                indices = torch.randperm(pc.shape[0])[:self.downsample_size]
                pc=pc[indices]

                selected_indicses.append(indices)
            pc_data.append(pc)

        pc_data=torch.stack(pc_data)

        features = self.back_bone(pc_data)

        # return features

        channels_size=features.shape[1]
        depth_shape_features=[]
        for i in range(b):
            instance_features=features[i].t()

            reconstructed_pc_features=torch.zeros((original_point_size[i],channels_size),device=depth.device)
            reconstructed_pc_features[selected_indicses[i]]=instance_features

            reconstructed_depth_features=torch.zeros((h,w,channels_size),device=depth.device)
            reconstructed_depth_features[masks_list[i]]=reconstructed_pc_features
            reconstructed_depth_features=reconstructed_depth_features.permute(2,0,1)
            depth_shape_features.append(reconstructed_depth_features)

        depth_shape_features=torch.stack(depth_shape_features)

        return depth_shape_features

if __name__ == "__main__":
    # pc2, mask2 = depth_to_point_clouds_torch(depth[0, 0], camera)
    # pc2 = transform_to_camera_frame_torch(pc2, reverse=True)
    points=torch.rand((1,1,480,712)).to('cuda')
    model=DepthPointNetAdapter(downsample_size=50000)
    y=model(points)
    print(y.shape)