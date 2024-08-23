import torch
from torch import nn
import torch.nn.functional as F
from Configurations.ENV_boundaries import ref_pc_center_6
from lib.models_utils import reshape_for_layer_norm
from models.pointnet2_backbone import PointNetbackbone

affordance_net_model_path=r'affordance_net_model'

class affordance_net(nn.Module):
    def __init__(self):
        super().__init__()
        r = 1.
        radii_list_new = [[0.0025 * r, 0.004 * r, 0.008 * r],
                          [0.015 * r, 0.03 * r, 0.06 * r],
                          [0.08 * r, 0.13 * r],
                          [0.18 * r, 0.25 * r]]
        n = 2
        npoint = [6144 * n, 1536 * n, 384 * n, 96 * n]
        self.back_bone = PointNetbackbone(points_feature_dim=0, radii_list=radii_list_new,npoint=npoint, use_bn=False,
                                          use_instance_norm=True).to('cuda')
        self.back_bone2 = PointNetbackbone(points_feature_dim=0, radii_list=radii_list_new, npoint=npoint, use_bn=False,
                                          use_instance_norm=True).to('cuda')


        self.decoder = nn.Sequential(
            nn.Linear(128+3, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 1),
        ).to('cuda')

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

        self.residual = nn.Sequential(
            nn.Linear(128+3, 64, bias=False),
            nn.LayerNorm([64]),
            nn.ReLU(True),
            nn.Linear(64, 32),
        ).to('cuda')

        self.fc = nn.Sequential(
            nn.Linear(48, 48),
        ).to('cuda')

        self.decoder2 = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 1),

        ).to('cuda')


    def forward(self, pc_data, **kwargs):
        shifted_pc_data = pc_data - ref_pc_center_6
        normals=shifted_pc_data[:, :, 3:6].transpose(1, 2)
        normals=reshape_for_layer_norm(normals)
        n_points=pc_data.shape[1]

        features = self.back_bone(shifted_pc_data[:,:,0:3])
        features=reshape_for_layer_norm(features,n_points=n_points)

        features = torch.cat([features, normals], dim=1)
        output = self.decoder(features)  # [b,1,n_points]
        output=reshape_for_layer_norm(output,reverse=True,n_points=n_points)

        #method2
        # features2 = self.back_bone2(shifted_pc_data[:,:,0:3])
        # features2=reshape_for_layer_norm(features2,n_points=n_points)

        # residual=torch.cat([features2,normals],dim=-1)
        # residual=self.residual(residual)

        # key=self.key(features2)
        # query=self.query(normals)
        # value=self.value(features2)

        # att_map=key * query
        # att_map=F.softmax(att_map,dim=-1)
        # att_score=att_map*value
        # cat_features=torch.cat([att_score,residual],dim=-1)


        # output2 = self.decoder2(cat_features)  # [b,1,n_points]
        # output2=reshape_for_layer_norm(output2,reverse=True,n_points=n_points)

        return output