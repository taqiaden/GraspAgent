import torch.nn as nn
from Configurations import config
from models.utils.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG

def activate_parameters_training(module_list,activate):
    for p in module_list.parameters():
        p.requires_grad = activate
class PointNetbackbone(nn.Module):
    def __init__(self,points_feature_dim=0,radii_list=None,npoint=None,use_bn=True,use_instance_norm=False):
        super().__init__()

        self.use_xyz = config.use_xyz
        self.points_feature_dim = points_feature_dim
        self.use_bn=use_bn
        self.use_instance_norm=use_instance_norm
        self.radii_list = [[0.005, 0.01, 0.015],
                           [0.02, 0.03, 0.04],
                           [0.05, 0.1],
                           [0.1, 0.15]] if radii_list is None else radii_list
        self.npoint=[6144,1536,384,96] if npoint is None else npoint
        self._build_model()


    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.npoint[0],  # 4096
                radii=self.radii_list[0],
                nsamples=[8, 16, 32],

                mlps=[[self.points_feature_dim, 16, 32], [self.points_feature_dim, 16, 32],
                      [self.points_feature_dim, 32, 64]],
                use_xyz=self.use_xyz,
                bn=self.use_bn,
        instance_norm=self.use_instance_norm

            )
        )
        c_out_0 = 32 + 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.npoint[1],  # 1024
                radii=self.radii_list[1],
                nsamples=[8, 16, 32],

                mlps=[[c_in, 64, 96], [c_in, 64, 128], [c_in, 96, 128]],
                use_xyz=self.use_xyz,
                bn=self.use_bn,
                instance_norm=self.use_instance_norm
            )
        )
        c_out_1 = 96 + 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.npoint[2],  # 256
                radii=self.radii_list[2],
                nsamples=[16, 32],

                mlps=[[c_in, 196, 256], [c_in, 196, 256]],
                use_xyz=self.use_xyz,
                bn=self.use_bn,
                instance_norm=self.use_instance_norm
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.npoint[3],  # 64
                radii=self.radii_list[3],
                nsamples=[16, 32],

                mlps=[[c_in, 256, 512], [c_in, 384, 512]],
                use_xyz=self.use_xyz,
                bn=self.use_bn,
                instance_norm=self.use_instance_norm
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + self.points_feature_dim, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, **kwargs):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)


        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]

if __name__ == "__main__":
    pass
