from torch import nn

from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import depth_xy_spatial_data
from registration import camera, standardize_depth

use_bn=False
use_in=True

class AbstractHead(nn.Module):
    def __init__(self,in_c1, in_c2, out_c):
        super().__init__()
        self.att_block = att_res_mlp_LN(in_c1=in_c1, in_c2=in_c2, out_c=8).to('cuda')
        # self.res_block=res_block_mlp_LN(in_c=in_c1+in_c2,medium_c=32,out_c=1).to('cuda')
        # self.d = nn.Sequential(
        #     nn.Linear(16, 8, bias=False),
        #     nn.LayerNorm([8]),
        #     nn.ReLU(),
        #     nn.Linear(8, 1),
        # ).to('cuda')
        self.sigmoid=nn.Sigmoid()

    def forward(self, features,pose ):
        '''flatten'''
        features_2d = reshape_for_layer_norm(features, camera=camera, reverse=False)
        pose_2d = reshape_for_layer_norm(pose, camera=camera, reverse=False)

        '''decode'''
        output_2d = self.att_block(features_2d,pose_2d)
        # output_2d_a = self.att_block(features_2d,pose_2d)
        # output_2d_b = self.res_block(torch.cat([features_2d,pose_2d],dim=-1))
        # output_2d=self.d(torch.cat([output_2d_a,output_2d_b],dim=-1))

        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)
        output=self.sigmoid(output)
        return output

class JointQualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')
        self.gripper_collision =AbstractHead(in_c1=64, in_c2=7, out_c=1)
        self.suction_quality=AbstractHead(in_c1=64, in_c2=3, out_c=1)
        self.shift_affordance=AbstractHead(in_c1=64, in_c2=2, out_c=1)
        self.spatial_encoding = depth_xy_spatial_data(batch_size=1)

    def forward(self, depth,gripper_pose,suction_pose):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        features = self.back_bone(depth)
        '''spatial data'''
        if self.spatial_encoding.shape[0]!=depth.shape[0]:self.spatial_encoding=depth_xy_spatial_data(batch_size=depth.shape[0])

        '''gripper collision head'''
        griper_collision_classifier = self.gripper_collision(features,gripper_pose)

        '''suction quality head'''
        suction_quality_classifier=self.suction_quality(features,suction_pose)

        '''shift affordance head'''
        shift_affordance_regressor=self.shift_affordance(features,self.spatial_encoding)

        return griper_collision_classifier,suction_quality_classifier,shift_affordance_regressor

