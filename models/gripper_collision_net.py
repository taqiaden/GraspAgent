from torch import nn

from lib.models_utils import reshape_for_layer_norm
from models.decoders import att_res_mlp_LN
from models.resunet import res_unet
from registration import camera, standardize_depth

gripper_collision_net_path=r'gripper_collision_model_state'

use_bn=False
use_in=True


class GripperCollisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')
        self.att_block = att_res_mlp_LN(in_c1=64,in_c2=7, out_c=1).to('cuda')

    def forward(self, depth,pose):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        features = self.back_bone(depth)
        features_2d=reshape_for_layer_norm(features, camera=camera, reverse=False)
        pose_2d=reshape_for_layer_norm(pose, camera=camera, reverse=False)

        '''decode'''
        output_2d = self.att_block(features_2d,pose_2d)
        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)
        return output

