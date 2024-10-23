import torch
from torch import nn
import torch.nn.functional as F

from lib.custom_activations import GripperGraspRegressor
from lib.depth_map import depth_to_mesh_grid
from models.decoders import att_res_mlp_LN
from models.resunet import res_unet
from registration import camera, standardize_depth

gripper_sampler_path=r'gripper_sampler_model_state'
gripper_critic_path=r'gripper_critic_model_state'

use_bn=False
use_in=True

def reshape_for_layer_norm(tensor,camera=camera,reverse=False):
    if reverse==False:
        channels=tensor.shape[1]
        tensor=tensor.permute(0,2,3,1).reshape(-1,channels)
        return tensor
    else:
        batch_size=int(tensor.shape[0]/(camera.width*camera.height))
        channels=tensor.shape[-1]
        tensor=tensor.reshape(batch_size,camera.height,camera.width,channels).permute(0,3,1,2)
        return tensor

class gripper_sampler_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')
        self.get_approach=att_res_mlp_LN(in_c1=64, in_c2=2, out_c=3).to('cuda')
        self.get_beta_dist=att_res_mlp_LN(in_c1=64, in_c2=5, out_c=3).to('cuda')
        self.get_width=att_res_mlp_LN(in_c1=64, in_c2=8, out_c=1).to('cuda')

        self.gripper_regressor_layer=GripperGraspRegressor()

    def forward(self, depth):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        representation = self.back_bone(depth)
        representation_2d=reshape_for_layer_norm(representation, camera=camera, reverse=False)

        '''spatial data'''
        b=depth.shape[0]
        xymap=depth_to_mesh_grid(camera)

        '''decode'''
        params1=xymap.repeat(b,1,1,1)
        params1=reshape_for_layer_norm(params1, camera=camera, reverse=False)

        approach=self.get_approach(representation_2d,params1)

        params2=torch.cat([approach,params1],dim=1)
        beta_dist=self.get_beta_dist(representation_2d,params2)

        params3=torch.cat([beta_dist,params2],dim=1)

        width=self.get_width(representation_2d,params3)

        output_2d = torch.cat([approach, beta_dist, width], dim=1)

        output=reshape_for_layer_norm(output_2d, camera=camera, reverse=True)

        output=self.gripper_regressor_layer(output)
        return output

class critic_net(nn.Module):
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

