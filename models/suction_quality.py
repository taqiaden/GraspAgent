import torch
from torch import nn
from lib.depth_map import depth_to_mesh_grid
from models.decoders import res_block_mlp_LN
from models.resunet import res_unet
from registration import camera, standardize_depth

suction_quality_model_state_path=r'suction_quality_model_state'

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

def get_coordinates_from_pixels(batch_size):
    xymap = depth_to_mesh_grid(camera)
    xymap = xymap.repeat(batch_size, 1, 1, 1)
    return xymap


class suction_quality_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')
        self.pose_transform = nn.Linear(3, 16).to('cuda')

        self.res_block= res_block_mlp_LN(in_c=64+16,medium_c=32,out_c=16,activation=nn.SiLU(True)).to('cuda')

        self.decoder= nn.Sequential(
            nn.LayerNorm(16),
            nn.SiLU(True),
            nn.Linear(16, 1),
        ).to('cuda')


    def forward(self, depth,pose ):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''depth backbone'''
        depth_features = self.back_bone(depth)

        '''flatten'''
        depth_features_2d = reshape_for_layer_norm(depth_features, camera=camera, reverse=False)
        pose_2d = reshape_for_layer_norm(pose, camera=camera, reverse=False)

        '''pose encoder'''
        pose_features_2d = self.pose_transform(pose_2d)

        '''concatenate'''
        features_2d = torch.cat([pose_features_2d, depth_features_2d], dim=1)

        '''decode'''
        output_2d = self.decoder(self.res_block(features_2d))

        '''unflatten'''
        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)

        return output
