import torch
from torch import nn
from lib.models_utils import reshape_for_layer_norm
from models.decoders import res_block_mlp_LN, att_res_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import SpatialEncoder, depth_xy_spatial_data
from registration import camera, standardize_depth

use_bn=False
use_in=True

class ShiftNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')

        self.att_block= att_res_mlp_LN(in_c1=64, in_c2=2, out_c=1).to('cuda')
        self.spatial_encoding= depth_xy_spatial_data(1)


    def forward(self, depth ):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''spatial data'''
        if self.spatial_encoding.shape[0] != depth.shape[0]: self.spatial_encoding = depth_xy_spatial_data(
            batch_size=depth.shape[0])
        spatial_data_2d = reshape_for_layer_norm(self.spatial_encoding, camera=camera, reverse=False)

        '''depth backbone'''
        depth_features = self.back_bone(depth)

        '''flatten'''
        depth_features_2d = reshape_for_layer_norm(depth_features, camera=camera, reverse=False)

        '''decode'''
        output_2d = self.att_block(depth_features_2d,spatial_data_2d)

        '''unflatten'''
        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)
        return output
