from torch import nn
from lib.models_utils import reshape_for_layer_norm
from models.decoders import  att_res_mlp_LN
from models.resunet import res_unet
from models.spatial_encoder import SpatialEncoder
from registration import camera, standardize_depth

use_bn=False
use_in=True

class ShiftNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')

        self.att_block= att_res_mlp_LN(in_c1=64, in_c2=8, out_c=16).to('cuda')
        self.spatial_encoding= SpatialEncoder()
        self.decoder= nn.Sequential(
            nn.LayerNorm(16),
            nn.ReLU(True),
            nn.Linear(16, 1),
        ).to('cuda')

    def forward(self, depth ):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''spatial data'''
        spatial_data_2d=self.spatial_encoding(depth.shape[0])

        '''depth backbone'''
        depth_features = self.back_bone(depth)

        '''flatten'''
        depth_features_2d = reshape_for_layer_norm(depth_features, camera=camera, reverse=False)

        '''decode'''
        output_2d = self.decoder(self.att_block(depth_features_2d,spatial_data_2d))

        '''unflatten'''
        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)

        return output
