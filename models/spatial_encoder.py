from torch import nn
from lib.depth_map import depth_to_mesh_grid
from lib.models_utils import reshape_for_layer_norm
from registration import camera


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial_encoding= nn.Linear(2, 8).to('cuda')


    def forward(self, batch_size ):
        '''input standardization'''

        '''spatial data'''
        xymap=depth_to_mesh_grid(camera)
        spatial_data=xymap.repeat(batch_size, 1, 1, 1)
        spatial_data_2d=reshape_for_layer_norm(spatial_data, camera=camera, reverse=False)
        spatial_data_2d=self.spatial_encoding(spatial_data_2d)

        return spatial_data_2d