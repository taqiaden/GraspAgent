from torch import nn
from lib.depth_map import depth_to_mesh_grid
from lib.models_utils import reshape_for_layer_norm
from registration import camera

def depth_xy_spatial_data( batch_size):
    xymap = depth_to_mesh_grid(camera)
    spatial_data = xymap.repeat(batch_size, 1, 1, 1)
    return spatial_data

class SpatialEncoder(nn.Module):
    def __init__(self,out_dimension=8):
        super().__init__()
        self.out_dimension=out_dimension
        if out_dimension is not None and out_dimension>2:
            self.spatial_encoding= nn.Linear(2, out_dimension).to('cuda')

    def forward(self, batch_size ):
        spatial_data=depth_xy_spatial_data( batch_size)
        spatial_data_2d=reshape_for_layer_norm(spatial_data, camera=camera, reverse=False)
        if self.out_dimension:
            spatial_data_2d=self.spatial_encoding(spatial_data_2d)
        return spatial_data_2d

if __name__ == '__main__':
    x=depth_xy_spatial_data(1)
    print(x[0,1,0,:])
    print(x[0,1,:,0])
    x = reshape_for_layer_norm(x, camera=camera, reverse=False)
    x = reshape_for_layer_norm(x, camera=camera, reverse=True)
    print('-------------------------------------')
    print(x[0,1,0,:])
    print(x[0,1,:,0])
    print(x.shape)
