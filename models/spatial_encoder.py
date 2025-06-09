from torch import nn

from lib.depth_map import depth_to_mesh_grid
from lib.models_utils import reshape_for_layer_norm
from registration import camera


def depth_xy_spatial_data( batch_size):
    xymap = depth_to_mesh_grid(camera)
    spatial_data = xymap.repeat(batch_size, 1, 1, 1)
    return spatial_data

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
