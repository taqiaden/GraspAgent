import numpy as np

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def depth_to_point_clouds(RGB_D, camera):
    depth=RGB_D[:,:,0]
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width), 'depth shape error! depth.shape = {}'.format(depth.shape)

    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    if RGB_D.shape[-1]==4:
        colors=RGB_D[:,:,1:]
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud=np.concatenate([cloud,colors],axis=-1)
        cloud = cloud.reshape([-1, 6])
    else:
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud = cloud.reshape([-1, 3])
    mask = cloud[:, 2] != 0
    cloud = cloud[mask]
    return cloud

def point_clouds_to_depth(pc, camera):
    depth_map = np.zeros([camera.height, camera.width])
    mask = pc[:, 2] != 0
    pc = pc[mask]
    depth = pc[:, 2]
    y = pc[:, 0] * camera.fx / depth + camera.cx
    x = pc[:, 1] * camera.fy / depth + camera.cy
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    x = np.clip(x, 0, camera.height - 1)
    y = np.clip(y, 0, camera.width - 1)
    depth = depth * camera.scale

    for i, j, d in zip(x, y, depth):
        depth_map[i, j] = d
    return depth_map


if __name__ == '__main__':
    pass