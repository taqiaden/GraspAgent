import numpy as np
import torch

def transform_to_camera_frame(pc, reverse=False):
    a=-0.4*np.pi/180
    angle_correction1=np.array([[np.cos(a), -np.sin(a), 0.0, 0.0],
                       [np.sin(a), np.cos(a), 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    b=-0.6*np.pi/180
    angle_correction2=np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, np.cos(b), -np.sin(b), 0.0],
                       [0.0, np.sin(b), np.cos(b), 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    matrix = np.array([[0.0, -1.0, 0.0, 0.393],
                       [-1.0, 0.0, 0.0, -0.280],
                       [0.0, 0.0, -1.0, 1.337],
                       [0.0, 0.0, 0.0, 1.0]])
    matrix=np.matmul(angle_correction2,matrix)

    matrix=np.matmul(matrix,angle_correction1)

    if reverse==True:
        transformation=matrix
    else:
        transformation = np.linalg.inv(matrix)


    column = np.ones(len(pc))
    stacked = np.column_stack((pc, column))
    transformed_pc = np.dot(transformation, stacked.T).T[:, :3]
    transformed_pc = np.ascontiguousarray(transformed_pc)
    return transformed_pc

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

xymap=None
def depth_to_mesh_grid(camera,normalize=True):
    global xymap
    if xymap is None:
        xmap = torch.arange(camera.width)
        ymap = torch.arange(camera.height)
        xmap, ymap = torch.meshgrid(xmap, ymap)

        if normalize:
            xmap=xmap/camera.width
            ymap=ymap/camera.height
        xymap=torch.stack([xmap,ymap]).to('cuda').transpose(1,2)

    return xymap

def depth_to_ordered_cloud(depth,camera):
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    return cloud


def get_pixel_index( camera,target_point):
    # cloud = depth_to_ordered_cloud(depth, camera)
    transformed_target_point = transform_to_camera_frame(target_point[None,:])

    # distance=cloud-transformed_target_point[None,:]
    # distance = np.linalg.norm(distance, axis=-1,keepdims=True)
    # pixel_index=np.unravel_index(distance.argmin(),distance.shape)[0:2]

    # '''verify closest pixel to the point'''
    # assert cloud[pixel_index][2]>0.0 , f'{cloud[pixel_index][2]}'
    # assert distance[pixel_index] < radius , f'{distance[pixel_index]}'

    pixel_index=target_to_pixel(transformed_target_point[0],camera)

    pixel_index = np.array(pixel_index)
    return pixel_index

def pixel_to_point(pixel_index,depth_value,camera):
    z=depth_value/camera.scale
    x = (pixel_index[1] - camera.cx) * z / camera.fx
    y = (pixel_index[0] - camera.cy) * z / camera.fy
    point=np.stack([x,y,z])
    return point

def depth_to_point_clouds(depth, camera,rgb=None):
    '''check camera intrinsic'''
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width), 'depth shape error! depth.shape = {}'.format(depth.shape)

    '''process point clouds'''
    cloud = depth_to_ordered_cloud(depth,camera)

    '''assign colors'''
    if rgb is  not None:
        cloud=np.concatenate([cloud,rgb],axis=-1)

    '''remove points with zero depth'''
    mask = cloud[:,:, 2] != 0
    cloud = cloud[mask]

    return cloud,mask

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

def target_to_pixel(target_point,camera):
    depth = target_point[ 2]
    y = target_point[ 0] * camera.fx / depth + camera.cx
    x = target_point[ 1] * camera.fy / depth + camera.cy
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    x = np.clip(x, 0, camera.height - 1)
    y = np.clip(y, 0, camera.width - 1)
    return (x,y)



if __name__ == '__main__':
    pass