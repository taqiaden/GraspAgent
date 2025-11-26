import math

import numpy as np
import torch
torch_version = tuple(int(v) for v in torch.__version__.split('.')[:2])

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


def transform_to_camera_frame_torch(pc: torch.Tensor, reverse: bool = False, device=None):
    """
    pc: [N, 3] torch tensor (point cloud)
    reverse: whether to apply transformation or its inverse
    device: optional torch device (cpu or cuda)
    """
    if device is None:
        device = pc.device
    dtype = pc.dtype

    # Convert angles to radians
    a = -0.4 * math.pi / 180
    b = -0.6 * math.pi / 180

    # Construct transformation matrices (batched)
    angle_correction1 = torch.tensor([
        [math.cos(a), -math.sin(a), 0.0, 0.0],
        [math.sin(a), math.cos(a), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)

    angle_correction2 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, math.cos(b), -math.sin(b), 0.0],
        [0.0, math.sin(b), math.cos(b), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)

    # base transformation matrix
    matrix = torch.tensor([
        [0.0, -1.0,  0.0,  0.393],
        [-1.0, 0.0,  0.0, -0.280],
        [0.0,  0.0, -1.0,  1.337],
        [0.0,  0.0,  0.0,  1.0]
    ], device=device)

    # compose transforms
    matrix = angle_correction2 @ matrix
    matrix = matrix @ angle_correction1

    # invert if needed
    if not reverse:
        matrix = torch.linalg.inv(matrix)

    # add homogeneous coordinate
    ones = torch.ones((pc.shape[0], 1), device=device, dtype=pc.dtype)
    stacked = torch.cat([pc, ones], dim=1)  # [N, 4]

    # apply transform
    transformed_pc = (matrix @ stacked.T).T[:, :3].contiguous()

    return transformed_pc


# def transform_to_camera_frame_torch(pc, reverse=False):
#     """
#     Transform point cloud to/from camera frame (PyTorch version).
#
#     Args:
#         pc (torch.Tensor): Point cloud of shape (N, 3) or (B, N, 3).
#         reverse (bool): If True, applies forward transform (camera->world).
#
#     Returns:
#         torch.Tensor: Transformed point cloud with same shape as input.
#     """
#     device = pc.device
#     dtype = pc.dtype
#
#     # Convert angles to radians
#     a = -0.4 * math.pi / 180
#     b = -0.6 * math.pi / 180
#
#     # Construct transformation matrices (batched)
#     angle_correction1 = torch.tensor([
#         [math.cos(a), -math.sin(a), 0.0, 0.0],
#         [math.sin(a), math.cos(a), 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ], device=device, dtype=dtype)
#
#     angle_correction2 = torch.tensor([
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, math.cos(b), -math.sin(b), 0.0],
#         [0.0, math.sin(b), math.cos(b), 0.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ], device=device, dtype=dtype)
#
#     base_matrix = torch.tensor([
#         [0.0, -1.0, 0.0, 0.393],
#         [-1.0, 0.0, 0.0, -0.280],
#         [0.0, 0.0, -1.0, 1.337],
#         [0.0, 0.0, 0.0, 1.0]
#     ], device=device, dtype=dtype)
#
#     # Compute final transformation matrix
#     matrix = angle_correction2 @ base_matrix @ angle_correction1
#     transformation = torch.linalg.inv(matrix) if not reverse else matrix
#
#     # Handle batched (B, N, 3) vs unbatched (N, 3) input
#     if pc.dim() == 2:
#         pc = pc.unsqueeze(0)  # Add batch dim if needed
#         squeeze_output = True
#     else:
#         squeeze_output = False
#
#     # Convert to homogeneous coordinates
#     ones = torch.ones(pc.shape[:-1] + (1,), device=device, dtype=dtype)
#     homogeneous_pc = torch.cat([pc, ones], dim=-1)  # (B, N, 4)
#
#     # Apply transformation (batched matrix multiplication)
#     transformed_pc = (homogeneous_pc @ transformation.T)[..., :3]  # (B, N, 3)
#
#     if squeeze_output:
#         transformed_pc = transformed_pc.squeeze(0)  # Remove batch dim if input was 2D
#
#     return transformed_pc.contiguous()

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
        if torch_version >= (1, 10):
            xmap, ymap = torch.meshgrid(xmap, ymap, indexing='xy')
        else:
            ymap,xmap = torch.meshgrid(ymap,xmap)

        if normalize:
            xmap=xmap/camera.width
            ymap=ymap/camera.height
        xymap=torch.stack([xmap,ymap]).to('cuda')#.transpose(1,2)

    return xymap

def depth_to_ordered_cloud(depth,camera):
    if torch.is_tensor(depth):
        xmap = torch.arange(camera.width, device=depth.device)
        ymap = torch.arange(camera.height, device=depth.device)
        if torch_version >= (1, 10):
            xmap, ymap = torch.meshgrid(xmap, ymap, indexing='xy')
        else:
            ymap,xmap = torch.meshgrid(ymap,xmap)

        points_z = depth / camera.scale
        points_x = (xmap - camera.cx) * points_z / camera.fx
        points_y = (ymap - camera.cy) * points_z / camera.fy
        cloud = torch.stack([points_x, points_y, points_z], dim=-1)
    else:
        xmap = np.arange(camera.width)
        ymap = np.arange(camera.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth / camera.scale
        points_x = (xmap - camera.cx) * points_z / camera.fx
        points_y = (ymap - camera.cy) * points_z / camera.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
    return cloud


def depth_to_ordered_cloud_torch(depth, camera):
    """
    Convert depth map to ordered point cloud (PyTorch version).

    Args:
        depth (torch.Tensor): Depth map of shape (B, H, W) or (H, W).
        camera: Object with attributes:
            - width (int), height (int)
            - fx, fy, cx, cy (float): Camera intrinsics.
            - scale (float): Depth scaling factor.

    Returns:
        torch.Tensor: Point cloud of shape (B, H, W, 3) or (H, W, 3).
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)  # Add batch dim if needed

    # Create meshgrid (batched)
    b, h, w = depth.shape
    xmap = torch.arange(w, device=depth.device).float()  # (W,)
    ymap = torch.arange(h, device=depth.device).float()  # (H,)
    if torch_version >= (1, 10):
        xmap, ymap = torch.meshgrid(xmap, ymap, indexing='xy')
    else:
        ymap,xmap = torch.meshgrid(ymap,xmap)

    # Expand to batch and move to same device as depth
    xmap = xmap.unsqueeze(0).expand(b, -1, -1)  # (B, H, W)
    ymap = ymap.unsqueeze(0).expand(b, -1, -1)  # (B, H, W)

    # Compute point cloud
    points_z = depth / camera.scale  # (B, H, W)
    points_x = (xmap - camera.cx) * points_z / camera.fx  # (B, H, W)
    points_y = (ymap - camera.cy) * points_z / camera.fy  # (B, H, W)

    # Stack coordinates
    cloud = torch.stack([points_x, points_y, points_z], dim=-1)  # (B, H, W, 3)

    return cloud.squeeze(0) if b == 1 else cloud  # Remove batch dim if input was 2D

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
        if torch.is_tensor(depth):
            cloud = torch.cat([cloud, rgb], dim=-1)
        else:
            cloud=np.concatenate([cloud,rgb],axis=-1)

    '''remove points with zero depth'''
    mask = cloud[:,:, 2] != 0
    cloud = cloud[mask]

    return cloud,mask

# def depth_to_point_clouds_torch(depth, camera,rgb=None):
#     '''check camera intrinsic'''
#     assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width), 'depth shape error! depth.shape = {}'.format(depth.shape)
#
#     '''process point clouds'''
#     cloud = depth_to_ordered_cloud_torch(depth,camera)
#
#     '''assign colors'''
#     if rgb is  not None:
#         cloud=torch.cat([cloud,rgb],dim=-1)
#
#     '''remove points with zero depth'''
#     mask = cloud[:,:, 2] != 0
#     cloud = cloud[mask]
#
#     return cloud,mask

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