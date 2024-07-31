import numpy as np
import cv2
from PIL import Image
from Configurations.ENV_boundaries import depth_lower_bound, depth_factor
from lib.depth_map import create_depth_image_from_point_cloud, CameraInfo, create_point_cloud_from_depth_image
from lib.image_utils import view_image
from lib.pc_utils import refine_point_cloud, apply_mask
from visualiztion import view_npy_open3d

camera = CameraInfo(712, 480, 1428, 1466, 682, 287, 1000)

def depth_image_to_voxel(RGB_D, camera):
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
        max_=np.max(cloud[:,:,2])
        cloud=np.concatenate([cloud,colors],axis=-1)
        cloud = cloud.reshape([-1, 6])
    else:
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud = cloud.reshape([-1, 3])
    mask = cloud[:, 2] != 0
    cloud = cloud[mask]

    # cloud2=np.copy(cloud)
    # cloud3=np.copy(cloud)
    # upper_bound=1.3192
    # lower_bound=1.0992
    # print(max_)
    # cloud2[:,2]=cloud2[:,2]*0.0+max_+0.1
    # cloud3[:,2]=cloud3[:,2]*0.0+max_-0.3
    # cloud=np.concatenate([cloud,cloud2,cloud3],axis=0)

    view_npy_open3d(cloud)
    return cloud
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

def pc_to_depth_map(pc):
    transformed = transform_to_camera_frame(pc)
    depth = create_depth_image_from_point_cloud(transformed, camera)
    return depth[:,:,np.newaxis]

def get_rgb_heap(rgb_full):
    rgb = cv2.rotate(rgb_full, cv2.ROTATE_180)
    heap_rgb = rgb[229:709, 295:1007, :]
    return heap_rgb

def standardize_depth(depth,reverse=False):
    result_depth=np.copy(depth)
    if reverse:
        result_depth = (result_depth * depth_factor) + depth_lower_bound
        result_depth[result_depth ==depth_lower_bound] = 0.0
    else:
        result_depth[result_depth < 0.0001] = depth_lower_bound
        result_depth=(result_depth-depth_lower_bound)/depth_factor
    return result_depth

if __name__ == "__main__":
    rgb = cv2.imread('Frame_0.ppm')  # [1200,1920,3]
    # rgb_full=cv2.cvtColor(bgr_full,cv2.COLOR_BGR2RGB)
    # view_image(rgb_full)

    heap_rgb=get_rgb_heap(rgb)
    # view_image(heap_rgb)

    pc = np.load('pc_tmp_data.npy')
    pc = refine_point_cloud(pc)
    pc = apply_mask(pc)
    heap_depth=pc_to_depth_map(pc)

    standard_heap_depth=standardize_depth(heap_depth, reverse=False)
    # view_image(standard_heap_depth)

    # print(heap_depth[0,0,0])
    # print(standard_heap_depth[0,0,0])
    # print(standard_heap_depth)
    print(heap_depth.shape)
    np.save('depth.npy',heap_depth)
    test = np.load('depth.npy')


    print(np.sum(test-heap_depth))

    view_image(test)
    #
    # print(heap_rgb.shape)
    # BGR=cv2.cvtColor(heap_rgb,cv2.COLOR_RGB2BGR)
    # view_image(heap_rgb)
    cv2.imwrite('rgb.png',heap_rgb)
    np.save('rgb.npy', heap_rgb)
    test = cv2.imread('rgb.png')
    print(np.sum(test-heap_rgb))

    # view_image(test)

    heap_rgb=cv2.cvtColor(np.float32(heap_rgb), cv2.COLOR_BGR2RGB)/255
    RGB_D=np.concatenate([heap_depth,heap_rgb],axis=-1)
    np.save('rgb_d.npy',RGB_D)
    depth_image_to_voxel(RGB_D, camera)
    colored_pc = create_point_cloud_from_depth_image(RGB_D, camera)


