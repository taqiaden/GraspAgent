import numpy as np
import cv2
from Configurations.ENV_boundaries import depth_lower_bound, depth_mean, depth_std
from lib.depth_map import point_clouds_to_depth, CameraInfo, depth_to_point_clouds, transform_to_camera_frame
from lib.image_utils import view_image
from lib.pc_utils import refine_point_cloud
from visualiztion import view_npy_open3d

camera = CameraInfo(712, 480, 1428, 1466, 682, 287, 1000)



def pc_to_depth_map(pc):
    transformed = transform_to_camera_frame(pc)
    depth = point_clouds_to_depth(transformed, camera)
    return depth[:,:,np.newaxis]

def depth_map_to_pc(depth):
    pc,mask = depth_to_point_clouds(depth, camera)
    pc = transform_to_camera_frame(pc,reverse=True)
    return pc

def get_rgb_heap(rgb_full):
    rgb = cv2.rotate(rgb_full, cv2.ROTATE_180)
    heap_rgb = rgb[229:709, 295:1007, :]
    return heap_rgb

def standardize_depth(depth):
    depth[depth < 0.0001] = depth_lower_bound
    depth=(depth-depth_mean)/depth_std
    return depth

def view_colored_point_cloud(RGB,Depth):
    heap_rgb = cv2.cvtColor(np.float32(RGB), cv2.COLOR_BGR2RGB) / 255
    colored_pc,mask = depth_to_point_clouds(Depth.squeeze(), camera,rgb=heap_rgb)
    colored_pc[:,0:3] = transform_to_camera_frame(colored_pc[:,0:3], reverse=True)
    view_npy_open3d(colored_pc)

if __name__ == "__main__":
    rgb = cv2.imread('Frame_0.ppm')  # [1200,1920,3]

    heap_rgb=get_rgb_heap(rgb)

    pc = np.load('pc_tmp_data.npy')
    pc = refine_point_cloud(pc)

    heap_depth=pc_to_depth_map(pc)

    standard_heap_depth=standardize_depth(heap_depth, reverse=False)

    view_image(heap_rgb)
    view_image(heap_depth)
    view_colored_point_cloud(heap_rgb,heap_depth)
