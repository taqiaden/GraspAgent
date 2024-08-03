import numpy as np
import cv2
from Configurations.ENV_boundaries import depth_lower_bound, depth_factor
from lib.depth_map import point_clouds_to_depth, CameraInfo, depth_to_point_clouds
from lib.image_utils import view_image
from lib.pc_utils import refine_point_cloud
from visualiztion import view_npy_open3d

camera = CameraInfo(712, 480, 1428, 1466, 682, 287, 1000)

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

def standardize_depth(depth,reverse=False):
    result_depth=np.copy(depth)
    if reverse:
        result_depth = (result_depth * depth_factor) + depth_lower_bound
        result_depth[result_depth ==depth_lower_bound] = 0.0
    else:
        result_depth[result_depth < 0.0001] = depth_lower_bound
        result_depth=(result_depth-depth_lower_bound)/depth_factor
    return result_depth

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
