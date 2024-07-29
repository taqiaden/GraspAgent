import numpy as np
import cv2

from lib.depth_map import create_depth_image_from_point_cloud, CameraInfo, create_point_cloud_from_depth_image
from lib.pc_utils import refine_point_cloud, apply_mask
from lib.voxelization.voxelize3d import voxelize
from visualiztion import view_npy_open3d


def view_image(image,title=''):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def pc_to_depth_map(pc):
    # matrix = np.array([[0.010182, -0.999944, 0.003005, 0.39310000],
    #                    [-0.985716, -0.009532, 0.168148, -0.2809940],
    #                    [-0.168110, -0.004674, -0.985757, 1.3378300],
    #                    [0.0, 0.0, 0.0, 1.0]])
    matrix = np.array([[0.0, -1.0, 0.0, 0.393],
                       [-1.0, 0.0, 0.0, -0.280],
                       [0.0, 0.0, -1.0, 1.337],
                       [0.0, 0.0, 0.0, 1.0]])

    matrix_inv = np.linalg.inv(matrix)
    column = np.ones(len(pc))
    stacked = np.column_stack((pc, column))
    transformed = np.dot(matrix_inv, stacked.T).T[:, :3]
    transformed = np.ascontiguousarray(transformed)

    # view_npy_open3d(transformed,view_coordinate=True)
    # camera = CameraInfo(530, 600, 1122.375, 1122.375, 516, 386, 1000)
    camera = CameraInfo(712, 480, 1425.42, 1425.42, 685, 284, 1000)
    depth = create_depth_image_from_point_cloud(transformed, camera)

    # pc2 = create_point_cloud_from_depth_image(depth[:,:,np.newaxis], camera)
    # view_npy_open3d(pc2)
    view_image(depth)
    # view_image(depth[160:540,:])
    return depth[:,:,np.newaxis]

if __name__ == "__main__":
    rgb = cv2.imread('Frame_0.ppm')  # [1200,1920,3]
    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    rgb=rgb[229:709, 295:1007, :]
    view_image(rgb)

    pc = np.load('pc_tmp_data.npy')

    pc = refine_point_cloud(pc)
    pc = apply_mask(pc)
    depth=pc_to_depth_map(pc)
    RGB_D=np.concatenate([depth,rgb],axis=-1)
    RGB_D[:,:,[1,2,3]]=RGB_D[:,:,[3,2,1]]/255
    f = 1122.375 * 1.27
    camera = CameraInfo(712, 480, f, f, 688, 284, 1000)
    colored_pc = create_point_cloud_from_depth_image(RGB_D, camera)
    view_npy_open3d(colored_pc)

    # view_image(RGB_D)
    print(RGB_D.shape)
    print(depth.shape)
    print(rgb.shape)
    # voxels, coordinates, num_points_per_voxel=voxelize(pc,np.array([0.5,0.5,0.5]),np.array([-100,-100,-100,100,100,100]),10,1000)
    # print(voxels.shape)
    # print(coordinates)
    # print(num_points_per_voxel)
    # exit()

    # view_npy_open3d(pc,view_coordinate=True)

    # texture = cv2.imread('texture_image.jpg') #[772,1032,3]


    # img_suction = texture[200:540, 0:160, 1]
    # img_grasp = texture[180:570, 780:1032, 1]
    # heap_texture=texture[180:570,180:740,0]
    # print(pc.shape)
    # print(texture.shape)
    # print(rgb.shape)
    # view_image(texture)


