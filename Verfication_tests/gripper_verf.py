from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from pose_object import pose_7_to_transformation
from registration import camera
from visualiztion import vis_scene


def view_gripper_batch(depth,pose_7,pixel_index,batch_size):
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index[j].cpu().numpy(), depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

        target_pose_7 = pose_7[j]

        T_d, width, distance=pose_7_to_transformation(target_pose_7, target_point)

        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        '''view'''
        vis_scene(T_d, width, npy=pc)


def view_single_gripper_grasp(depth,pose_7,pixel_index,j,pix_A, pix_B):
        depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index[j].cpu().numpy(), depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

        target_pose_7 = pose_7[j]

        T_d, width, distance=pose_7_to_transformation(target_pose_7, target_point)

        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        '''view'''
        vis_scene(T_d, width, npy=pc)