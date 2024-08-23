from lib.bbox import decode_gripper_pose
from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from pose_object import approach_vec_to_theta_phi, output_processing
from registration import camera
from visualiztion import vis_scene


def view_gripper_label(depth,pose_7,pixel_index,batch_size):
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index[j].cpu().numpy(), depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

        target_pose = pose_7[j]
        approach = target_pose[:, 0:3]
        theta, phi_sin, phi_cos = approach_vec_to_theta_phi(approach)
        target_pose[:, 0:1] = theta
        target_pose[:, 1:2] = phi_sin
        target_pose[:, 2:3] = phi_cos
        pose_5 = output_processing(target_pose[:, :, None]).squeeze(-1)
        pose_good_grasp = decode_gripper_pose(pose_5, target_point)

        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        '''view'''
        vis_scene(pose_good_grasp[:, :].reshape(1, 14), npy=pc)