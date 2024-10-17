from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from registration import camera
from visualiztion import visualize_suction_pose

def view_suction_label(depth,normal,pixel_index,batch_size):
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index[j].cpu().numpy(), depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)
        target_normal = normal[j].cpu().numpy()
        from grasp_post_processing import get_suction_pose_
        target_point, pre_grasp_mat, end_effecter_mat, T, normal = get_suction_pose_(target_point, target_normal)

        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        '''view'''
        visualize_suction_pose(target_point, normal.reshape(1, 3) , T, end_effecter_mat, npy=pc)