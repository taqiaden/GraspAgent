import numpy as np
import trimesh
from Configurations import config, ENV_boundaries
from Configurations.run_config import simulation_mode
from lib.ROS_communication import save_grasp_data, save_suction_data
from lib.bbox import decode_gripper_pose
from lib.collision_unit import local_exploration
from lib.grasp_utils import get_homogenous_matrix, get_grasp_width, get_pose_matrixes
from visualiztion import vis_scene, visualize_suction_pose

exploration_probabilty=0.0
view_masked_grasp_pose = False
grasp_data_path = config.home_dir + 'grasp_data_tmp.npy'
pre_grasp_data_path = config.home_dir + 'pre_grasp_data_tmp.npy'
suction_data_path = config.home_dir + 'suction_data_tmp.npy'
pre_suction_data_path = config.home_dir + 'pre_suction_data_tmp.npy'
def inference_dense_gripper_pose(point_data_npy,center_point,index):
    global poses
    # point_data = torch.from_numpy(point_data_npy).to('cuda')
    # point_data = point_data[None, :, :]
    # poses=dense_gripper_net.dense_gripper_generator_net_(point_data)
    pose=poses[:,:,index]



    print('prediction------------------------------------------------',pose)
    # if np.random.random()>0.3:
    #     pose=torch.rand_like(pose)
    #     print('random pose=------------------------------------------------',pose)

    # gripper_pose_net.gripper_net.eval()
    # theta_phi_output_GAN=gripper_pose_net.gripper_net(depth_image,center_point_)
    # output = theta_phi_output_GAN

    pose_good_grasp=decode_gripper_pose(pose,center_point)
    return pose_good_grasp

def get_suction_pose_( target_point, normal):
    suction_pose = normal.reshape(1, 3)  # Normal [[xn,yn,zn]]
    target_point=target_point.reshape(1,3)

    pose_good_suction = np.concatenate((target_point, suction_pose), axis=1)  # [[x,y,z,xn,yn,zn]]
    position = pose_good_suction[0, [0, 1, 2]]  # [x,y,z]
    v0 = np.array([1, 0, 0])
    v1 = -pose_good_suction[0, [3, 4, 5]]  # [-xn,-yn,-zn]
    pred_approch_vector = pose_good_suction[0, [3, 4, 5]]
    a = trimesh.transformations.angle_between_vectors(v0, v1)
    b = trimesh.transformations.vector_product(v0, v1)
    matrix_ori = trimesh.transformations.rotation_matrix(a, b)
    matrix_ori[:3, 3] = position.T
    T = matrix_ori

    pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T, k_end_effector=0.184, k_pre_grasp=0.25)

    target_point = target_point.squeeze()

    return target_point, pre_grasp_mat, end_effecter_mat, suction_pose, T, pred_approch_vector

def get_suction_pose(index, point_data, normal):
    return  get_suction_pose_(point_data[index],normal)
def  gripper_processing(index,point_data,isvis):
    # view_npy_open3d(point_data,view_coordinate=True)

    # Get the pose_good_grasp
    center_point = point_data[index]
    # print('center------------',center_point)
    # view_npy_open3d(point_data,view_coordinate=True)
    # print(point_data.shape)
    pose_good_grasp=inference_dense_gripper_pose(point_data, center_point, index)

    # pose_good_grasp=inference_gripper_pose(point_data,center_point,index)
    # vis_scene(pose_good_grasp[:, :].reshape(1, 14),npy=point_data)
    activate_exploration=True if np.random.rand()<exploration_probabilty else False
    # view_npy_open3d(point_data,view_coordinate=True)
    pose_good_grasp,collision_intensity = local_exploration(pose_good_grasp,point_data, exploration_attempts=5,
                                             explore_if_collision=False, view_if_sucess=view_masked_grasp_pose,explore=activate_exploration)
    success=collision_intensity==0
    # collision_intensity=1.0 if success else 0.0
    # view_npy_open3d(point_data,view_coordinate=True)

    # Get related parameters
    T = get_homogenous_matrix(pose_good_grasp)
    distance = pose_good_grasp[0, -1]

    grasp_width = get_grasp_width(pose_good_grasp)
    # gripper_net_processing(point_data, index, pose_good_grasp,collision_intensity,center_point_)

    if not success:
        return False, pose_good_grasp,grasp_width, distance, T, center_point

    if simulation_mode==False:
        pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T, k_end_effector=0.169, k_pre_grasp=0.23)
        save_grasp_data(end_effecter_mat, grasp_width, grasp_data_path)
        save_grasp_data(pre_grasp_mat, grasp_width, pre_grasp_data_path)

    if isvis: vis_scene(pose_good_grasp[:, :].reshape(1, 14),npy=point_data)

    return True, pose_good_grasp,grasp_width, distance, T, center_point

def suction_processing(index,point_data,isvis):

    global normals
    normal=normals[index]
    normal=normal[None,:]

    suction_xyz, pre_grasp_mat, end_effecter_mat, suction_pose, T, pred_approch_vector \
        = get_suction_pose(index, point_data, normal)
    # suction_net_processing(point_data, index)

    if pre_grasp_mat[0, 3] < ENV_boundaries.x_min_dis:
        return False, suction_xyz,pred_approch_vector

    save_suction_data(end_effecter_mat, suction_data_path)
    save_suction_data(pre_grasp_mat, pre_suction_data_path)

    if isvis: visualize_suction_pose(suction_xyz, suction_pose, T, end_effecter_mat,npy=point_data)

    return True,suction_xyz,pred_approch_vector