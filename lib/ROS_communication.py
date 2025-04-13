import numpy as np
import trimesh

from Configurations import config
from Configurations.config import home_dir
from lib.grasp_utils import get_pose_matrixes, adjust_final_matrix, shift_a_distance

ROS_communication_file="ros_execute.txt"

gripper_grasp_data_path = config.home_dir + 'gripper_grasp_data_tmp.npy'
gripper_pre_grasp_data_path = config.home_dir + 'gripper_pre_grasp_data_tmp.npy'
suction_grasp_data_path = config.home_dir + 'suction_grasp_data_tmp.npy'
suction_pre_grasp_data_path = config.home_dir + 'suction_pre_grasp_data_tmp.npy'

gripper_pre_shift_data_path = config.home_dir + 'gripper_pre_shift_data_tmp.npy'
gripper_shift_data_path = config.home_dir + 'gripper_shift_data_tmp.npy'
gripper_end_shift_data_path = config.home_dir + 'gripper_end_shift_data_tmp.npy'

suction_shift_data_path = config.home_dir + 'suction_shift_data_tmp.npy'
suction_pre_shift_data_path = config.home_dir + 'suction_pre_shift_data_tmp.npy'
suction_end_shift_data_path = config.home_dir + 'suction_end_shift_data_tmp.npy'

handover_rotation_data_path=config.home_dir + 'rotation_angle_tmp.npy'


def deploy_action( action):
    if action.is_grasp and action.handover_state is None:
        '''grasp'''
        if action.use_gripper_arm:
            print('deploy gripper grasp command')
            deploy_gripper_grasp_command(action)
        else:
            '''suction'''
            print('deploy suction grasp command')
            deploy_suction_grasp_command(action)

    elif action.is_shift:
        '''shift'''
        if action.use_gripper_arm:
            print('deploy gripper shift command')
            deploy_gripper_shift_command(action)
        else:
            '''suction'''
            print('deploy suction shift command')
            deploy_suction_shift_command(action)

    elif action.handover_state is not None:
        '''handover'''
        if action.use_gripper_arm:
            if action.handover_state==0:
                '''hold'''
                print('deploy handover (gripper grasp) command')
                deploy_gripper_grasp_command(action)
            elif action.handover_state==1:
                '''rotate'''
                print('deploy handover (gripper rotate) command')
                deploy_handover_rotate_command()
            elif action.handover_state==2:
                '''handover'''
                print('deploy handover (suction grasp) command')
                deploy_gripper_grasp_command(action)
            else:
                '''drop'''
                print('deploy handover (drop) command')
                pass
        else:
            '''suction'''
            if action.handover_state==0:
                '''hold'''
                print('deploy handover (suction grasp) command')
                deploy_suction_grasp_command(action)
            elif action.handover_state==1:
                '''rotate'''
                print('deploy handover (suction rotate) command')
                deploy_handover_rotate_command()
            elif action.handover_state==2:
                '''handover'''
                print('deploy handover (gripper grasp) command')
                deploy_suction_grasp_command(action)
            else:
                '''drop'''
                print('deploy handover (drop) command')
                pass

def deploy_handover_rotate_command( angle=np.pi/2):
    angle=np.array(angle)
    np.save(handover_rotation_data_path, angle)


def deploy_gripper_grasp_command( action,angle=0.):
    pre_mat = adjust_final_matrix(action.transformation, x_correction=-0.23)
    end_mat = adjust_final_matrix(action.transformation, x_correction=-0.169)
    save_gripper_data(end_mat, action.real_width, gripper_grasp_data_path)
    save_gripper_data(pre_mat, action.real_width, gripper_pre_grasp_data_path)

    angle = np.array(angle)
    np.save(handover_rotation_data_path, angle)

def deploy_suction_grasp_command( action,angle=0.0):
    pre_mat = adjust_final_matrix(action.transformation, x_correction=-0.25)
    end_mat = adjust_final_matrix(action.transformation, x_correction=-0.184)
    save_suction_data(end_mat, suction_grasp_data_path)
    save_suction_data(pre_mat, suction_pre_grasp_data_path)

    angle = np.array(angle)
    np.save(handover_rotation_data_path, angle)

def deploy_gripper_shift_command( action):
    pre_mat=adjust_final_matrix(action.transformation, x_correction=-0.23)
    end_mat = action.transformation
    end_mat = shift_a_distance(end_mat,  0.005)
    end_mat=adjust_final_matrix(end_mat, x_correction=-0.169)

    shift_end_mat = np.copy(action.transformation)
    shift_end_mat[0:3, 3] = action.shift_end_point.cpu().numpy()
    # shift_end_mat=shift_a_distance(shift_end_mat, -0.01)
    shift_end_mat=adjust_final_matrix(shift_end_mat, x_correction=-0.169)

    save_gripper_data(pre_mat, action.real_width, gripper_pre_shift_data_path)
    save_gripper_data(end_mat, action.real_width, gripper_shift_data_path)
    save_gripper_data(shift_end_mat, action.real_width, gripper_end_shift_data_path)

def deploy_suction_shift_command( action):
    pre_mat = adjust_final_matrix(action.transformation, x_correction=-0.25)
    end_mat=action.transformation
    end_mat=shift_a_distance(end_mat,- 0.001)
    end_mat = adjust_final_matrix(end_mat, x_correction=-0.184)
    shift_end_mat = np.copy(action.transformation)
    shift_end_mat[0:3, 3] = action.shift_end_point.cpu().numpy()
    shift_end_mat=shift_a_distance(shift_end_mat,- 0.009)
    shift_end_mat = adjust_final_matrix(shift_end_mat, x_correction=-0.184)

    save_suction_data(pre_mat, suction_pre_shift_data_path)
    save_suction_data(end_mat, suction_shift_data_path)
    save_suction_data(shift_end_mat, suction_end_shift_data_path)

def save_suction_data(end_effecter_mat, file_path):
    wxyz = trimesh.transformations.quaternion_from_matrix(end_effecter_mat)
    xyz = end_effecter_mat[:3, 3]
    suction_data = np.array([xyz[0], xyz[1], xyz[2], wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    np.save(file_path, suction_data)
    # print('suction_data: ', suction_data)

def save_gripper_data(end_effecter_mat, grasp_width, file_path):
    wxyz = trimesh.transformations.quaternion_from_matrix(end_effecter_mat)
    xyz = end_effecter_mat[:3, 3]

    grasp_data = np.array([xyz[0], xyz[1], xyz[2], wxyz[1], wxyz[2], wxyz[3], wxyz[0], grasp_width])
    np.save(file_path, grasp_data)
def set_wait_flag():
    with open(config.home_dir + ROS_communication_file, 'w') as f:
        f.write('Wait')

def read_robot_feedback():
    with open(home_dir + ROS_communication_file, 'r') as f:
        txt = f.read()
    return txt
