import numpy as np
import trimesh
from Configurations.config import home_dir
from lib.report_utils import wait_indicator as wi

ROS_communication_file="ros_execute.txt"

def save_suction_data(end_effecter_mat, file_path):
    wxyz = trimesh.transformations.quaternion_from_matrix(end_effecter_mat)
    xyz = end_effecter_mat[:3, 3]
    suction_data = np.array([xyz[0], xyz[1], xyz[2], wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    np.save(file_path, suction_data)
    # print('suction_data: ', suction_data)

def save_grasp_data(end_effecter_mat, grasp_width, file_path):
    wxyz = trimesh.transformations.quaternion_from_matrix(end_effecter_mat)
    xyz = end_effecter_mat[:3, 3]

    grasp_data = np.array([xyz[0], xyz[1], xyz[2], wxyz[1], wxyz[2], wxyz[3], wxyz[0], grasp_width])
    np.save(file_path, grasp_data)

def wait_for_feedback(txt):
    # wait until grasp or suction finished
    wait = wi('Waiting for robot feedback')
    while txt == 'Wait':
        wait.step(0.5)
        with open(home_dir + ROS_communication_file, 'r') as f:
            txt = f.read()
    else:
        wait.end()
        print('Robot state: ' + txt)
    return txt