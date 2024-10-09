# -*- coding: utf-8 -*-
import torch
from lib.ROS_communication import wait_for_feedback, save_grasp_data
from lib.dataset_utils import configure_smbclient
from Configurations import config
import subprocess

from lib.grasp_utils import get_pose_matrixes
from process_perception import get_new_perception, get_side_bins_images, get_real_data

execute_suction_bash = './bash/run_robot_suction.sh'
execute_grasp_bash = './bash/run_robot_grasp.sh'

configure_smbclient()

def main():

    first_loop=True

    while True:
        if first_loop:
            get_new_perception()
            first_loop=False

        img_suction_pre, img_grasp_pre = get_side_bins_images()

        with torch.no_grad():
            full_pc = get_real_data()
            full_pc_torch = torch.from_numpy(full_pc).float().unsqueeze(0).cuda(non_blocking=True)

            for i in range(1000):

                with open(config.home_dir + "ros_execute.txt", 'w') as f:
                    f.write('Wait')
                state_ = 'Wait'
                action_ = ''

                print('***********************explore grasp***********************')
                pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T, k_end_effector=0.169, k_pre_grasp=0.23)
                save_grasp_data(end_effecter_mat, grasp_width, grasp_data_path)
                save_grasp_data(pre_grasp_mat, grasp_width, pre_grasp_data_path)

                action_ = 'grasp'

                subprocess.run(execute_grasp_bash)

                state_ = wait_for_feedback(state_)


if __name__ == "__main__":
    main()

