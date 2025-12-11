# -*- coding: utf-8 -*-
import argparse
import subprocess
import torch
from Configurations.run_config import simulation_mode, activate_segmentation_queries
from Grasp_Agent_ import GraspAgent
from action import Action
from lib.ROS_communication import deploy_handover_rotate_command
from lib.dataset_utils import configure_smbclient
from lib.image_utils import depth_to_gray_scale, view_image, check_image_similarity
from process_perception import trigger_new_perception, get_side_bins_images, get_scene_depth, get_scene_RGB
from registration import view_colored_point_cloud

configure_smbclient()

'''return robot arm position to home'''
subprocess.run(["bash", './bash/pass_command.sh', "5"])

parser = argparse.ArgumentParser()
# parser.add_argument("--text-prompt", default="plate. stapler. banana. apple. box. shampoo.
# small ball.")
parser.add_argument("--text-prompt", default="medicine. drug.")
parser.add_argument("--placement-bin", default="s") # (g) means place the object ath the parallel gripper side bin while (s) means place the object at the suction side

args = parser.parse_args()
grasp_agent = GraspAgent()

'''test code'''
# grasp_agent.last_handover_action=Action()
# grasp_agent.last_handover_action.suction_at_home_position=False

grasp_agent.print_report()
grasp_agent.buffer.trim_uncompleted_episodes()
grasp_agent.initialize_check_points()
trigger_new_perception()

while True:
    img_suction_pre, img_grasp_pre,img_main_pre = get_side_bins_images()
    with torch.no_grad():
        '''get modalities'''
        depth=get_scene_depth()
        rgb=get_scene_RGB()
        grasp_agent.inputs(depth, rgb,args)
        if activate_segmentation_queries and grasp_agent.last_handover_action is None:
            # args.text_prompt=input('Enter object/s name/s to be found.')
            grasp_agent.publish_segmentation_query(args)
            grasp_agent.retrieve_segmentation_mask()
            # grasp_agent.view_mask_as_2dimage()
        # view_image(rgb)
        # view_colored_point_cloud(rgb,depth)
        '''infer dense action value pairs'''
        grasp_agent.sample_masked_actions()
        grasp_agent.report_current_scene_metrics()
        # grasp_agent.view_predicted_normals()

        while True:
            grasp_agent.dense_view(view_gripper_sampling=True)
            # exit()
            '''make decision'''
            while True:
                first_action_obj,second_action_obj=grasp_agent.pick_action()
                grasp_agent.actions_view(first_action_obj,second_action_obj)
                # r=input('To execute the action press Y')
                # if r.lower()=="y":
                break # comment out to visualize all actions one by one

            if first_action_obj is not None and not simulation_mode:
                '''execute action/s'''
                grasp_agent.deploy_action_metrics(first_action_obj,second_action_obj)
                for i in range(2):
                    first_action_obj,second_action_obj = grasp_agent.run_robot(first_action_obj,second_action_obj)
                    '''wait'''
                    first_action_obj,second_action_obj =grasp_agent.wait_robot_feedback(first_action_obj,second_action_obj)
                    if i==0 and first_action_obj.is_synchronous:
                        '''if dual actions is kinetically unfeasible switch to single action'''
                        switch_to_single_action,first_action_obj,second_action_obj=grasp_agent.completion_check_for_dual_grasp(first_action_obj,second_action_obj)
                        if switch_to_single_action:
                            '''second attempt'''
                            first_action_obj, second_action_obj = grasp_agent.run_robot(first_action_obj,
                                                                                        second_action_obj)
                            first_action_obj, second_action_obj = grasp_agent.wait_robot_feedback(first_action_obj,
                                                                                                  second_action_obj)
                        else:
                            break
                    else:break
                '''report result'''
                new_state_is_avaliable=grasp_agent.process_feedback(first_action_obj,second_action_obj, img_grasp_pre, img_suction_pre,img_main_pre)
                if new_state_is_avaliable: break
            else:
                trigger_new_perception()
                break
            '''clear dense data'''
        grasp_agent.rollback()