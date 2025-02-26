# -*- coding: utf-8 -*-
import argparse
import torch
from Configurations.run_config import simulation_mode, activate_segmentation_queries
from Grasp_Agent_ import GraspAgent
from lib.dataset_utils import configure_smbclient
from lib.image_utils import depth_to_gray_scale, view_image, check_image_similarity
from process_perception import trigger_new_perception, get_side_bins_images, get_scene_depth, get_scene_RGB
from registration import view_colored_point_cloud

configure_smbclient()

parser = argparse.ArgumentParser()
# parser.add_argument("--text-prompt", default="plate. stapler. banana. apple. box. shampoo.
# small ball.")
parser.add_argument("--text-prompt", default="gloves. ")
parser.add_argument("--placement-bin", default="g") # (g) means place the object ath the parallel gripper side bin while (s) means place the object at the suction side

args = parser.parse_args()
grasp_agent = GraspAgent()
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
        if activate_segmentation_queries:
            # args.text_prompt=input('Enter object/s name/s to be found.')
            grasp_agent.publish_segmentation_query(args)
            grasp_agent.retrieve_segmentation_mask()
            # grasp_agent.view_mask_as_2dimage()
        # view_image(rgb)
        # view_colored_point_cloud(rgb,depth)
        '''infer dense action value pairs'''
        grasp_agent.model_inference()
        # grasp_agent.report_current_scene_metrics()
        grasp_agent.view_predicted_normals()

        while True:
            # grasp_agent.dense_view()
            '''make decision'''
            first_action_obj,second_action_obj=grasp_agent.pick_action()
            # grasp_agent.actions_view(first_action_obj,second_action_obj)
            if first_action_obj is not None and not simulation_mode:
                '''execute action/s'''
                first_action_obj,second_action_obj = grasp_agent.execute(first_action_obj,second_action_obj)
                '''wait'''
                first_action_obj,second_action_obj =grasp_agent.wait_robot_feedback(first_action_obj,second_action_obj)
                '''report result'''
                new_state_is_avaliable=grasp_agent.process_feedback(first_action_obj,second_action_obj, img_grasp_pre, img_suction_pre,img_main_pre)
                if new_state_is_avaliable: break
            else:
                break
            '''clear dense data'''
        grasp_agent.rollback()