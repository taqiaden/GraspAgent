# -*- coding: utf-8 -*-
import argparse
import torch
from Configurations.run_config import simulation_mode, activate_segmentation_queries
from Grasp_Agent_ import GraspAgent
from lib.dataset_utils import configure_smbclient
from lib.image_utils import depth_to_gray_scale, view_image
from process_perception import trigger_new_perception, get_side_bins_images, get_scene_depth, get_scene_RGB, \
    get_side_bins_RGB_images
from registration import view_colored_point_cloud

configure_smbclient()

parser = argparse.ArgumentParser()
# parser.add_argument("--text-prompt", default="plate. banana. apple. box. shampoo.
# small ball.")
parser.add_argument("--text-prompt", default="apple. ")

args = parser.parse_args()

grasp_agent = GraspAgent(args)
grasp_agent.initialize_check_points()
grasp_agent.report()

trigger_new_perception()

while True:
    # img_suction_pre, img_grasp_pre,img_main_pre = get_side_bins_images()
    img_suction_pre, img_grasp_pre,img_main_pre = get_side_bins_RGB_images()

    with torch.no_grad():
        '''get modalities'''
        depth=get_scene_depth()
        rgb=get_scene_RGB()
        grasp_agent.inputs(depth, rgb,args)
        if activate_segmentation_queries:
            grasp_agent.publish_segmentation_query()
            grasp_agent.retrieve_segmentation_mask()
        # view_image(rgb)
        # view_colored_point_cloud(rgb,depth)
        '''infer dense action value pairs'''
        grasp_agent.model_inference()
        grasp_agent.report_current_scene_metrics()
        # grasp_agent.view_mask_as_2dimage()
        # grasp_agent.view_predicted_normals()
        while True:
            grasp_agent.dense_view()
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
            '''clear dense data'''
        grasp_agent.clear()