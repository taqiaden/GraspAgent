# -*- coding: utf-8 -*-
import sys

import numpy
import numpy as np
import torch
from Configurations.run_config import simulation_mode, use_real_rgb
from Grasp_Agent_ import GraspAgent
from lib.dataset_utils import configure_smbclient
from lib.image_utils import depth_to_gray_scale, view_image
from process_perception import trigger_new_perception, get_side_bins_images, get_scene_depth, get_scene_RGB
from registration import view_colored_point_cloud

configure_smbclient()
grasp_agent = GraspAgent()
grasp_agent.initialize_check_points()

while True:
    trigger_new_perception()
    # img_suction_pre, img_grasp_pre,img_main_pre = get_side_bins_images()
    with torch.no_grad():
        '''get modalities'''
        depth=get_scene_depth()
        if use_real_rgb:
            rgb=get_scene_RGB()
        else:
            rgb=depth_to_gray_scale(depth[:,:,np.newaxis], view=False, convert_to_three_channels=True, colorize=True)
        # view_image(rgb)
        # view_colored_point_cloud(rgb,depth)
        '''infer dense action value pairs'''
        grasp_agent.model_inference(depth,rgb)
        # grasp_agent.dense_view()
        '''make decision'''
        first_action_obj,second_action_obj=grasp_agent.pick_action()
        grasp_agent.actions_view(first_action_obj,second_action_obj)
        if first_action_obj is not None and not simulation_mode:
            '''execute action/s'''
            first_action_obj,second_action_obj = grasp_agent.execute(first_action_obj,second_action_obj)
            '''wait'''
            first_action_obj,second_action_obj =grasp_agent.wait_robot_feedback(first_action_obj,second_action_obj)
            '''report result'''
            grasp_agent.process_feedback(first_action_obj,second_action_obj, img_grasp_pre, img_suction_pre,img_main_pre)
        '''clear dense data'''
        grasp_agent.clear()
