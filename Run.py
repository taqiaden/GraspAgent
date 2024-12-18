# -*- coding: utf-8 -*-
import numpy as np
import torch
from Grasp_Agent_ import GraspAgent
from lib.bin_utils import empty_bin_check
from lib.dataset_utils import configure_smbclient
from lib.image_utils import depth_to_gray_scale
from process_perception import get_new_perception, get_side_bins_images, get_scene_depth,get_scene_RGB

configure_smbclient()
grasp_agent = GraspAgent()
grasp_agent.initialize_check_points()

while True:
    get_new_perception()
    img_suction_pre, img_grasp_pre = get_side_bins_images()
    with torch.no_grad():
        '''get modalities'''
        depth=get_scene_depth()
        # rgb=get_scene_RGB()
        random_rgb=depth_to_gray_scale(depth[:,:,np.newaxis], view=False, convert_to_three_channels=True, colorize=True)
        '''infer dense action value pairs'''
        grasp_agent.model_inference(depth,random_rgb)
        grasp_agent.view()
        '''make decision'''
        first_action_obj,second_action_obj=grasp_agent.pick_action()
        '''execute action/s'''
        actions, states, data = grasp_agent.execute(first_action_obj,second_action_obj)
        if grasp_agent.mode.simulation: get_new_perception()
        for action_, state_, data_ in zip(actions, states, data):
            if empty_bin_check(state_):
                # grasp_agent.initialize_check_points()
                break
            '''report result'''
            grasp_agent.process_feedback(action_, state_, data_, img_grasp_pre, img_suction_pre)
