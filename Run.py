# -*- coding: utf-8 -*-
import torch
from Grasp_Agent_ import GraspAgent
from lib.bin_utils import empty_bin_check
from lib.dataset_utils import configure_smbclient
from process_perception import get_new_perception, get_side_bins_images, get_scene_point_clouds, get_scene_depth, \
    get_scene_RGB

configure_smbclient()

grasp_agent = GraspAgent()
grasp_agent.initialize_check_points()

while True:
    get_new_perception()
    img_suction_pre, img_grasp_pre = get_side_bins_images()
    with torch.no_grad():
        # point_clouds = get_scene_point_clouds()
        depth=get_scene_depth()
        rgb=get_scene_RGB()
        grasp_agent.model_inference(depth,rgb)
        actions, states, data = grasp_agent.execute()
        if grasp_agent.mode.simulation: get_new_perception()
        for action_, state_, data_ in zip(actions, states, data):
            if empty_bin_check(state_):
                grasp_agent.initialize_check_points()
                break
            grasp_agent.process_feedback(action_, state_, data_, img_grasp_pre, img_suction_pre)

