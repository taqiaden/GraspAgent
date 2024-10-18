# -*- coding: utf-8 -*-
import torch
from Grasp_Agent_ import GraspAgent
from lib.bin_utils import empty_bin_check
from process_perception import get_new_perception, get_side_bins_images, get_real_data

grasp_agent = GraspAgent()
grasp_agent.initialize_check_points()

while True:
    get_new_perception()
    img_suction_pre, img_grasp_pre = get_side_bins_images()
    with torch.no_grad():
        point_clouds = get_real_data()
        grasp_agent.model_inference(point_clouds)
        actions, states, data = grasp_agent.execute()
        if grasp_agent.mode.simulation: get_new_perception()
        for action_, state_, data_ in zip(actions, states, data):
            if empty_bin_check(state_):
                grasp_agent.initialize_check_points()
                break
            grasp_agent.process_feedback(action_, state_, data_, img_grasp_pre, img_suction_pre)

