import numpy as np
from colorama import Fore
from Configurations.dynamic_config import get_int, save_key
from lib.dataset_utils import online_data

online_data=online_data()
grasp_data_counter_key='grasp_data'

def standard_label_structure(width, transformation, normal, target_point, use_gripper, use_suction, success):
    if use_gripper:
        arm_index=0
    elif use_suction:
        arm_index=1
    else:
        arm_index=None
    transformation = transformation.reshape(-1)

    label = target_point.tolist() + [success] + arm_index+transformation.tolist() + [width] + normal.tolist()
    return np.array(label)

def save_grasp_sample(rgb,depth,width, transformation, normal, target_point, use_gripper, use_suction, success ):
    '''set unique identifier'''
    index = get_int(grasp_data_counter_key) + 1

    '''set label'''
    label = standard_label_structure(width, transformation, normal, target_point, use_gripper, use_suction, success)

    '''save labeled sample'''
    online_data.rgb.save_as_image(rgb,idx=index)
    online_data.depth.save(depth,idx=depth)
    online_data.label.save(label,idx=index)

    if success==1:print(Fore.GREEN,'Report successful grasp attempt', Fore.RESET)
    else: print(Fore.YELLOW,'Report failed grasp attempt',Fore.RESET)

    '''update index'''
    save_key(grasp_data_counter_key, index)

    # tabulated data:
    # [0:3]: target_point
    # [3]: success
    # [4]: arm_index
    # [5:21]: transformation
    # [21]: width
    # [21:24]: normal
