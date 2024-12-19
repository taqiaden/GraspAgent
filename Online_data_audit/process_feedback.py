import numpy as np
from colorama import Fore

from Configurations.dynamic_config import get_int, save_key
from lib.dataset_utils import online_data2

online_data2=online_data2()
grasp_data_counter_key='grasp_data'

def articulate_action_index(action):
    # 0 for idle, 1 for grasp, 2 for shift
    if action is None:
        return 0
    elif action.is_grasp:
        return 1
    elif action.is_shift:
        return 2
    else:
        return 0

def articulate_action_result(action):
    # 0 for failure, 1 for success, or None for no record
    if action is None:
        return None
    elif action.is_grasp:
        return action.grasp_result
    elif action.is_shift:
        return action.shift_result
    else:
        return None

def standard_label_structure(gripper_action ,suction_action):
    label = (gripper_action.target_point.tolist() + suction_action.target_point.tolist()
        +gripper_action.transformation.reshape(-1).tolist() + suction_action.transformation.reshape(-1).tolist()
             + [gripper_action.width] + articulate_action_index(gripper_action) + articulate_action_index(suction_action)
                + articulate_action_result(gripper_action) + articulate_action_result(suction_action))
    assert len(label)==43
    return np.array(label)

def print_(action_name,result,arm_name):
    if result is not None:
        print(Fore.LIGHTBLACK_EX, f'Report {action_name} action on {arm_name} arm', Fore.RESET)
    elif result==1:
        print(Fore.GREEN, f'Report successful {action_name} action on {arm_name} arm', Fore.RESET)
    elif result == 1:
        print(Fore.YELLOW, f'Report failed {action_name} action on {arm_name} arm', Fore.RESET)
    else:
        assert 1==2, 'Error while reporting an action'

def print_result(action):
    if action is not None:
        if action.is_grasp:
            action_name='grasp'
            result=action.grasp_result
        else:
            action_name='shift'
            result=action.shift_result

        arm_name='gripper' if action.use_gripper_arm else 'suction'
        print_(action_name, result, arm_name)



def save_grasp_sample(rgb,depth, gripper_action ,suction_action ):
    '''set unique identifier'''
    index = get_int(grasp_data_counter_key) + 1

    '''set label'''
    label = standard_label_structure(gripper_action ,suction_action)

    '''save labeled sample'''
    online_data2.rgb.save_as_image(rgb,idx=index)
    online_data2.depth.save(depth,idx=depth)
    online_data2.label.save(label,idx=index)

    '''update index'''
    save_key(grasp_data_counter_key, index)

    '''print brief report'''
    print_result(gripper_action)
    print_result(suction_action)

    # tabulated data:
    # [0:3]: gripper_target_point
    # [3:6]: suction_target_point
    # [6:22]: gripper_transformation
    # [22:38]: suction_transformation
    # [38]: gripper_width
    # [39]: gripper_action_index
    # [40]: suction_action_index
    # [41]: gripper_result
    # [42]: suction_result
