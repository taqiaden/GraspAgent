import numpy as np
from colorama import Fore

from Configurations.dynamic_config import get_int, save_key
from action import Action
from lib.dataset_utils import online_data2

online_data2=online_data2()
grasp_data_counter_key='episodic_data_counter2'

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

def standard_label_structure(gripper_action:Action ,suction_action:Action,step_number,is_end_of_task=None):

    label = (gripper_action.target_point.tolist() + suction_action.target_point.tolist()
        +gripper_action.transformation.reshape(-1).tolist() + suction_action.transformation.reshape(-1).tolist()
             + [gripper_action.real_width] + gripper_action.shift_end_point.tolist()+suction_action.shift_end_point.tolist()+
             [articulate_action_index(gripper_action)] + [articulate_action_index(suction_action)]
                + [articulate_action_result(gripper_action)] + [articulate_action_result(suction_action)]
             + [step_number]+[is_end_of_task])

    assert len(label)==51

    return np.array(label)


def save_grasp_sample(rgb,depth,mask, gripper_action ,suction_action ,run_sequence):
    '''set unique identifier'''
    index = get_int(grasp_data_counter_key) + 1

    '''set label'''
    label = standard_label_structure(gripper_action ,suction_action,run_sequence)

    '''save labeled sample'''
    online_data2.rgb.save_as_image(rgb,idx=index)
    online_data2.depth.save(depth,idx=index)
    online_data2.mask.save(mask,idx=index)
    online_data2.label.save(label,idx=index)

    gripper_action.file_id=index
    suction_action.file_id=index

    '''update index'''
    save_key(grasp_data_counter_key, index)

    print(Fore.LIGHTBLACK_EX,f'Save labeled data to : {online_data2.address}/{index}*',Fore.RESET)

    return gripper_action,suction_action

    # tabulated data:
    # [0:3]: gripper_target_point
    # [3:6]: suction_target_point
    # [6:22]: gripper_transformation
    # [22:38]: suction_transformation
    # [38]: gripper_width
    # [39:42] gripper shift end position
    # [42:45] suction shift end position
    # [45]: gripper_action_index
    # [46]: suction_action_index
    # [47]: gripper_result
    # [48]: suction_result
    # [49]: step_number
    # [50]: is end of an episode