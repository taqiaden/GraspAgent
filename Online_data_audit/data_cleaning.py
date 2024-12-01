import os
import numpy as np
import torch
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker, DataTracker
from label_unpack import LabelObj
from lib.dataset_utils import online_data
from lib.models_utils import initialize_model_state
from lib.report_utils import progress_indicator
from models.scope_net import scope_net_vanilla, suction_scope_model_state_path
from lib.report_utils import progress_indicator as pi


online_data = online_data()

def track_samples_scope_score():
    indexes=online_data.get_indexes()
    model = initialize_model_state(scope_net_vanilla(in_size=6), suction_scope_model_state_path)
    model.eval()

    counter=0
    total_counter=0
    for index in indexes:
        label=online_data.label.load(index)
        label_obj = LabelObj(label=label)
        if label_obj.is_gripper or label_obj.failure: continue
        normal=label_obj.normal
        approach=normal
        approach[2]=-approach[2]
        transition=label_obj.target_point


        rotation=np.empty((3,3))

        input = np.concatenate([transition, approach])
        input=torch.from_numpy(input).to('cuda')[None,...].float()

        # print(input.shape)
        score=model(input)
        total_counter+=1
        if score<0.5:
            counter+=1
            print(score.item())

    print(counter)
    print(total_counter)

def clean_old_data_redundancy():
    indexes=online_data.get_indexes()
    print(f'total samples = {len(indexes)}')

    pi = progress_indicator('progress ', max_limit=len(indexes))
    print('sort files ')
    indexes.sort()
    print()
    last_pc=None
    counters=[0,0]
    for i in range(len(indexes)):
        current_index=indexes[i]
        current_pc = online_data.point_clouds.load(current_index)

        if last_pc is not None:
            if last_pc.shape==current_pc.shape:
                counters[0]+=1
                dif=np.abs(last_pc-current_pc).sum()
                if dif==0.0:
                    counters[1]+=1
                    print(f'remove file with index {indexes[i-1]}')
                    online_data.point_clouds.remove_file(indexes[i-1])
                    online_data.label.remove_file(indexes[i-1])


        pi.step(i)

        last_pc=current_pc
    pi.end()
    print(f'similarities in shapes ={counters[0]}')
    print(f'total similarities ={counters[1]}')

def convert_all_point_clouds_to_depth():
    file_indexes = online_data.get_indexes()

    progress_indicator=pi(f'total samples size = {len(file_indexes)}, progress:  ',len(file_indexes))
    counter=0
    for i,target_file_index in enumerate(file_indexes):
        '''get data'''
        # depth=online_data.load_depth(target_file_index)
        pc = online_data.point_clouds.load_as_numpy(target_file_index)
        label_obj = LabelObj()
        depth = label_obj.get_depth(point_clouds=pc)

        online_data.depth.save_as_numpy(depth,target_file_index)

        '''update counter'''
        counter+=1
        progress_indicator.step(counter)

def rename_files():
    path=online_data.point_clouds.dir
    for filename in os.listdir(path):
        idx=online_data.get_index(filename)
        os.rename(os.path.join(path,filename),os.path.join(path,idx+online_data.point_clouds.sufix))

def check_collision_in_data():
    indexes=online_data.get_indexes()
    # indexes=sample_positive_buffer(size=None, dict_name=gripper_grasp_tracker,disregard_collision_samples=True)

    print(f'total samples = {len(indexes)}')

    pi = progress_indicator('progress ', max_limit=len(indexes))

    counters=[0,0]
    for i in range(len(indexes)):
        current_index=indexes[i]
        label = online_data.label.load_as_numpy(current_index)
        label_obj = LabelObj(label=label)
        if label_obj.failure or label_obj.is_suction: continue
        depth = online_data.depth.load_as_numpy(current_index)
        collision_state=label_obj.check_collision(depth=depth,visualize=True)

        if collision_state>0:
            counters[0]+=1
        else:
            counters[1]+=1
        pi.step(i)

    pi.end()
    print(f'instances with collision={counters[0]}')
    print(f'instances without collision={counters[1]}')
check_collision_in_data()






