import random
import numpy as np
from action import Action
from lib.dataset_utils import online_data
from lib.dataset_utils import online_data2

online_data2=online_data2()

#[key] file id
'''gripper records'''
# [0] 1 if gripper is used else 1
# [1] 0 for grasp 1 for shift
# [2] grasp result
# [3] shift result
'''suction records'''
# [4] if suction is used else 1
# [5] 0 for grasp 1 for shift
# [6] grasp result
# [7] shift result

gripper_container_size=10
suction_container_size=10

online_data = online_data()

def sampling_p(sampling_rate,target_rate=0.25,exponent=10,k=0.75):
    if sampling_rate>target_rate:
        return ((target_rate - sampling_rate - 1)**exponent) *k
    else:
        return 1. - (sampling_rate/target_rate)*(1-k)

class DataTracker2():
    def __init__(self,name='',list_size=3):
        self.name=name
        self.dict=online_data2.load_pickle(self.name) if online_data2.file_exist(self.name) else {}
        self.list_size=list_size
        self.empty_list=[0]*list_size

    def dict_time_stamp(self):
        return online_data2.file_time_stamp(self.name)

    def get_value(self,key):
        if key in self.dict:
            return self.dict[key]
        else:
            return self.empty_list.copy()

    def push(self, action_obj:Action):
        new_record=self.get_value(action_obj.file_id)

        if action_obj.use_gripper_arm:
            '''gripper'''
            new_record[0] = 1 if action_obj.use_gripper_arm else 0
            new_record[1] = 0 if action_obj.is_grasp else 1
            new_record[2] = action_obj.grasp_result
            new_record[3] = action_obj.shift_result

        else:
            '''suction'''
            new_record[4] = 1 if action_obj.use_suction_arm else 0
            new_record[5] = 0 if action_obj.is_grasp else 1
            new_record[6] = action_obj.grasp_result
            new_record[7] = action_obj.shift_result

        self.dict[action_obj.file_id]=new_record

    def selective_grasp_sampling(self, size,sampling_rates=None):
        shuffled_keys=list(self.dict.keys())
        random.shuffle(shuffled_keys)
        sampling_probabilities=[sampling_p(sampling_rates[i]) for i in range(4)]
        ids=[]
        for key in shuffled_keys:
            record=self.dict[key]
            if len(ids)==size: break

            if record[0]==1 and record[1]==0:
                '''gripper grasp'''
                if record[3]==1 and np.random.random()<=sampling_probabilities[0]:
                    '''positive'''
                    ids.append(key)
                    continue
                elif np.random.random()<=sampling_probabilities[1]:
                    '''negative'''
                    ids.append(key)
                    continue

            if record[4]==1 and record[5]==0:
                '''suction grasp'''
                if record[3] == 1 and np.random.random()<=sampling_probabilities[2]:
                    '''positive'''
                    ids.append(key)
                    continue
                elif np.random.random()<=sampling_probabilities[3]:
                    '''negative'''
                    ids.append(key)
                    continue

        return ids


    def save(self):
        online_data2.save_pickle(self.name,self.dict)
        # save_dict(self.dict, self.path)

    def __len__(self):
        return len(self.dict)


def sample_positive_buffer(dict_name,size=None,disregard_collision_samples=False):
    positive_labels=[]
    data_tracker = DataTracker2(name=dict_name, list_size=10)

    for key in data_tracker.dict:
        record=data_tracker.dict[key]
        ground_truth=record[0]
        collision_state=record[2]
        if disregard_collision_samples and collision_state==1:continue

        if int(ground_truth)==1:
            positive_labels.append(key)
        else:
            continue

    random.shuffle(positive_labels)
    if size is not None and len(positive_labels) >= size: positive_labels=positive_labels[0:size]

    return positive_labels

def sample_all_positive_and_negatives(dict_name,list_size=10,shuffle=True,disregard_collision_samples=False):
    positive_labels=[]
    negative_labels=[]
    data_tracker = DataTracker2(name=dict_name, list_size=list_size)
    for key in data_tracker.dict:
        record=data_tracker.dict[key]

        collision_state=record[2]
        if disregard_collision_samples and collision_state == 1: continue
        # print(record)

        ground_truth=record[0]
        if int(ground_truth)==1:
            positive_labels.append(key)
        else:
            negative_labels.append(key)
    if shuffle:
        random.shuffle(positive_labels)
        random.shuffle(negative_labels)

    return positive_labels,negative_labels

def sample_random_buffer(dict_name,size=None,list_size=10,load_test_samples=1):
    positive_labels=[]
    negative_labels=[]
    data_tracker = DataTracker2(name=dict_name, list_size=list_size)

    for key in data_tracker.dict:
        record=data_tracker.dict[key]
        # print(record)

        ground_truth=record[0]
        if load_test_samples is not None:
            if record[1]!=load_test_samples:continue

        if int(ground_truth)==1:
            positive_labels.append(key)
        else:
            negative_labels.append(key)

    random.shuffle(positive_labels)
    random.shuffle(negative_labels)

    balanced_size=min(len(negative_labels),len(positive_labels))
    if size is not None:balanced_size=min(balanced_size,int(size/2))

    buffer_list=positive_labels[0:balanced_size]+negative_labels[0:balanced_size]

    return buffer_list

if __name__ == '__main__':
    pass
