import random

from Grasp_Agent_ import Action
from Online_data_audit.dictionary_utils import load_dict, save_dict
from label_unpack import LabelObj
from lib.dataset_utils import online_data
from lib.report_utils import progress_indicator as pi
from lib.statistics import moving_momentum

dictionary_directory=r'Online_data_audit/'

#[key] file id
# [0] 0 for gripper, 1 for suction
# [1] 0 for grasp 1 for shift
# [2] 0 for solo arm, 1 for both arms utilization
# [3] 0 for failed to grasp, 1 for successful grasp
# [4] 0 for failed shift, 1 for successful shift
gripper_container_size=10
suction_container_size=10

online_data = online_data()

class DataTracker2():
    def __init__(self,name='',list_size=3):
        self.name=name
        self.path=dictionary_directory+name+',pkl'
        self.dict=load_dict(self.path)

        self.list_size=list_size
        self.empty_list=[0]*list_size

        self.truncate_factor=1000

    def get_value(self,key):
        if key in self.dict:
            return self.dict[key]
        else:
            return self.empty_list.copy()

    def push(self, action_obj:Action):
        new_record=self.empty_list.copy()
        new_record[0]=0 if action_obj.use_gripper_arm else 1
        new_record[1]=0 if action_obj.is_grasp else 1
        new_record[2]=0 if not action_obj.is_synchronous else 1
        new_record[3]=0 if action_obj.grasp_result==0 else 1
        new_record[4]=0 if action_obj.shift_result==0 else 1
        self.dict[action_obj.file_id]=new_record

    def update_ground_truth_(self,file_id,ground_truth,list_index=0):
        old_record = self.get_value(file_id)
        new_record = old_record
        new_record[list_index]=ground_truth
        self.dict[file_id] = new_record

    def set_test_sample(self,file_id,list_index,data=1):
        old_record = self.get_value(file_id)
        new_record = old_record
        new_record[list_index] = data
        self.dict[file_id] = new_record

    def set_collision_state(self,file_id,list_index,data=2):
        old_record = self.get_value(file_id)
        new_record = old_record
        new_record[list_index] = data
        self.dict[file_id] = new_record

    def update_loss_record(self,file_ids,losses,start_index=0):
        for j in range(len(file_ids)):
            '''old record'''
            old_record=self.get_value(file_ids[j])

            '''compute'''
            first_moment = moving_momentum(old_record[1+start_index], losses[j].item(), decay_rate=0.99, exponent=1)
            second_moment = moving_momentum(old_record[2+start_index], losses[j].item(), decay_rate=0.99, exponent=2)

            '''truncated update'''
            new_record = old_record
            new_record[0+start_index]=float(int(losses[j] * self.truncate_factor))/self.truncate_factor
            new_record[1+start_index] = float(int(first_moment * self.truncate_factor))/self.truncate_factor
            new_record[2+start_index] = float(int(second_moment * self.truncate_factor))/self.truncate_factor

            # print(new_record)

            '''update'''
            self.dict[file_ids[j]] = new_record

    def save(self):
        save_dict(self.dict, self.path)

    def __len__(self):
        return len(self.dict)


def set_arm_dictionary(name,gripper=False,suction=False,clean_old_records=True,list_size=10):
    indexes = online_data.get_indexes()

    data_tracker = DataTracker2(name=name, list_size=list_size)
    if clean_old_records:data_tracker.dict.clear()
    progress_indicator = pi(f'total samples size = {len(indexes)}, progress:  ', len(indexes))

    for i in range(len(indexes)):
        label_obj = LabelObj(label=online_data.label.load_as_numpy(indexes[i]))
        if label_obj.is_suction and suction==False: continue
        if label_obj.is_gripper and gripper==False:continue
        data_tracker.update_ground_truth_(indexes[i], label_obj.success)
        progress_indicator.step(i)

    data_tracker.save()

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
