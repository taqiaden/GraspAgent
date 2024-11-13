import random

from Online_data_audit.dictionary_utils import load_dict, save_dict
from label_unpack import LabelObj
from lib.dataset_utils import online_data
from lib.statistics import moving_momentum
from lib.report_utils import progress_indicator as pi

dictionary_directory=r'Online_data_audit/'
gripper_grasp_tracker=r'gripper_grasp_dict'
suction_grasp_tracker=r'suction_grasp_dict'

online_data = online_data()


class DataTracker():
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

    def update_ground_truth_(self,file_id,ground_truth,list_index=0):
        old_record = self.get_value(file_id)
        new_record = old_record
        new_record[list_index]=ground_truth
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

def set_gripper_dictionary():
    set_arm_dictionary(gripper_grasp_tracker,gripper=True)

def set_suction_dictionary():
    set_arm_dictionary(suction_grasp_tracker,suction=True)

def set_arm_dictionary(name,gripper=False,suction=False,clean_old_records=True):
    indexes = online_data.get_indexes()

    data_tracker = DataTracker(name=name, list_size=4)
    if clean_old_records:data_tracker.dict.clear()
    progress_indicator = pi(f'total samples size = {len(indexes)}, progress:  ', len(indexes))

    for i in range(len(indexes)):
        label_obj = LabelObj(label=online_data.label.load_as_numpy(indexes[i]))
        if label_obj.is_suction and suction==False: continue
        if label_obj.is_gripper and gripper==False:continue
        data_tracker.update_ground_truth_(indexes[i], label_obj.success)
        progress_indicator.step(i)

    data_tracker.save()

def sample_random_buffer(dict_name,size=None):
    positive_labels=[]
    negative_labels=[]
    data_tracker = DataTracker(name=dict_name, list_size=4)

    for key in data_tracker.dict:
        record=data_tracker.dict[key]
        # print(record)

        ground_truth=record[0]
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
    set_suction_dictionary()
    set_gripper_dictionary()

    balanced_list=sample_random_buffer(dict_name=suction_grasp_tracker)

    print(len(balanced_list))

