import random
import numpy as np
from action import Action
from label_unpack import LabelObj
from lib.dataset_utils import online_data
from lib.dataset_utils import online_data2

online_data2=online_data2()

#[key] file id
'''gripper records'''
# [0] 1 if gripper is used else 0
# [1] 0 for grasp 1 for shift 2 for handover
# [2] grasp result
# [3] shift result
'''suction records'''
# [4] 1 if suction is used else 0
# [5] 0 for grasp 1 for shift 2 for handover
# [6] grasp result
# [7] shift result
'''both records'''
# [8] handover result

gripper_container_size=10
suction_container_size=10
online_data = online_data()

def sampling_p(sampling_rate,target_rate=0.25,exponent=10,min_p=0.001):
    # print('rate',sampling_rate)
    if sampling_rate>target_rate:
        p= ((target_rate - sampling_rate - 1)**-exponent) *target_rate
    else:
        p= 1. - (sampling_rate) *(1 - target_rate)/(target_rate)
    p=max(p,min_p)
    # print('p',p)
    return p

def sampling_p2(x,balance_indicator,probability_exponent=2.0):
    pivot_point = np.sqrt(np.abs(balance_indicator)) * np.sign(balance_indicator)
    xa = (1 - x) * pivot_point
    selection_probability = ((1 - pivot_point) / 2 + xa + 0.5 * (1 - abs(pivot_point)))
    selection_probability = selection_probability ** probability_exponent
    return selection_probability

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

    def old_push(self, label_obj_:LabelObj,file_id_):
        new_record = self.get_value(file_id_)
        if label_obj_.is_gripper:
            '''gripper'''
            new_record[0] = 1
            new_record[1] = 0
            new_record[2] = label_obj_.success

        if label_obj_.is_suction:
            '''suction'''
            new_record[4] = 1
            new_record[5] = 0
            new_record[6] = label_obj_.success

        self.dict[file_id_]=new_record
        # print(new_record)


    def push(self, action_obj:Action):
        new_record=self.get_value(action_obj.file_id)

        if action_obj.use_gripper_arm:
            '''gripper'''
            new_record[0] = 1 if action_obj.use_gripper_arm else 0
            if action_obj.handover_state is not None:
                new_record[1]=2
            else:
                new_record[1] = 0 if action_obj.is_grasp else 1
            new_record[2] = action_obj.grasp_result
            new_record[3] = action_obj.shift_result

        if action_obj.use_suction_arm:
            '''suction'''
            new_record[4] = 1 if action_obj.use_suction_arm else 0
            if action_obj.handover_state is not None:
                new_record[5]=2
            else:
                new_record[5] = 0 if action_obj.is_grasp else 1
            new_record[6] = action_obj.grasp_result
            new_record[7] = action_obj.shift_result

        new_record[8]=action_obj.handover_result

        self.dict[action_obj.file_id]=new_record

    def gripper_grasp_sampling(self,size,balance_indicator,data_pool=None):
        shuffled_keys = list(self.dict.keys())
        random.shuffle(shuffled_keys)
        positive_selection_p=sampling_p2(x=1,balance_indicator=balance_indicator,probability_exponent=10)
        negative_selection_p=sampling_p2(x=0,balance_indicator=balance_indicator,probability_exponent=10)

        # print(f'p={positive_selection_p}, n={negative_selection_p}')

        ids = []
        for key in shuffled_keys:
            if data_pool is not None:
                if not data_pool.label.exist(key):
                    print(f'Missing label: {key}')
                    continue
                if not data_pool.depth.exist(key):
                    print(f'Missing depth: {key}')
                    continue
            record = self.dict[key]
            if len(ids) == size: break

            if record[0] == 1 and record[1] == 0:
                '''gripper grasp'''
                ran=np.random.random()
                if record[2] == 1 and ran < positive_selection_p:
                    '''positive'''
                    ids.append(key)
                    continue
                elif record[2] == 0 and ran < negative_selection_p:
                    '''negative'''
                    ids.append(key)
                    continue

        return ids

    def suction_grasp_sampling(self,size,balance_indicator,data_pool=None):
        shuffled_keys = list(self.dict.keys())
        random.shuffle(shuffled_keys)
        positive_selection_p=sampling_p2(x=1,balance_indicator=balance_indicator,probability_exponent=10)
        negative_selection_p=sampling_p2(x=0,balance_indicator=balance_indicator,probability_exponent=10)
        ids = []
        for key in shuffled_keys:
            if data_pool is not None:
                if not data_pool.label.exist(key):
                    print(f'Missing label: {key}')
                    continue
                if not data_pool.depth.exist(key):
                    print(f'Missing depth: {key}')
                    continue
            record = self.dict[key]
            if len(ids) == size: break

            if record[4]==1 and record[5]==0:
                '''suction grasp'''
                if record[6] == 1 and np.random.random() < positive_selection_p:
                    '''positive'''
                    ids.append(key)
                    continue
                elif record[6] == 0 and np.random.random() < negative_selection_p:

                    '''negative'''
                    ids.append(key)
                    continue

        return ids

    def selective_grasp_sampling(self, size,sampling_rates=None):
        shuffled_keys=list(self.dict.keys())
        random.shuffle(shuffled_keys)
        sampling_probabilities=[sampling_p(sampling_rates[i]) for i in range(4)]
        '''normalize probability'''
        sum_=sum(sampling_probabilities)
        sampling_probabilities = [sampling_probabilities[i]/sum_ for i in range(4)]
        print(sampling_probabilities)

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
    # online_data.main_modality=online_data.depth
    file_ids=online_data.get_indexes()
    print('Total number of files = ',len(online_data))
    old_action_data_tracker_path = r'old_online_data_dict.pkl'
    data_tracker=DataTracker2(name=old_action_data_tracker_path, list_size=10)
    c=0
    s_g=0
    f_g=0
    s_s=0
    f_s=0
    n=0
    for file_id in file_ids:
        c+=1
        # print(c)

        label = online_data.label.load_as_numpy(file_id)
        label_obj = LabelObj(label=label)
        data_tracker.old_push(label_obj, file_id)
        v=data_tracker.get_value( file_id)
        print(v)
        if v[0]==1:
            if v[2]==1:s_g+=1
            elif v[2]==0:f_g+=1
            else:
                print('-------------------------------------')
                n+=1
        elif v[4]==1:
            if v[6]==1:s_s+=1
            elif v[6]==0:f_s+=1
            else:
                print('-------------------------------------')
                n+=1
        else:
            print('-------------------------------------')

            n+=1
    print(s_g)
    print(f_g)
    print(s_s)
    print(f_s)
    print(n)

    data_tracker.save()





