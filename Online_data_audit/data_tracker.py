import random
from lib.report_utils import  progress_indicator
from Online_data_audit.dictionary_utils import load_dict, save_dict
from label_unpack import LabelObj
from lib.dataset_utils import online_data
from lib.statistics import moving_momentum
from lib.report_utils import progress_indicator as pi

dictionary_directory=r'Online_data_audit/'
gripper_grasp_tracker=r'gripper_grasp_dict'
suction_grasp_tracker=r'suction_grasp_dict'

# [0] 1 for success
# [1] 1 for test data
# [2] 1 for for collision
gripper_container_size=10
suction_container_size=10

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

def set_gripper_dictionary():
    set_arm_dictionary(gripper_grasp_tracker,gripper=True,list_size=gripper_container_size)

def set_suction_dictionary():
    set_arm_dictionary(suction_grasp_tracker,suction=True,list_size=suction_container_size)

def set_arm_dictionary(name,gripper=False,suction=False,clean_old_records=True,list_size=10):
    indexes = online_data.get_indexes()

    data_tracker = DataTracker(name=name, list_size=list_size)
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
    data_tracker = DataTracker(name=dict_name, list_size=10)

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
    data_tracker = DataTracker(name=dict_name, list_size=list_size)
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
    data_tracker = DataTracker(name=dict_name, list_size=list_size)

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

def split_test_set(tracker_name,list_size=10,test_size_ratio=0.2,index=1):
    '''get number of test samples'''
    positive_labels,negative_labels=sample_all_positive_and_negatives(tracker_name, list_size, shuffle=True)
    minimum_balanced_size=min(len(positive_labels),len(negative_labels))
    test_data_size=int(test_size_ratio*minimum_balanced_size)

    '''split test and train labels'''
    test_labels=positive_labels[0:test_data_size]+negative_labels[0:test_data_size]
    train_labels=positive_labels[test_data_size:]+negative_labels[test_data_size:]


    '''mark as test sample'''
    data_tracker = DataTracker(name=tracker_name, list_size=list_size)
    for i in range(len(test_labels)):
        data_tracker.set_test_sample( test_labels[i], index, data=1)

    '''mark as train sample'''
    for i in range(len(train_labels)):
        data_tracker.set_test_sample( train_labels[i], index, data=0)

    data_tracker.save()

def split_gripper_data():
    split_test_set(gripper_grasp_tracker, list_size=gripper_container_size, test_size_ratio=0.2, index=1)
def split_suction_data():
    split_test_set(suction_grasp_tracker, list_size=suction_container_size, test_size_ratio=0.2, index=1)

def track_collision_state(list_size=10,index=2):
    '''get number of test samples'''
    positive_labels,negative_labels=sample_all_positive_and_negatives(gripper_grasp_tracker, list_size, shuffle=True)

    ids=positive_labels+negative_labels
    pi = progress_indicator('progress ', max_limit=len(ids))

    '''set collision state'''
    data_tracker = DataTracker(name=gripper_grasp_tracker, list_size=list_size)
    counter=0
    for i in range(len(ids)):
        '''check collision'''
        label = online_data.label.load_as_numpy(ids[i])
        label_obj = LabelObj(label=label)
        if label_obj.failure or label_obj.is_suction: continue
        depth = online_data.depth.load_as_numpy(ids[i])
        collision_state = label_obj.check_collision(depth=depth)
        counter+=collision_state
        data_tracker.set_collision_state( ids[i], index, data=collision_state)
        pi.step(i)

    pi.end()
    print(f'Found {counter} samples in collision state out of {len(ids)}')
    data_tracker.save()

if __name__ == '__main__':
    track_collision_state()
    # set_suction_dictionary()
    # set_gripper_dictionary()
    # split_gripper_data()
    # split_suction_data()

    # balanced_list=sample_random_buffer(dict_name=suction_grasp_tracker)

    # print(len(balanced_list))

