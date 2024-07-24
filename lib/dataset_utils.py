import logging
import math
import os
import re
import sys
from datetime import datetime

import numpy as np
import smbclient
import torch
from colorama import Fore

from lib.report_utils import counter_progress
from dataset.load_test_data import random_down_sampling
from lib.IO_utils import load_numpy_from_server, save_numpy_to_server
from Configurations.config import  ip_address
from Configurations import config
from lib.report_utils import wait_indicator

rehearsal_data_dir='dataset/rehearsal_data/'
training_data_dir='dataset/training_data/'
realtime_data_dir='dataset/realtime_dataset/'
online_data_dir=ip_address+r'\taqiaden_hub\online_data//'
online_data_local_dir=r'/media/shenxiaofei/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/taqiaden_hub/online_data/'
def configure_smbclient():
    # initialize smbclient
    smbclient.ClientConfig(username='shenxiaofei', password='774631499')
    # to hide INFO logging messages of smbclient
    if config.hide_smbclient_log:logging.disable(sys.maxsize)
configure_smbclient()

class data():
    def __init__(self,dir,is_local=True,dataset_name=None):
        self.dir=dir
        self.pool_name=dataset_name if dataset_name else 'Dataset'
        self.is_local=is_local
        self.pc_folder='point_cloud/'
        self.label_folder='label/'
        self.pc_dir=dir+self.pc_folder
        self.label_dir=dir+self.label_folder
        self.down_sampled_dir=dir+'down_sampled/'
        self.pc_file_sufix='_pointdata.npy'
        self.label_file_sufix='_label.npy'
        self.os=os if self.is_local else smbclient
        self.load_numpy=self.custom_np_load if self.is_local else load_numpy_from_server
        self.save_numpy=np.save if self.is_local else save_numpy_to_server

    def initialize_directories(self,ordered_dir_list):
        for dir in ordered_dir_list:
            self.make_dir(dir)
    def make_dir(self,dir):
        if not self.os.path.exists(dir):
            self.os.mkdir(dir)
    def activate_category_pool(self, category_id):
        if category_id is not None:
            category_sub_folder = f'Category_{category_id}'
            # self.pc_folder = category_sub_folder + '/' + 'point_cloud/'
            # self.label_folder = category_sub_folder + '/' + 'label/'
            self.dir = self.dir + category_sub_folder + '/'
            self.pc_dir = self.dir + self.pc_folder
            self.label_dir = self.dir + self.label_folder
            self.category_id = category_id
    def custom_np_load(self,file_path):
        return np.load(file_path,allow_pickle=True)
    def check_if_local_dir(self,dir):
        self.is_local=True if os.path.exists(dir) else False
    def get_names(self,dir,subset_indexes=None):
        entire_list=self.os.listdir(dir)
        if subset_indexes is None:
            return entire_list
        else:
            return [entire_list[x] for x in subset_indexes]
    def get_pc_names(self,subset_indexes=None):
        return self.get_names(self.pc_dir,subset_indexes)
    def get_label_names(self):
        return self.get_names(self.label_dir)
    def check_missing_label(self):
        filenames = self.os.listdir(self.pc_dir)
        for filename in filenames:
            label_path = self.label_dir + filename.replace(self.pc_file_sufix, self.label_file_sufix)
            file_exist=self.os.path.exists(label_path)
            if not file_exist:
                print(Fore.RED, f'Label data is missing, expected label path: {label_path}', Fore.RESET)
    def get_index(self,file_name):
        file_name_=os.path.splitext(file_name)[0]
        return re.findall(r'\d+', file_name_)[0]
    def get_indexes(self,names=None):
        file_names=self.get_names(self.dir) if names is None else names
        return [self.get_index(x) for x in file_names]
    def load_numpy(self,file_full_path):
        # check if data and label exist
        assert self.os.path.exists(file_full_path), f' The following file does not exist : {file_full_path}'
        file = self.load_numpy(file_full_path)
        return file
    def load_label_by_index(self,index):
        label_full_path = self.label_dir + index + self.label_file_sufix
        return self.load_numpy(label_full_path)
    def load_pc_by_index(self,index):
        pc_full_path=self.pc_dir+index+self.pc_file_sufix
        return self.load_numpy(pc_full_path)
    def load_label(self,filename):
        label_full_path = self.label_dir + filename
        return self.load_numpy(label_full_path)
    def load_pc(self, filename):

        pc_full_path = self.pc_dir + filename
        return self.load_numpy(pc_full_path)
    def load_pc_with_mean(self,filename):
        point_data=self.load_pc(filename)
        # point_data = refine_point_cloud(point_data)
        # point_data = apply_mask(point_data)
        point_data_choice = random_down_sampling(point_data, config.num_points)
        new_point_data = point_data_choice[:, :3]
        new_point_data = torch.from_numpy(new_point_data).float().unsqueeze(0)
        return new_point_data
    def load_labeled_data(self,pc_filename=None,label_filename=None):
        if pc_filename and (label_filename is None):label_filename = pc_filename.replace(self.pc_file_sufix, self.label_file_sufix)
        if label_filename and (pc_filename is None):pc_filename = label_filename.replace(self.label_file_sufix, self.pc_file_sufix)

        pc=self.load_pc(pc_filename)
        label=self.load_label(label_filename)
        return pc,label
    def save_labeled_data(self,pc, label, name_without_suffix,save_to=None):
        if save_to is None:save_to=self.dir
        self.save_numpy(save_to+self.pc_folder + name_without_suffix + self.pc_file_sufix, pc)
        self.save_numpy(save_to+self.label_folder + name_without_suffix + self.label_file_sufix, label)

    def save_label(self, label, name_without_suffix, save_to=None):
        if save_to is None: save_to = self.dir
        self.save_numpy(save_to + self.label_folder + name_without_suffix + self.label_file_sufix, label)
    def remove_all_files(self,dir):
        for filename in self.os.listdir(dir):
            file_path = self.os.path.join(dir, filename)
            try:
                if self.os.path.isfile(file_path) or self.os.path.islink(file_path):
                    self.os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                return False
        return True
    def remove_all_labeled_data(self):
        self.remove_all_files(self.pc_dir)
        self.remove_all_files(self.label_dir)
    def remove_labeled_data(self,name_without_suffix):
        self.os.remove(self.pc_dir+name_without_suffix+self.pc_file_sufix)
        self.os.remove(self.label_dir+name_without_suffix+self.label_file_sufix)
    def length_of_label_container(self):
        return len(self.os.listdir(self.label_dir))
    def detect_duplication(self,dir):
        file_names=self.get_names(dir)
        counter=0
        view_counter = counter_progress('Number of duplicated files found so far : ', counter)
        while len(file_names)>0:
            file = self.load_numpy(dir + file_names[0])
            del file_names[0]
            new_unique_file_names=[]
            for i in range(len(file_names)):
                file_x = self.load_numpy(dir + file_names[i])
                if file.shape==file_x.shape and (file==file_x).all():
                    counter+=1
                    view_counter.step(counter)
                else:
                    new_unique_file_names.append(file_names[i])
            file_names=new_unique_file_names
        view_counter.end()
    def detect_duplicated_labels(self):
        self.detect_duplication(self.label_dir)
    def detect_duplicated_pc(self):
        self.detect_duplication(self.pc_dir)
    def get_suction_labels(self):
        label_names=self.get_label_names()
        suction_label_names=[]
        counter = 0
        view_counter = counter_progress('Number of suction labels found so far : ', counter)
        for label_name_ in label_names:
            label=self.load_label(label_name_)
            is_suction=label[23]==1
            if is_suction:
                suction_label_names.append(label_name_)
                counter+=1
                view_counter.step(counter)
        view_counter.end()
        return suction_label_names
    def load_random_pc(self):
        load_finish=False
        while load_finish==False:
            try:
                pc_names=self.get_pc_names()
                time_seed = math.floor(datetime.now().timestamp())
                np.random.seed(time_seed)
                id=np.random.randint(0,len(pc_names))
                pc=self.load_pc(pc_names[id])
                load_finish=True
            except Exception as e:
                print(f'Load random pc error: {str(e)}')
        return pc
    def remove_all_suction_samples(self):
        print('***** Search for suction labels and delete them')
        suction_label_names=self.get_suction_labels()
        for label_name in suction_label_names:
            idx=self.get_index(label_name)
            self.remove_labeled_data(idx)

    def summary(self):
        wi=wait_indicator('Data are being analyzed ')
        label_names = self.get_label_names()
        grasp_times,suction_times,success_grasp,success_suction=0,0,0,0
        for filename in label_names:
            wi.step(skip_times=100)
            label = self.load_label(filename)
            if label[23]>0.:
                print(filename)
                print('Score=',label[3])
            grasp_times+=label[4]
            suction_times+=label[23]
            success_grasp+=label[4] * label[3]
            success_suction+=label[23] * label[3]

        wi.end()

        print('Number of grasp samples: ',grasp_times, f' ,among, {success_grasp} are successful attempts')
        print('Number of suction samples: ',suction_times, f' ,among, {success_suction} are successful attempts')
    def labels_len(self):
        return len(self.os.listdir(self.label_dir))
    def __len__(self):
        return len(self.os.listdir(self.pc_dir))

class realtime_data(data):
    def __init__(self):
        super(realtime_data,self).__init__(dir=realtime_data_dir,dataset_name='realtime')

class rehearsal_data(data):
    def __init__(self):
        super(rehearsal_data,self).__init__(dir=rehearsal_data_dir,dataset_name='rehearsal')

class training_data(data):
    def __init__(self):
        super(training_data,self).__init__(dir=training_data_dir,dataset_name='traning')

class online_data(data):
    def __init__(self):
        super(online_data,self).__init__(dir=online_data_dir,is_local=False,dataset_name='online')

class online_data_local(data):
    def __init__(self):
        super(online_data_local,self).__init__(dir=online_data_local_dir,is_local=True,dataset_name='online_local')

# rehearsal_data=rehearsal_data()
# training_data=training_data()
# online_data=online_data()


def export_suction_labels(from_dataset,to_dataset):
    suction_label_names = from_dataset.get_suction_labels()
    for label_name in suction_label_names:
        idx = from_dataset.get_index(label_name)
        pc, label = from_dataset.load_labeled_data(label_filename=label_name)
        to_dataset.save_labeled_data(pc, label, idx)

if __name__ == '__main__':

    realtime_data=realtime_data()
    online_data=online_data_local()
    rehearsal_data=rehearsal_data()

    label_names=online_data.get_label_names()
    COUNTER = 0
    print(len(label_names))
    l=28
    for i in range(len(label_names)):
        label=online_data.load_label(label_names[i])
        assert label.shape[0]==l
        l=label.shape[0]
        print(label.shape)
    exit()

    # pc_names=online_data.get_pc_names()
    # for i in range(len(pc_names)):
    #     pc=online_data.load_pc(pc_names[i])
    #     print(pc.shape)
    # exit()

    label_names=realtime_data.get_label_names()
    COUNTER=0
    for label_name in label_names:
        label=realtime_data.load_label(label_name)
        if label[23]==1:
            COUNTER+=1
            pc,label=realtime_data.load_labeled_data(label_filename=label_name)
            idx=realtime_data.get_index(label_name)
            rehearsal_data.save_labeled_data(pc,label,idx)
            print(label_name)
            print(COUNTER)


    # online_data.activate_category_pool(4)
    # online_data.remove_all_suction_samples()
    # rehearsal_data=rehearsal_data()
    # rehearsal_data.check_missing_label()
    # rehearsal_data.remove_all_suction_samples()
    # x=len(training_data)
    # print(training_data.pc_dir)
    # training_data=data(training_data_dir)