import logging
import math
import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np
import smbclient

from Configurations.config import ip_address, where_am_i

from lib.report_utils import counter_progress
from lib.IO_utils import load_numpy_from_server, save_numpy_to_server
from Configurations import config
from lib.report_utils import wait_indicator

training_data_dir='dataset/training_data/'

local_online_pools=True

if where_am_i=='chaoyun-server': # server
    online_data_dir = r'/home/taqiaden/online_data/'
    online_data_dir2 = r'/home/taqiaden/online_data2/'

elif where_am_i=='yumi': #edge unit
    online_data_dir=ip_address+r'\taqiaden_hub\online_data//'
    online_data_dir2=ip_address+r'\taqiaden_hub\online_data2//'
    local_online_pools=False

else:
    # online_data_dir=ip_address+r'\taqiaden_hub\online_data//'
    # online_data_dir=r'/home/taqiaden/online_data/'
    online_data_dir=r'/media/taqiaden/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/taqiaden_hub/online_data/'
    online_data_dir2=r'/media/taqiaden/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/taqiaden_hub/online_data2/'


online_data_local_dir=r'/media/taqiaden/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/taqiaden_hub/online_data/'


def configure_smbclient():
    # initialize smbclient
    smbclient.ClientConfig(username='taqiaden', password='774631499')
    # to hide INFO logging messages of smbclient
    if config.hide_smbclient_log:logging.disable(sys.maxsize)
configure_smbclient()

def custom_np_load(file_path):
    return np.load(file_path, allow_pickle=True)
class modality_pool():
    def __init__(self,key_name,parent_dir,extension='npy',is_local=True):
        self.key_name=key_name
        self.is_local=is_local
        self.extension=extension
        self.folder=self.key_name+'/'
        self.dir=parent_dir+self.folder
        self.sufix='_'+self.key_name+'.'+self.extension

        '''set proper local/server functions'''
        self.os=os if self.is_local else smbclient
        self.load_numpy_=custom_np_load if self.is_local else load_numpy_from_server
        self.save_numpy_=np.save if self.is_local else save_numpy_to_server

        '''set directory'''
        if not os.path.exists(self.dir): os.mkdir(self.dir)

    def make_dir(self):
        if not self.os.path.exists(self.dir):
            self.os.mkdir(self.dir)
    def get_names(self):
        return self.os.listdir(self.dir)
    def get_index(self,file_name):
        file_name_=os.path.splitext(file_name)[0]
        return re.findall(r'\d+', file_name_)[0]
    def get_indexes(self):
        file_names=self.get_names()
        return [self.get_index(x) for x in file_names]
    def load_as_numpy(self, idx):
        full_path = self.dir + idx+self.sufix
        return self.load_numpy_(full_path)
    def load_as_image(self, idx):
        full_path = self.dir + idx + self.sufix
        return cv2.imread(full_path)
    def load(self, idx):
        if self.extension=='jpg':
            return self.load_as_image(idx)
        else:
            return self.load_as_numpy(idx)
    def save_as_numpy(self, data, idx):
        self.save_numpy_( self.dir + idx + self.sufix, data)
    def save_as_image(self, data, idx):
        cv2.imwrite(self.dir + idx + self.sufix,data)
    def save(self, data,idx):
        if self.extension=='jpg':
            return self.save_as_image(data,idx)
        else:
            return self.save_as_numpy(data,idx)
    def remove_file(self,idx):
        full_path = self.dir + idx + self.sufix
        try:
            if self.os.path.isfile(full_path) or self.os.path.islink(full_path):
                self.os.unlink(full_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (full_path, e))
            return False
    def remove_all_files(self):
        for filename in self.os.listdir(self.dir):
            file_path = self.os.path.join(self.dir, filename)
            try:
                if self.os.path.isfile(file_path) or self.os.path.islink(file_path):
                    self.os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                return False
        return True
    def exist(self,idx):
        full_path = self.dir + idx + self.sufix
        result=self.os.path.exists(full_path)
        return result
    def __len__(self):
        return len(self.os.listdir(self.dir))

class data_pool():
    def __init__(self,dir,is_local=True,dataset_name=None):
        self.dir=dir
        self.pool_name=dataset_name if dataset_name else 'Dataset'

        '''check if dir is local'''
        if is_local is not None:
            self.is_local=is_local
        elif os.path.exists(dir):
            self.is_local=True
        else:
            self.is_local=False

        '''set local/server functions'''
        self.os=os if self.is_local else smbclient
        self.load_numpy_=custom_np_load if self.is_local else load_numpy_from_server
        self.save_numpy_=np.save if self.is_local else save_numpy_to_server

        '''set directory'''
        if not os.path.exists(self.dir): os.mkdir(self.dir)

        '''define modalities pool'''
        self.point_clouds=modality_pool('point_clouds',self.dir,'npy',self.is_local)
        self.label=modality_pool('label',self.dir,'npy',self.is_local)
        self.rgb=modality_pool('rgb',self.dir,'jpg',self.is_local)
        self.depth=modality_pool('depth',self.dir,'npy',self.is_local)

        '''set main modality'''
        self.main_modality=self.label

    def load_numpy(self,file_full_path):
        # check if data and label exist
        assert self.os.path.exists(file_full_path), f' The following file does not exist : {file_full_path}'
        file = self.load_numpy(file_full_path)
        return file
    def get_index(self,file_name):
        file_name_=os.path.splitext(file_name)[0]
        return re.findall(r'\d+', file_name_)[0]
    def get_indexes(self):
        return self.main_modality.get_indexes()
    def clear(self,wait=False):
        self.point_clouds.remove_all_files()
        self.label.remove_all_files()
        self.rgb.remove_all_files()
        self.depth.remove_all_files()

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
        label_names=self.label.get_names()
        suction_label_names=[]
        counter = 0
        view_counter = counter_progress('Number of suction labels found so far : ', counter)
        for label_name_ in label_names:
            label=self.label.load_as_numpy(label_name_)
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
                pc_names=self.point_clouds.get_names()
                time_seed = math.floor(datetime.now().timestamp())
                np.random.seed(time_seed)
                id=np.random.randint(0,len(pc_names))
                index=self.get_index(pc_names[id])
                pc=self.point_clouds.load_as_numpy(index)
                load_finish=True
            except Exception as e:
                print(f'Load random pc error: {str(e)}')
        return pc
    def remove_all_suction_samples(self):
        print('***** Search for suction labels and delete them')
        suction_label_names=self.get_suction_labels()
        for label_name in suction_label_names:
            idx=self.label.get_index(label_name)
            self.label.remove_file(idx)
    def summary(self):
        wi=wait_indicator('Data are being analyzed ')
        label_names = self.label.get_names()
        grasp_times,suction_times,success_grasp,success_suction=0,0,0,0
        for filename in label_names:
            wi.step(skip_times=100)
            label = self.label.load_numpy(filename)
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
    def __len__(self):
        return len(self.main_modality)

class training_data(data_pool):
    def __init__(self):
        super(training_data,self).__init__(dir=training_data_dir,dataset_name='training')

class online_data(data_pool):
    def __init__(self):
        super(online_data,self).__init__(dir=online_data_dir,is_local=local_online_pools,dataset_name='online')

class online_data2(data_pool):
    def __init__(self):
        super(online_data2,self).__init__(dir=online_data_dir2,is_local=True,dataset_name='online2')

class online_data_local(data_pool):
    def __init__(self):
        super(online_data_local,self).__init__(dir=online_data_local_dir,is_local=True,dataset_name='online_local')


def export_suction_labels(from_dataset,to_dataset):
    suction_label_names = from_dataset.get_suction_labels()
    for label_name in suction_label_names:
        idx = from_dataset.get_index(label_name)
        pc, label = from_dataset.load_labeled_data(label_filename=label_name)
        to_dataset.save_labeled_data(pc, label, idx)

if __name__ == '__main__':
    pass
    # online_data=online_data_local()
    #
    # label_names=online_data.label.get_names()
    # COUNTER = 0
    # print(len(label_names))
    # l=28
    # for i in range(len(label_names)):
    #     label=online_data.label.load_numpy(label_names[i])
    #     assert label.shape[0]==l
    #     l=label.shape[0]
    #     print(label.shape)
