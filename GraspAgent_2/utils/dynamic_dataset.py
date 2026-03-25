import os
import pickle
import random
import re
from collections import deque

import numpy as np
from colorama import Fore

where_am_i = os.popen('hostname').read()
where_am_i = re.sub(r"[\n\t\s]*", "", where_am_i)
root_dir=None
if where_am_i=='chaoyun-server': # server
    root_dir = r'/home/taqiaden/'
elif where_am_i=='yumi':
    #yumi edge unit
    pass
elif where_am_i=='yons-MS-7D99':
    # company computer
    pass
else:
    root_dir=r'/media/taqiaden/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/taqiaden_hub/'


class SynthesisedData:
    def __init__(self):
        self.id=None
        self.obj_ids = []
        self.obj_poses = []

        self.grasp_target_points = []
        self.grasp_parameters = []
        self.target_indexes = []
        self.importance=[]
        self.grasped_objects=[]

    def unique_obj_max_scores(self):
        best_indices = {}

        for i, (obj, imp) in enumerate(zip(self.grasped_objects, self.importance)):
            if obj not in best_indices or imp > self.importance[best_indices[obj]]:
                best_indices[obj] = i

        # Keep original order of first occurrences
        indices_to_keep = sorted(best_indices.values())

        importance = [self.importance[i] for i in indices_to_keep]

        return importance

    def filter_best_grasps(self):
        best_indices = {}

        for i, (obj, imp) in enumerate(zip(self.grasped_objects, self.importance)):
            if obj not in best_indices or imp > self.importance[best_indices[obj]]:
                best_indices[obj] = i

        # Keep original order of first occurrences
        indices_to_keep = sorted(best_indices.values())

        importance = [self.importance[i] for i in indices_to_keep]

        self.grasp_target_points = [self.grasp_target_points[i] for i in indices_to_keep]
        self.grasp_parameters = [self.grasp_parameters[i] for i in indices_to_keep]
        self.target_indexes =  [self.target_indexes[i] for i in indices_to_keep]
        self.importance=[self.importance[i] for i in indices_to_keep]
        self.grasped_objects=[self.grasped_objects[i] for i in indices_to_keep]

    def save_npz(self, filename):
        # self.filter_best_grasps()
        """Save as compressed NumPy arrays"""
        # Convert lists to numpy arrays
        np.savez_compressed(
            filename,
            obj_ids=np.array(self.obj_ids),
            obj_poses=np.array(self.obj_poses),
            grasp_target_points=np.array(self.grasp_target_points),
            grasp_parameters=np.array(self.grasp_parameters),
            target_indexes=np.array(self.target_indexes),
            importance=np.array(self.importance),
            grasped_objects=np.array(self.grasped_objects),
        )

    def load_npz(self, filename):
        # self.filter_best_grasps()
        """Load from npz file"""
        data = np.load(filename,allow_pickle=True)
        self.obj_ids = data['obj_ids'].tolist()
        self.obj_poses = data['obj_poses'].tolist()
        self.grasp_target_points = data['grasp_target_points'].tolist()
        self.grasp_parameters = data['grasp_parameters'].tolist()
        self.target_indexes = data['target_indexes'].tolist()
        self.importance = (data['importance']+0.00001).tolist() if 'importance' in data else [0.5]*len(self.target_indexes)
        self.grasped_objects = data['grasped_objects'].tolist() if 'grasped_objects' in data else [None]*len(self.target_indexes)

    @classmethod
    def from_npz(cls, filename):
        """Create instance from npz file"""
        instance = cls()
        instance.load_npz(filename)
        return instance

    def sample_pop(self):
        if len(self.importance)!=len(self.target_indexes): self.importance=[0.5]*len(self.target_indexes)
        # selected_idx = random.randint(0, len(self.target_indexes) - 1)

        # selected_idx = random.choices(range(len(self.importance)), weights=self.importance)[0]
        selected_idx = self.importance.index(max(self.importance))

        target_index=self.target_indexes.pop(selected_idx)
        target_point=self.grasp_target_points.pop(selected_idx)
        grasp_parameters=self.grasp_parameters.pop(selected_idx)
        importance=self.importance.pop(selected_idx)
        grasped_objects=self.grasped_objects.pop(selected_idx)

        return target_index,target_point,grasp_parameters,importance,grasped_objects

    def random_pop(self):
        if len(self.importance)!=len(self.target_indexes): self.importance=[0.5]*len(self.target_indexes)
        selected_idx = random.randint(0, len(self.target_indexes) - 1)

        target_index=self.target_indexes.pop(selected_idx)
        target_point=self.grasp_target_points.pop(selected_idx)
        grasp_parameters=self.grasp_parameters.pop(selected_idx)
        importance=self.importance.pop(selected_idx)
        grasped_objects=self.grasped_objects.pop(selected_idx)

        return target_index,target_point,grasp_parameters,importance,grasped_objects

    def __len__(self):
        return len(self.target_indexes)


class DynamicDataManagement:
    def __init__(self,key,root_dir=root_dir):
        self.key=key
        self.folder_dir=root_dir+key+'//'

        if not os.path.exists(self.folder_dir):
            os.makedirs(self.folder_dir)

        self.last_id=0
        self.get_last_id()

        self.low_quality_samples_tracker=deque()


    def get_last_id(self):
        ids = [
            int(f[:-4])
            for f in os.listdir(self.folder_dir)
            if f.endswith(".npz") and f[:-4].isdigit()
        ]
        if len(ids)>0:
            self.last_id = max(ids)
        else:
            self.last_id = 0

    def save_data_point(self,obj:SynthesisedData):
        if len(self.low_quality_samples_tracker)>0:
            id =self.low_quality_samples_tracker.pop()
            path =self.folder_dir+str(id)+'.npz'
            obj.save_npz(path)
            print(Fore.LIGHTMAGENTA_EX,'Replace data point, path :',path,f' , grasped_objects({obj.grasped_objects}), instances({len(obj.target_indexes)}) #obj={len(obj.obj_ids)}',Fore.RESET)
        else:
            path =self.folder_dir+str(self.last_id+1)+'.npz'
            obj.save_npz(path)
            # with open(path, "wb") as f:
            #     pickle.dump(obj, f)

            self.last_id=self.last_id+1

            print(Fore.LIGHTMAGENTA_EX,'Save new data point, path :',path,f' , grasped_objects({obj.grasped_objects}),  instances({len(obj.target_indexes)}) #obj={len(obj.obj_ids)}',Fore.RESET)

    def update_old_record(self,obj:SynthesisedData):
        id=obj.id
        path = self.folder_dir + str(id) + '.npz'
        obj.save_npz(path)

        print(Fore.LIGHTMAGENTA_EX,'Update dynamic data, path :',path,f' , grasped_objects({obj.grasped_objects}), instances({len(obj.target_indexes)}) #obj={len(obj.obj_ids)}',Fore.RESET)

    def load_random_sample(self):
        for i in range(10):
            id = random.randint(1, self.last_id)
            try:
                data=self.load_data_point(id)
                return data
            except Exception as e:
                print(Fore.RED,f'{str(e)}', Fore.RESET)
                self.low_quality_samples_tracker.append(id)
        return None

    def load_data_point(self,id):
        path =self.folder_dir+str(id)+'.npz'

        loaded_obj = SynthesisedData()
        loaded_obj.load_npz(path)
        loaded_obj.id=id
        # with open(path, "rb") as f:
        #     loaded_obj = pickle.load(f)

        return loaded_obj

    def __len__(self):
        return self.last_id

if __name__ == "__main__":
    dataset=DynamicDataManagement(key='test_dynamic_data')

    data_point=SynthesisedData()

    data_point.obj_ids=5

    dataset.save_data_point(data_point)

    data_point=dataset.load_data_point(1)

    print(data_point.obj_ids)
