import numpy as np
from torch.utils import data
from lib.dataset_utils import training_data, online_data

training_data=training_data()
online_data=online_data()

class gripper_sampler_dataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        label = self.data_pool.label.load_as_numpy(target_index)
        return depth[np.newaxis,:,:]

    def __len__(self):
        return len(self.files_indexes)

