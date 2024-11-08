import numpy as np
from torch.utils import data
from lib.dataset_utils import  online_data, data_pool


training_buffer_dir='dataset/BS_training_buffer/'

class BSBuffer(data_pool):
    def __init__(self):
        super(BSBuffer,self).__init__(dir=training_buffer_dir,dataset_name='training')
        self.main_modality=self.depth

online_data=online_data()


class BackgroundSegDataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        return depth[np.newaxis,:,:],target_index

    def __len__(self):
        return len(self.files_indexes)
