import numpy as np
from torch.utils import data

class GraspGANDataset2(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        file_id = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(file_id)

        return depth[np.newaxis,:,:],file_id

    def __len__(self):
        return len(self.files_indexes)

