import numpy as np
from torch.utils import data
from label_unpack import LabelObj2


class YPGDataset2(data.Dataset):
    def __init__(self, data_pool,file_ids,downsample_size=50000):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids
        self.downsample_size=downsample_size

    def __getitem__(self, idx):
        file_id = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(file_id)

        return depth,file_id

    def __len__(self):
        return len(self.files_indexes)

class YPGDataset3(data.Dataset):
    def __init__(self, data_pool,file_ids,downsample_size=50000):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids
        self.downsample_size=downsample_size

    def __getitem__(self, idx):
        file_id = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(file_id)
        label_obj=LabelObj2()
        pc=label_obj.get_point_clouds_from_depth(depth)

        return pc,file_id

    def __len__(self):
        return len(self.files_indexes)