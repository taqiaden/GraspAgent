import numpy as np
from torch.utils import data
from explore_arms_scope import scope_data_dir
from lib.dataset_utils import modality_pool

class gripper_scope_dataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.data_pool = modality_pool('gripper_label', parent_dir=scope_data_dir, is_local=True)
        self.files_indexes = self.data_pool.get_indexes()

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]

        label = self.data_pool.load_as_numpy(target_index)

        score=label[0:1]
        transformation=label[1:].reshape(-1, 4)
        transition=transformation[0:3, 3].reshape(-1)
        approach=transformation[0:3,0]
        input=np.concatenate([transition,approach])
        return input,score,target_index

    def __len__(self):
        return len(self.files_indexes)

class suction_scope_dataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.data_pool = modality_pool('suction_label', parent_dir=scope_data_dir, is_local=True)
        self.files_indexes = self.data_pool.get_indexes()

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]

        label = self.data_pool.load_as_numpy(target_index)

        score=label[0:1]
        transformation=label[1:].reshape(-1, 4)

        transition=transformation[0:3, 3].reshape(-1)
        approach=transformation[0:3,0]
        input=np.concatenate([transition,approach])
        return input,score,target_index

    def __len__(self):
        return len(self.files_indexes)