import numpy as np
from torch.utils import data

from label_unpack import LabelObj

training_buffer_dir='dataset/GQ_training_buffer/'


class gripper_quality_dataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        label = self.data_pool.label.load_as_numpy(target_index)
        assert label[4]==1 and label[23]==0,f'{label[4]},   {label[23]}'

        label_obj = LabelObj(label=label, depth=depth)
        score=label_obj.success
        # score = np.asarray(score, dtype=np.float32)
        pose_7=label_obj.get_gripper_pose_7()
        pixel_index=label_obj.get_pixel_index()

        return depth[np.newaxis,:,:],pose_7, score,pixel_index

    def __len__(self):
        return len(self.files_indexes)

