import numpy as np
from torch.utils import data

from label_unpack import LabelObj2, LabelObj


class GraspGANDataset2(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        file_id = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(file_id)
        if self.data_pool.label.exist(file_id):
            label = self.data_pool.label.load_as_numpy(file_id)
            label_obj = LabelObj2(label=label)

            gripper_pixel_index = label_obj.gripper.pixel_index()

            gripper_score=label_obj.gripper.result
            used_gripper=label_obj.gripper.used
            valid_gripper_pose=used_gripper and gripper_score>0.5

            pose_7 = label_obj.gripper.pose_7
        else:
            pose_7=np.array([0.]*7)
            valid_gripper_pose=0
            gripper_pixel_index=np.array([0,0])

        return depth[np.newaxis,:,:],file_id,pose_7,valid_gripper_pose,gripper_pixel_index

    def __len__(self):
        return len(self.files_indexes)

class GraspGANDataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        file_id = self.files_indexes[idx]
        label = self.data_pool.label.load_as_numpy(file_id)
        depth=self.data_pool.depth.load_as_numpy(file_id)

        label_obj = LabelObj(label=label)

        gripper_pixel_index = label_obj.get_pixel_index_()

        pose_7=label_obj.get_gripper_pose_7()

        score=label_obj.success

        used_gripper=label_obj.is_gripper

        valid_gripper_pose=(score==1.0) and used_gripper

        return depth[np.newaxis,:,:],file_id,pose_7,valid_gripper_pose,gripper_pixel_index

    def __len__(self):
        return len(self.files_indexes)


