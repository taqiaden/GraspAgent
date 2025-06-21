from torch.utils import data
from label_unpack import LabelObj2, LabelObj
import numpy as np

class DemonstrationsDataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        file_id = self.files_indexes[idx]
        label = self.data_pool.label.load_as_numpy(file_id)
        rgb = self.data_pool.rgb.load_as_image(file_id)
        depth=self.data_pool.depth.load_as_numpy(file_id)
        label_=[x if (x is not None) and ( ~np.isnan(x)) else -1 for x in label]
        if len(label_)<10:label_+=[-1]*(10-len(label_))
        label_=np.array(label_)
        return rgb,depth[np.newaxis,:,:],label_,file_id

    def __len__(self):
        return len(self.files_indexes)

class SeizePolicyDataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        file_id = self.files_indexes[idx]
        label = self.data_pool.label.load_as_numpy(file_id)
        rgb = self.data_pool.rgb.load_as_image(file_id)
        depth=self.data_pool.depth.load_as_numpy(file_id)

        label_obj = LabelObj2(label=label)

        gripper_pixel_index = label_obj.gripper.pixel_index()
        suction_pixel_index = label_obj.suction.pixel_index()

        pose_7=label_obj.gripper.pose_7

        gripper_score=label_obj.gripper.result
        suction_score=label_obj.suction.result

        used_gripper=label_obj.gripper.used
        used_suction=label_obj.suction.used

        normal= label_obj.suction.normal

        return rgb,depth[np.newaxis,:,:],pose_7,gripper_pixel_index,\
               suction_pixel_index,gripper_score,\
               suction_score,normal,used_gripper,used_suction,file_id

    def __len__(self):
        return len(self.files_indexes)


class GraspDataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        file_id = self.files_indexes[idx]
        label = self.data_pool.label.load_as_numpy(file_id)
        depth=self.data_pool.depth.load_as_numpy(file_id)

        label_obj = LabelObj(label=label)

        pixel_index = label_obj.get_pixel_index_()

        pose_7=label_obj.get_gripper_pose_7()

        score=label_obj.success

        used_gripper=label_obj.is_gripper
        used_suction=label_obj.is_suction

        normal= label_obj.normal

        return depth[np.newaxis,:,:],pose_7,pixel_index,\
               score,normal,used_gripper,used_suction,file_id

    def __len__(self):
        return len(self.files_indexes)
