from torch.utils import data
from label_unpack import  LabelObj2
import numpy as np
class GraspQualityDataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        label = self.data_pool.label.load_as_numpy(target_index)
        rgb = self.data_pool.rgb.load_as_image(target_index)
        mask=self.data_pool.mask.load_as_numpy(target_index)
        depth=self.data_pool.depth.load_as_numpy(target_index)

        label_obj = LabelObj2(label=label)

        gripper_pixel_index = label_obj.gripper.pixel_index()
        suction_pixel_index = label_obj.suction.pixel_index()

        pose_7=label_obj.gripper.pose_7

        gripper_score=label_obj.gripper.result
        suction_score=label_obj.suction.result

        used_gripper=label_obj.gripper.used
        used_suction=label_obj.suction.used

        normal= label_obj.suction.normal


        return rgb,depth[np.newaxis,:,:],mask[np.newaxis,:,:],pose_7,gripper_pixel_index,\
               suction_pixel_index,gripper_score,\
               suction_score,normal,used_gripper,used_suction

    def __len__(self):
        return len(self.files_indexes)
