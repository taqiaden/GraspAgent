import numpy as np
from torch.utils import data
from label_unpack import LabelObj
from lib.image_utils import depth_to_gray_scale


class PolicyDataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        label = self.data_pool.label.load_as_numpy(target_index)
        label_obj = LabelObj(label=label, depth=depth)

        pixel_index = label_obj.get_pixel_index()
        pose_7=label_obj.get_gripper_pose_7()

        score = label_obj.success
        normal= label_obj.normal

        random_rgb=depth_to_gray_scale(depth[:,:,np.newaxis], view=False, convert_to_three_channels=True, colorize=True)

        return depth[np.newaxis,:,:],pose_7,pixel_index,score,normal,label_obj.is_gripper,random_rgb

    def __len__(self):
        return len(self.files_indexes)
