from torch.utils import data
from label_unpack import  LabelObj2
import numpy as np
from training.ppo_memory import PPOMemory

class SeizePolicyDataset(data.Dataset):
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
               suction_score,normal,used_gripper,used_suction,target_index

    def __len__(self):
        return len(self.files_indexes)

class ClearPolicyDataset(data.Dataset):
    def __init__(self, data_pool,policy_buffer:PPOMemory,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.buffer=policy_buffer
        self.file_ids=file_ids

    def __getitem__(self, idx):
        buffer_index=self.file_ids[idx]

        target_index = self.buffer.episodic_file_ids[buffer_index]
        value=self.buffer.values[buffer_index]
        advantage=self.buffer.advantages[buffer_index]
        action_index=self.buffer.action_indexes[buffer_index]
        point_index=self.buffer.point_indexes[buffer_index]
        prob=self.buffer.probs[buffer_index]
        reward=self.buffer.rewards[buffer_index]
        end_of_episode=self.buffer.is_end_of_episode[buffer_index]

        rgb = self.data_pool.rgb.load_as_image(target_index)
        mask=self.data_pool.mask.load_as_numpy(target_index)
        depth=self.data_pool.depth.load_as_numpy(target_index)


        return (rgb,depth[np.newaxis,:,:],mask[np.newaxis,:,:],
                value,advantage,action_index,point_index,prob,
                reward,end_of_episode)


    def __len__(self):
        return self.buffer.last_ending_index+1
