from torch.utils import data
from label_unpack import  LabelObj2
import numpy as np
from training.ppo_memory import PPOMemory


class ClearPolicyDataset(data.Dataset):
    def __init__(self, data_pool,policy_buffer:PPOMemory,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.buffer=policy_buffer
        self.file_ids=file_ids

    def __getitem__(self, idx):
        file_id=self.file_ids[idx]

        target_index = self.buffer.episodic_file_ids[file_id]
        value=self.buffer.values[file_id]
        advantage=self.buffer.advantages[file_id]
        action_index=self.buffer.action_indexes[file_id]
        point_index=self.buffer.point_indexes[file_id]
        prob=self.buffer.probs[file_id]
        reward=self.buffer.rewards[file_id]
        end_of_episode=self.buffer.is_end_of_episode[file_id]

        rgb = self.data_pool.rgb.load_as_image(target_index)
        mask=self.data_pool.mask.load_as_numpy(target_index)
        depth=self.data_pool.depth.load_as_numpy(target_index)


        return (rgb,depth[np.newaxis,:,:],mask[np.newaxis,:,:],
                value,advantage,action_index,point_index,prob,
                reward,end_of_episode,file_id)


    def __len__(self):
        return self.buffer.last_ending_index+1
