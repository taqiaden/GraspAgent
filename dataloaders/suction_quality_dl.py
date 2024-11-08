import numpy as np
from torch.utils import data
from label_unpack import LabelObj
from lib.dataset_utils import  online_data, data_pool


training_buffer_dir='dataset/SQ_training_buffer/'

class SQBuffer(data_pool):
    def __init__(self):
        super(SQBuffer,self).__init__(dir=training_buffer_dir,dataset_name='training')
        self.main_modality=self.depth

online_data=online_data()

force_balanced_data=True



class suction_quality_dataset(data.Dataset):
    def __init__(self, data_pool,file_ids):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = file_ids

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        label = self.data_pool.label.load_as_numpy(target_index)
        assert label[4] == 0 and label[23] == 1, f'{label[4]},   {label[23]}'
        label_obj = LabelObj(label=label, depth=depth)

        score = label_obj.success
        normal= label_obj.normal
        pixel_index = label_obj.get_pixel_index()

        return depth[np.newaxis,:,:],normal,score,pixel_index

    def __len__(self):
        return len(self.files_indexes)


