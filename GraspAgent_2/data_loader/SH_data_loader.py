from torch.utils import data


class SHDataset(data.Dataset):
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