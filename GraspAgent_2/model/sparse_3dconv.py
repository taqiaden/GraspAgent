import torch
from torch_sparse import SparseTensor


def occupancy_to_sparse_tensor(occupancy: torch.Tensor, voxel_size=1.0, origin=(0, 0, 0), batch_idx=0):
    """
    Convert [D,H,W] occupancy grid to TorchSparse SparseTensor
    """
    # Get occupied voxel indices
    idx = torch.nonzero(occupancy > 0, as_tuple=False).float()  # [N,3]

    if idx.shape[0] == 0:
        return None

    # voxel coordinates
    coords = (idx * voxel_size + torch.tensor(origin)).int()  # [N,3]

    # Add batch index as first column
    batch_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32)
    coords = torch.cat([batch_col, coords], dim=1)  # [N,4]

    # features: occupancy=1
    feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)

    return SparseTensor(feats=feats, coords=coords)
