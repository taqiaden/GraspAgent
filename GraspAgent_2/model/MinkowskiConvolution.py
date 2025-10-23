import torch
import MinkowskiEngine as ME

def occgrid_to_sparse(coords_bool: torch.BoolTensor, voxel_origin=(0,0,0), voxel_size=1.0, batch_idx: int = 0):
    """
    Convert a boolean occupancy grid [X, Y, Z] (torch.BoolTensor) to ME coordinates & features.
    coords_bool[x,y,z] == True means occupied.

    Returns:
        coords: LongTensor [N, 4] with (batch_idx, x_idx, y_idx, z_idx)
        feats: FloatTensor [N, C] (we'll use C=1 occupancy)
    """
    device = coords_bool.device
    occ_idxs = coords_bool.nonzero(as_tuple=False)  # [N, 3] (x, y, z)
    if occ_idxs.numel() == 0:
        # Return empty tensors
        return torch.empty((0,4), dtype=torch.int32, device=device), torch.empty((0,1), device=device)
    # coords for MinkowskiEngine require int32 (batch, x, y, z)
    batch_col = torch.full((occ_idxs.shape[0], 1), batch_idx, dtype=torch.int32, device=device)
    coords = torch.cat([batch_col, occ_idxs.to(dtype=torch.int32)], dim=1)
    feats = torch.ones((occ_idxs.shape[0], 1), dtype=torch.float32, device=device)  # occupancy feature
    return coords, feats

def points_to_sparse(points: torch.Tensor, voxel_size: float, cube_origin: torch.Tensor, batch_idx: int = 0):
    """
    Convert point cloud [N,3] into voxel coordinates and features for ME.
    - voxel_size: float
    - cube_origin: [3] float coordinate that maps to voxel index [0,0,0]
    """
    # Compute integer voxel indices
    voxel_idx = torch.floor((points - cube_origin) / voxel_size).to(torch.int32)  # [N,3]
    batch_col = torch.full((voxel_idx.shape[0], 1), batch_idx, dtype=torch.int32, device=points.device)
    coords = torch.cat([batch_col, voxel_idx], dim=1)  # [N,4]
    feats = torch.ones((coords.shape[0], 1), dtype=torch.float32, device=points.device)
    # Optionally remove duplicates (same voxel)
    coords_feats = torch.cat([coords, feats], dim=1)
    # Keep unique coordinates by unique rows
    coords_unique, idx = torch.unique(coords, dim=0, return_inverse=False, return_counts=False)
    # Simpler: let ME handle duplicate coordinates (ME supports aggregation), or deduplicate if desired.
    return coords, feats
