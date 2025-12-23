import torch



def crop_cube(points, center, cube_size):
    half_size = cube_size / 2
    lower = center - half_size
    upper = center + half_size
    mask = (points >= lower) & (points <= upper)
    mask = mask.all(dim=1)
    return points[mask]


def crop_sphere_torch(points, center, radius):
    """
    points: [N, 3] or [B, N, 3]
    center: [3] or [B, 1, 3]
    radius: float
    returns: cropped points
    """
    diff = points - center
    dist2 = (diff * diff).sum(dim=-1)
    mask = dist2 <= radius * radius
    return points[mask]

def voxelize_2d_avg_height(points,  cube_size, grid_size,center=None):
    """
    Convert point cloud to 2D occupancy grid with average height per cell.

    points: [N,3] tensor
    center: [3] tensor
    cube_size: scalar, length of cube side
    grid_size: int, number of pixels along x and y
    Returns:
        avg_height_grid: [grid_size, grid_size] tensor with average z per cell
    """
    half_size = cube_size / 2
    if center is None:
        center = torch.zeros(3, device=points.device)
    # Crop points inside the cube
    lower = center - half_size
    upper = center + half_size
    mask = (points >= lower) & (points <= upper)
    mask = mask.all(dim=1)
    points = points[mask]
    if points.shape[0] == 0:
        return torch.zeros((grid_size, grid_size), dtype=torch.float32)

    # Normalize x and y to [0,1]
    normalized_xy = (points[:, :2] - (center[:2] - half_size)) / cube_size
    # Convert to grid indices
    indices = (normalized_xy * grid_size).long()
    indices = indices.clamp(0, grid_size - 1)

    x_idx = indices[:, 0]
    y_idx = indices[:, 1]
    z_vals = points[:, 2]

    # Flatten 2D indices for bincount
    flat_idx = x_idx * grid_size + y_idx

    # Sum heights and counts per cell
    height_sum = torch.bincount(flat_idx, weights=z_vals, minlength=grid_size * grid_size)
    counts = torch.bincount(flat_idx, minlength=grid_size * grid_size)

    # Avoid division by zero
    avg_height = height_sum / (counts + 1e-6)

    # Reshape back to 2D grid
    avg_height_grid = avg_height.view(grid_size, grid_size)

    return avg_height_grid
def voxelize(points, cube_size, grid_size, center=None):
    """
    points: [N,3] tensor of cropped points
    center: [3] center of cube
    cube_size: scalar
    grid_size: number of voxels along each axis
    """
    half_size = cube_size / 2
    # Normalize points to [0, 1] inside the cube
    if center is None:
        normalized = (points + half_size) / cube_size
    else:
        normalized = (points - (center - half_size)) / cube_size
    # Convert to voxel indices
    indices = (normalized * (grid_size - 1)).long()
    # Clamp to be safe
    indices = indices.clamp(0, grid_size - 1)

    # Create empty occupancy grid
    grid = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.float32)
    # Fill occupancy
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    return grid
def project_3d_to_2d_avg_height(occupancy_grid):
    """
    occupancy_grid: [n, n, n] tensor
    Returns:
        2D tensor [n,n] with average height along z
    """
    n = occupancy_grid.shape[0]
    # z indices as "height" (cube length = 1)
    z_indices = torch.arange(n, device=occupancy_grid.device).view(1, 1, n)

    # sum of heights
    height_sum = (occupancy_grid * z_indices).sum(dim=2)

    # number of occupied voxels along z
    count = occupancy_grid.sum(dim=2)

    # avoid division by zero
    avg_height = height_sum / (count + 1e-6)

    return avg_height


def occupancy_to_pointcloud(occupancy: torch.Tensor, voxel_size=1.0, origin=(0, 0, 0)):
    """
    Convert occupancy grid [D,H,W] -> point cloud [N,3].

    Args:
        occupancy: [D,H,W] tensor (0 = empty, 1 = occupied).
        voxel_size: float, size of each voxel.
        origin: tuple of (x,y,z), coordinates of grid[0,0,0].

    Returns:
        points: [N,3] tensor of point cloud coordinates.
    """
    assert occupancy.dim() == 3, f"Occupancy grid must be [D,H,W] , {occupancy.dim()}"

    # 1. Get indices of occupied voxels
    idx = torch.nonzero(occupancy > 0, as_tuple=False).float()  # [N,3], each row = (d,h,w)

    # 2. Scale by voxel size
    points = idx * voxel_size

    # 3. Shift by origin
    points += torch.tensor(origin, device=points.device)

    return points