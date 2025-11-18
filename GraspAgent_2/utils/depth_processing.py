import torch
import torch.nn.functional as F


def depth_gradients(depth):
    """
    Convert a depth map to relative depth (gradients).

    Args:
        depth: Tensor of shape [B, 1, H, W] (batch, channel, height, width)

    Returns:
        grad: Tensor of shape [B, 2, H, W] with [dx, dy] channels
    """
    # horizontal differences (dx)
    dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))  # pad last column to keep same width

    # vertical differences (dy)
    dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))  # pad last row to keep same height

    # concatenate dx and dy as 2-channel input
    grad = torch.cat([dx, dy], dim=1)
    return grad


def nearest_fill(depth, mask):
    # depth: [H,W], mask: bool (True=valid, False=missing)
    depth_filled = depth.clone()
    missing = ~mask

    # simple approach using convolution to propagate neighbors
    kernel = torch.ones(3,3, device=depth.device)
    kernel[1,1] = 0

    # valid pixels count in neighborhood
    mask_float = mask.float().unsqueeze(0).unsqueeze(0)
    neighbor_count = F.conv2d(mask_float, kernel.unsqueeze(0).unsqueeze(0), padding=1)
    neighbor_sum = F.conv2d((depth).unsqueeze(0).unsqueeze(0)*mask_float, kernel.unsqueeze(0).unsqueeze(0), padding=1)

    # avoid division by zero
    neighbor_count[neighbor_count==0] = 1

    depth_avg = (neighbor_sum / neighbor_count).squeeze()
    depth_filled[missing] = depth_avg[missing]

    return depth_filled

def iterative_fill(depth, mask, max_iter=500):
    filled = depth.clone()
    current_mask = mask.clone()
    for i in range(max_iter):
        filled = nearest_fill(filled, current_mask)
        current_mask = current_mask | (filled != 0)
        if bool(current_mask.all()):break
    return filled

def sobel_gradients(depth):
    # assume [B,1,H,W]
    Kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3).to(depth.device)
    Ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3).to(depth.device)
    Gx = F.conv2d(depth, Kx, padding=1)
    Gy = F.conv2d(depth, Ky, padding=1)
    return Gx, Gy

def masked_sobel_gradients(depth, mask):
    """
    Compute Sobel gradients for depth while ignoring missing pixels.

    Args:
        depth: [B,1,H,W] tensor, depth values
        mask: [B,1,H,W] tensor, bool or 0/1, True=valid

    Returns:
        Gx, Gy: [B,1,H,W] masked gradients (0 where invalid)
    """
    # Sobel kernels
    Kx = torch.tensor([[1,0,-1],
                       [2,0,-2],
                       [1,0,-1]], dtype=depth.dtype, device=depth.device).view(1,1,3,3)
    Ky = torch.tensor([[1,2,1],
                       [0,0,0],
                       [-1,-2,-1]], dtype=depth.dtype, device=depth.device).view(1,1,3,3)

    # Convert mask to float
    mask_f = mask.float()

    # Mask depth before convolution
    depth_masked = depth * mask_f

    # Convolve masked depth
    Gx = F.conv2d(depth_masked, Kx, padding=1)
    Gy = F.conv2d(depth_masked, Ky, padding=1)

    # Count valid neighbors
    neighbor_count = F.conv2d(mask_f, torch.ones_like(Kx), padding=1)

    # Avoid division by zero
    neighbor_count[neighbor_count == 0] = 1.0

    # Normalize gradients
    Gx = Gx / neighbor_count
    Gy = Gy / neighbor_count

    # Zero out gradients where mask is invalid
    Gx = Gx * mask_f
    Gy = Gy * mask_f

    return Gx, Gy