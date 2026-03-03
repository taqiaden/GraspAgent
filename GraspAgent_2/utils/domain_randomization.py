import torch
import torch.nn.functional as F

def add_reflective_blob_noise(depth, n_blobs=3, blob_radius=6, outlier_scale=0.02):
    d = depth.clone()
    H, W = d.shape

    for _ in range(n_blobs):
        cy = torch.randint(0, H, (1,),device=depth.device)
        cx = torch.randint(0, W, (1,),device=depth.device)

        # circular mask
        yy, xx = torch.meshgrid(
            torch.arange(H,device=depth.device), torch.arange(W,device=depth.device), indexing="ij"
        )
        mask = ((yy - cy)**2 + (xx - cx)**2) < blob_radius**2

        # outliers = depth shifted strongly up/down
        noise = outlier_scale * torch.randn(1,device=depth.device)
        noise=-1*torch.abs(noise)
        d[mask] = d[mask] + noise - torch.abs(d[mask]) * torch.abs(torch.randn(1,device=depth.device)) * 0.2

    return d

def add_depth_noise(z: torch.Tensor,
                    keep_mask: torch.Tensor | None = None,
                    sigma: float = 0.001,
                    quant: float = 1e-3,
                    max_depth: float = 5.0) -> torch.Tensor:
    """
    z         : float tensor [H, W] (metres)
    keep_mask : optional bool tensor [H, W].  True → quantise + noise, False → untouched
    returns noisy tensor same shape / device
    """
    if keep_mask is None:
        keep_mask = torch.ones_like(z, dtype=torch.bool)

    # quantise only masked pixels
    z = torch.where(keep_mask,z,
                    quant * torch.round(z / quant) )

    # noise only masked pixels
    noise = torch.normal(0., sigma, z.shape, device=z.device)
    z = torch.where(keep_mask,z, z + noise)

    return z.clamp(0., max_depth)

def add_flying_pixels(depth, prob=0.002, magnitude=0.05):
    d = depth.clone()
    mask = torch.rand_like(d,device=depth.device) < prob
    d[mask] += magnitude * torch.randn(mask.sum(),device=depth.device)
    return d

def reflective_dropouts(depth, patch_prob=0.3, patch_size=20):
    d = depth.clone()
    H, W = d.shape

    if torch.rand(1) < patch_prob:
        y = torch.randint(0, H - patch_size, (1,),device=depth.device)
        x = torch.randint(0, W - patch_size, (1,),device=depth.device)
        d[y:y+patch_size, x:x+patch_size] = 0.0  # or torch.nan
    return d
