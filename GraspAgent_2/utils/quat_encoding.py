import torch
import torch.nn.functional as F

def sign_invariant_quat_encoding_2d(q):
    """
    Sign-invariant encoding for quaternion tensors.

    Args:
        q: torch.Tensor of shape [B, 4, W, H]
           Each quaternion is assumed to be (w, x, y, z).

    Returns:
        encoded: torch.Tensor of shape [B, 10, W, H]
                 Rotation-invariant (sign-invariant) feature map.
    """
    # Normalize quaternions to unit length
    q = F.normalize(q, dim=1)

    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Compute quadratic terms (sign-invariant)
    encoded = torch.stack([
        qw * qw,  # 1
        qx * qx,  # 2
        qy * qy,  # 3
        qz * qz,  # 4
        2 * qw * qx,  # 5
        2 * qw * qy,  # 6
        2 * qw * qz,  # 7
        2 * qx * qy,  # 8
        2 * qy * qz,  # 9
        2 * qz * qx,  # 10
    ], dim=1)

    return encoded