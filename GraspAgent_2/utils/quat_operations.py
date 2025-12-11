import math

import numpy as np
import torch
import torch.nn.functional as F

def batch_quat_mul(q1, q2):
    """
    Multiply two batches of quaternions q1 * q2 element-wise.

    Args:
        q1: Tensor of shape [n, 4], quaternions [w, x, y, z]
        q2: Tensor of shape [n, 4], quaternions [w, x, y, z]

    Returns:
        Tensor of shape [n, 4], resulting quaternions.
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)
def quat_from_z_to_vec_single(v):
    """
    Compute quaternion that rotates [0,0,1] to vector v.
    Input: v -> torch tensor of shape [3]
    Output: q -> torch tensor [4] (w, x, y, z)
    """
    v = v / (v.norm() + 1e-8)
    z = torch.tensor([0.0, 0.0, 1.0], device=v.device)

    # Cross and dot products
    axis = torch.cross(z, v)
    dot = torch.dot(z, v)

    if dot > 0.999999:  # almost same direction
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=v.device)  # identity quaternion
    elif dot < -0.999999:  # opposite direction (180°)
        # Rotate around any perpendicular axis, e.g. x-axis
        return torch.tensor([0.0, 1.0, 0.0, 0.0], device=v.device)

    w = torch.sqrt((1.0 + dot) / 2.0)
    xyz = axis / (torch.norm(axis) + 1e-8) * torch.sqrt((1.0 - dot) / 2.0)
    q = torch.cat([w.view(1), xyz])
    return q / q.norm()
def random_quaternion():
    """
    Generate a random unit quaternion (w, x, y, z).
    """
    u1 = np.random.rand()
    u2 = np.random.rand()
    u3 = np.random.rand()

    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    return q  # (x, y, z, w) convention

def quat_mul(q1, q2):
    # Hamilton product
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_vec(q, v):
    vq = np.array([0.0, *v])
    return quat_mul(quat_mul(q, vq), quat_conj(q))[1:]

def transform_frame(v, u, q):
    v_new = rotate_vec(q, v)
    u_new = rotate_vec(q, u)
    return v_new, u_new

def quat_from_two_frames(v1, u1, v2, u2):
    def nrm(x):
        return x / np.linalg.norm(x)

    # Build orthonormal frame A (source)
    xA = nrm(v1)
    yA = nrm(u1 - np.dot(u1, xA) * xA)
    zA = np.cross(xA, yA)
    A = np.column_stack((xA, yA, zA))

    # Build orthonormal frame B (target)
    xB = nrm(v2)
    yB = nrm(u2 - np.dot(u2, xB) * xB)
    zB = np.cross(xB, yB)
    B = np.column_stack((xB, yB, zB))

    # Rotation from A to B
    R = B @ A.T
    t = np.trace(R)

    # Convert rotation matrix to quaternion (stable)
    q = np.zeros(4)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2,1] - R[1,2]) * s
        q[2] = (R[0,2] - R[2,0]) * s
        q[3] = (R[1,0] - R[0,1]) * s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q[0] = (R[2,1] - R[1,2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q[0] = (R[0,2] - R[2,0]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            q[0] = (R[1,0] - R[0,1]) / s
            q[1] = (R[0,2] + R[2,0]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s

    return q / np.linalg.norm(q)
def quat_between(v_from, v_to):
    v_from = v_from / torch.norm(v_from)
    v_to = v_to / torch.norm(v_to)
    cross = torch.cross(v_from, v_to)
    dot = torch.dot(v_from, v_to)
    w = torch.sqrt((torch.norm(v_from)**2) * (torch.norm(v_to)**2)) + dot
    quat = torch.cat([torch.tensor([w]), cross])
    return quat / torch.norm(quat)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_rotate_vector(q, v):
    """
    Rotate 3D vector v by quaternion q.
    q: [w, x, y, z]
    v: [x, y, z]
    """
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    w, x, y, z = q
    q_vec = np.array([x, y, z])

    # Cross products for quaternion rotation
    t = 2.0 * np.cross(q_vec, v)
    v_rot = v + w * t + np.cross(q_vec, t)

    v_rot = v_rot / np.linalg.norm(v_rot)

    return v_rot

def sign_invariant_quat_encoding_1d(q,normalzie=True):
    """
    Sign-invariant encoding for a batch of quaternion sets.

    Args:
        q: torch.Tensor of shape [B, N, 4]
           Each quaternion is (w, x, y, z).

    Returns:
        encoded: torch.Tensor of shape [B, N, 10]
                 Sign-invariant quadratic features per quaternion.
    """
    # Normalize quaternions along the last dimension
    if normalzie:q = F.normalize(q, dim=-1)

    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute quadratic, sign-invariant terms
    encoded = torch.stack([
        qw * qw,      # 1
        qx * qx,      # 2
        qy * qy,      # 3
        qz * qz,      # 4
        2 * qw * qx,  # 5
        2 * qw * qy,  # 6
        2 * qw * qz,  # 7
        2 * qx * qy,  # 8
        2 * qy * qz,  # 9
        2 * qz * qx,  # 10
    ], dim=-1)  # stack along feature dimension → [B, N, 10]

    return encoded

def bulk_quat_mul(q1, q2):
    """
    Multiply two batches of quaternions q1 * q2 element-wise.

    Args:
        q1: Tensor of shape [n, 4], quaternions [w, x, y, z]
        q2: Tensor of shape [n, 4], quaternions [w, x, y, z]

    Returns:
        Tensor of shape [n, 4], resulting quaternions.
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)

def expmap_to_quat_1d(v):
    theta = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
    half = 0.5 * theta
    w = torch.cos(half)
    imag = v / theta * torch.sin(half)
    q = torch.cat([w, imag], dim=-1)
    # canonicalize w ≥ 0
    sign = torch.where(q[..., :1] >= 0, 1.0, -1.0)
    return q * sign

def expmap_to_quat_map_2d(v):
    """
    v: tensor of shape [B, 3, H, W]
    returns: quaternion map [B, 4, H, W] with w >= 0
    """

    # vector norm over channel dimension (the 3D vector)
    theta = torch.norm(v, dim=1, keepdim=True).clamp_min(1e-8)  # [B,1,H,W]

    half = 0.5 * theta
    w = torch.cos(half)                                          # [B,1,H,W]

    imag = v / theta * torch.sin(half)                           # [B,3,H,W]

    q = torch.cat([w, imag], dim=1)                              # [B,4,H,W]

    # Canonicalize so w >= 0
    sign = torch.where(q[:, :1] >= 0, 1.0, -1.0)                 # [B,1,H,W]
    return q * sign

def sign_invariant_quat_encoding_2d(q,normalzie=True):
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
    if normalzie:q = F.normalize(q, dim=1)

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


def quaternion_angular_distance(q1, q2, eps=1e-7, degrees=False):
    """
    Advanced angular distance calculation with batch support.

    Args:
        q1: Tensor of shape (..., 4) or (N, 4) or (4,)
        q2: Tensor of shape (..., 4) or (N, 4) or (4,)
        eps: Numerical stability epsilon
        degrees: If True, return result in degrees instead of radians

    Returns:
        Tensor of angular distances
    """
    # Ensure both tensors have the same shape

    # Normalize along the last dimension
    q1_norm = F.normalize(q1, dim=-1)
    q2_norm = F.normalize(q2, dim=-1)

    # Compute dot product
    dot = torch.sum(q1_norm * q2_norm, dim=-1)

    # Handle double cover and numerical stability
    dot = torch.clamp(torch.abs(dot), 0.0, 1.0 - eps)

    # Calculate angular distance
    angular_dist = 2 * torch.acos(dot)

    if degrees:
        angular_dist = torch.rad2deg(angular_dist)

    return angular_dist


def quaternion_pairwise_angular_distance(quats, eps=1e-7, degrees=False):
    """
    Compute pairwise quaternion angular distances between all quaternions in a batch.

    Args:
        quats: Tensor of shape [n, 4]
        eps: Small epsilon for numerical stability
        degrees: If True, return distance in degrees

    Returns:
        dist_matrix: Tensor of shape [n, n] with angular distances
    """
    # Normalize quaternions
    quats = F.normalize(quats, dim=-1)  # [n,4]

    # Compute dot products between all pairs: [n,n]
    dot = torch.matmul(quats, quats.T)

    # Handle double cover and numerical stability
    dot = torch.clamp(torch.abs(dot), 0.0, 1.0 - eps)

    # Compute angular distance (in radians)
    dist = 2 * torch.acos(dot)

    if degrees:
        dist = torch.rad2deg(dist)

    return dist


def signed_cosine_distance(q1, q2, eps=1e-8):
    """
    Compute signed cosine distance between two quaternion tensors.

    Args:
        q1, q2: Tensors of shape [B, 4, H, W]
        eps: Small constant to prevent division by zero

    Returns:
        dist: Tensor of shape [B, 1, H, W] — signed cosine distance (0 to 2)
    """
    # Normalize both quaternions
    q1 = q1 / (q1.norm(dim=1, keepdim=True) + eps)
    q2 = q2 / (q2.norm(dim=1, keepdim=True) + eps)

    # Signed dot product (cosine similarity)
    dot = (q1 * q2).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    # Convert to signed cosine distance
    dist = 1.0 - dot  # smaller = more similar (same sign)

    return dist


def combine_quaternions(q1, q2, r1, r2, eps=1e-8):
    """
    SLERP between two unit quaternions of shape [4].
    q1, q2: torch.Tensor with dtype float32/float64
    r1, r2: non-negative Python floats
    Returns: unit quaternion torch.Tensor of shape [4]
    """
    q1 = q1 / q1.norm()          # ensure unit
    q2 = q2 / q2.norm()

    dot = q1.dot(q2).item()
    if dot < 0:                  # shortest path
        q2 = -q2
        dot = -dot

    t = r2 / (r1 + r2 + eps)

    if abs(dot) > 1 - eps:       # fall back to lerp
        out = (1 - t) * q1 + t * q2
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        scale1 = math.sin((1 - t) * theta) / sin_theta
        scale2 = math.sin(t * theta) / sin_theta
        out = scale1 * q1 + scale2 * q2

    return out / out.norm()      # re-normalize


