import numpy as np
import torch


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

    return v_rot