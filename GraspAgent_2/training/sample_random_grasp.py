import numpy as np
import torch
import torch.nn.functional as F
from GraspAgent_2.hands_config.sh_config import fingers_range, fingers_min, fingers_max


def generate_random_beta_dist_widh( size):
    sampled_approach = (torch.rand((size, 2), device='cuda') - 0.5)  # *1.5
    ones_ = torch.ones_like(sampled_approach[:, 0:1])
    sampled_approach = torch.cat([sampled_approach, ones_], dim=1)

    verticle = torch.zeros((size, 3), device='cuda')
    verticle[:, -1] += 1
    # sampled_approach=verticle
    sampled_approach = sampled_approach * 0.5 + verticle * 0.5
    sampled_approach = F.normalize(sampled_approach, dim=-1)

    sampled_beta = (torch.rand((size, 2), device='cuda') - 0.5) * 2
    sampled_beta = F.normalize(sampled_beta, dim=1)

    sampled_dist = torch.distributions.LogNormal(loc=-1.337, scale=0.791)
    sampled_dist = sampled_dist.sample((size, 1)).cuda()

    sampled_width = torch.distributions.LogNormal(loc=-1.312, scale=0.505)
    sampled_width = 1. - sampled_width.sample((size, 1)).cuda()

    sampled_pose = torch.cat([sampled_approach, sampled_beta, sampled_dist, sampled_width], dim=1)

    return sampled_pose

def pose_interpolation( gripper_pose, objects_mask,annealing_factor=0.99):
    assert objects_mask.sum() > 0

    ref_pose = gripper_pose.detach().clone()

    ref_pose[:, 5:] = torch.clamp(ref_pose[:,5:], 0.01, 0.99)

    annealing_factor = max(min(annealing_factor,0.99),0.01)
    p1=annealing_factor#**2
    p2=annealing_factor**0.5


    print(f'p1= {p1}')

    sampled_pose = generate_random_beta_dist_widh(ref_pose[:,  0].numel()).reshape(480,-1,7).permute(2,0,1)[None,...]

    sampling_ratios1 = 1/(1+((1-p1)*torch.rand_like(ref_pose[:,0:5])) /(p1*torch.rand_like(ref_pose[:,0:5])))
    sampling_ratios2 = 1/(1+((1-p2)*torch.rand_like(ref_pose[:,5:])) /(p2*torch.rand_like(ref_pose[:,5:])))
    sampling_ratios=torch.cat([sampling_ratios1,sampling_ratios2],dim=1)

    sampled_pose = sampled_pose.detach().clone() * sampling_ratios + (1 - sampling_ratios) * ref_pose
    assert not torch.isnan(sampled_pose).any(), f'{sampled_pose}, {sampling_ratios1.min()}, {sampled_pose.min()}, {ref_pose.min()}, {sampling_ratios1.max()}, {sampled_pose.max()}, {ref_pose.max()}, {p1}'

    sampled_pose[:, 3:5] = F.normalize(sampled_pose[:, 3:5], dim=1)
    sampled_pose[:, 0:3] = F.normalize(sampled_pose[:, 0:3], dim=1)

    sampled_pose[:, 5:] = torch.clamp(sampled_pose[:, 5:], 0.01, 0.99)

    return sampled_pose
def beta_peak_intensity_tensor(n, c, centers, data_range, peak_intensity=30.0):
    """
    Generate tensor with controllable peak intensity for each channel

    Args:
        n: number of samples per channel
        c: number of channels
        centers: tensor of size [c] with center for each channel
        data_range: (min, max) values
        peak_intensity:
            - 1.0: uniform distribution
            - 2-5: moderate peak
            - 5-10: strong peak
            - 10+: very sharp peak, minimal tails
    """
    min_val, max_val = data_range

    # print('centers: ',centers)
    # print('data_range: ',data_range)

    # Ensure centers is a tensor of correct shape
    if isinstance(centers, (list, np.ndarray)):
        centers = torch.tensor(centers, dtype=torch.float32,device=centers.device)
    assert centers.shape == (c,), f"Centers must have shape [c], got {centers.shape}"

    # Normalize centers to [0,1] within the range for each channel
    centers_norm = (centers - min_val) / (max_val - min_val)

    # Calculate Beta parameters for each channel
    # Shape: [c] for alpha and beta
    alpha = peak_intensity * centers_norm + 1
    beta = peak_intensity * (1 - centers_norm) + 1

    # Generate Beta distributed samples for each channel
    # We'll generate samples separately for each channel and then combine
    samples_list = []
    for i in range(c):
        # print((alpha[i], beta[i]))
        beta_dist = torch.distributions.Beta(alpha[i], beta[i])
        channel_samples = beta_dist.sample((n,))
        samples_list.append(channel_samples)

    # Stack along channel dimension to get [n, c]
    samples = torch.stack(samples_list, dim=1)

    # Scale to desired range
    tensor_data = samples * (max_val - min_val) + min_val

    return tensor_data


def quat_between_batch(v_from, v_to):
    """
    Compute quaternions to rotate a single vector v_from to each vector in v_to.

    Args:
        v_from: Tensor of shape [3], source vector.
        v_to: Tensor of shape [n, 3], target vectors.

    Returns:
        quats: Tensor of shape [n, 4], quaternions in [w, x, y, z] format.
    """
    # Normalize input vectors
    v_from = v_from / torch.norm(v_from)
    v_to = v_to / torch.norm(v_to, dim=1, keepdim=True)

    # Compute cross product and dot product
    cross = torch.cross(v_from.expand_as(v_to), v_to, dim=1)
    dot = torch.sum(v_to * v_from, dim=1, keepdim=True)

    # Compute quaternion scalar part
    w = torch.sqrt(torch.sum(v_from ** 2) * torch.sum(v_to ** 2, dim=1, keepdim=True)) + dot

    # Combine w and cross
    quat = torch.cat([w, cross], dim=1)

    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=1, keepdim=True)
    return quat


def quat_mul(q1, q2):
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

def sample_quat(size):
    first_quat = torch.tensor([[0., 1., 0., 0.]],device='cuda')

    beta_quat=torch.zeros((size,4),device='cuda')
    beta_quat[:,[0,3]]=torch.randn((size, 2), device='cuda')
    beta_quat = F.normalize(beta_quat, dim=-1)

    approach=torch.rand((size, 3), device='cuda')-0.5
    approach[:,-1]=torch.abs(approach[:,-1]*2)
    approach[:, :2]*=0.5
    approach = F.normalize(approach, dim=-1)

    approach_quat=quat_between_batch(torch.tensor([0.0, 0.0, 1.0],device='cuda'),approach)
    approach_quat = F.normalize(approach_quat, dim=-1)

    quat=quat_mul(beta_quat,first_quat)

    quat=quat_mul(approach_quat,quat)
    quat = F.normalize(quat, dim=-1)

    return quat

def generate_random_SH_poses(size,sampling_centroid):

    quat = sample_quat(size)



    fingers_beta=beta_peak_intensity_tensor(n=size,c=3,data_range=[-1,1],centers=torch.tensor([0,0,0],device='cuda'))

    fingers_main_joints=beta_peak_intensity_tensor(n=size,c=3,data_range=[-1,1],centers=torch.tensor([-1,-1,-1],device='cuda'))
    fingers_tip_joints=beta_peak_intensity_tensor(n=size,c=3,data_range=[-1,1],centers=torch.tensor([-1,-1,-1],device='cuda'))


    transition=beta_peak_intensity_tensor(n=size,c=1,data_range=[sampling_centroid[-1].item()-1,sampling_centroid[-1].item()+1],centers=torch.tensor([0],device='cuda'))

    sampled_pose = torch.cat([quat, fingers_beta,fingers_main_joints,fingers_tip_joints, transition], dim=1)



    return sampled_pose

def sh_pose_interpolation( gripper_pose,sampling_centroid, annealing_factor=0.99):

    ref_pose = gripper_pose.detach().clone()

    p = max(min(annealing_factor,0.99),0.01)

    print(f'p= {p}')

    sampled_pose = generate_random_SH_poses(ref_pose[:,  0].numel(),sampling_centroid).reshape(600,600,14).permute(2,0,1)[None,...]

    sampling_ratios = 1/(1+((1-p)*torch.rand_like(ref_pose)) /(p*torch.rand_like(ref_pose)))

    sampled_pose = sampled_pose * sampling_ratios + (1 - sampling_ratios) * ref_pose
    assert not torch.isnan(sampled_pose).any(), f'{sampled_pose}, {sampling_ratios.min()}, {sampled_pose.max()}'

    sampled_pose[:, :4] = F.normalize(sampled_pose[:, :4], dim=1)

    '''clip fingers to scope'''
    fingers_min_=torch.repeat_interleave(torch.from_numpy(fingers_min[0:3]), 3, dim=0).view(1, -1, 1, 1).cuda()
    fingers_max_=torch.repeat_interleave(torch.from_numpy(fingers_max[0:3]), 3, dim=0).view(1, -1, 1, 1).cuda()

    sampled_pose[:,4:4+9]=torch.clamp(sampled_pose[:,4:4+9],min=fingers_min_+0.01,max=fingers_max_-0.01)

    return sampled_pose

def pose_interpolation1d( gripper_pose, objects_mask,annealing_factor=0.99):
    assert objects_mask.sum() > 0

    ref_pose = gripper_pose.detach().clone()[0]

    ref_pose[:, 5:] = torch.clamp(ref_pose[:,5:], 0.01, 0.99)

    annealing_factor = max(min(annealing_factor,0.99),0.01)
    p1=annealing_factor#**2
    p2=annealing_factor**0.5


    print(f'p1= {p1}')

    sampled_pose = generate_random_beta_dist_widh(ref_pose[:,  0].numel())

    sampling_ratios1 = 1/(1+((1-p1)*torch.rand_like(ref_pose[:,0:5])) /(p1*torch.rand_like(ref_pose[:,0:5])))
    sampling_ratios2 = 1/(1+((1-p2)*torch.rand_like(ref_pose[:,5:])) /(p2*torch.rand_like(ref_pose[:,5:])))
    sampling_ratios=torch.cat([sampling_ratios1,sampling_ratios2],dim=1)

    sampled_pose = sampled_pose.detach().clone() * sampling_ratios + (1 - sampling_ratios) * ref_pose
    assert not torch.isnan(sampled_pose).any(), f'{sampled_pose}, {sampling_ratios1.min()}, {sampled_pose.min()}, {ref_pose.min()}, {sampling_ratios1.max()}, {sampled_pose.max()}, {ref_pose.max()}, {p1}'

    sampled_pose[:, 3:5] = F.normalize(sampled_pose[:, 3:5], dim=1)
    sampled_pose[:, 0:3] = F.normalize(sampled_pose[:, 0:3], dim=1)

    sampled_pose[:, 5:] = torch.clamp(sampled_pose[:, 5:], 0.01, 0.99)

    return sampled_pose[None,...]