import numpy as np
import torch
import torch.nn.functional as F
from GraspAgent_2.hands_config.sh_config import fingers_range, fingers_min, fingers_max
from GraspAgent_2.utils.quat_operations import bulk_quat_mul, signed_cosine_distance


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

def pose_interpolation( gripper_pose, objects_mask,annealing_factor):
    assert objects_mask.sum() > 0

    ref_pose = gripper_pose.detach().clone()

    ref_pose[:, 5:] = torch.clamp(ref_pose[:,5:], 0.01, 0.99)

    annealing_factor = torch.clip(annealing_factor,0.01,0.99)



    sampled_pose = generate_random_beta_dist_widh(ref_pose[:,  0].numel()).reshape(480,-1,7).permute(2,0,1)[None,...]

    sampling_ratios = 1/(1+((1-annealing_factor)*torch.rand_like(ref_pose)) /(annealing_factor*torch.rand_like(ref_pose)))
    # sampling_ratios2 = 1/(1+((1-p2)*torch.rand_like(ref_pose[:,5:])) /(p2*torch.rand_like(ref_pose[:,5:])))
    # sampling_ratios=torch.cat([sampling_ratios1,sampling_ratios2],dim=1)

    sampled_pose = sampled_pose.detach().clone() * sampling_ratios + (1 - sampling_ratios) * ref_pose
    assert not torch.isnan(sampled_pose).any(), f'{sampled_pose}, {sampled_pose.min()}, {ref_pose.min()}, {sampled_pose.max()}, {ref_pose.max()}'

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

def sample_sh_quat(size):
    ref_quat = torch.tensor([[0., 1., 0., 0.]],device='cuda')

    beta_quat=torch.zeros((size,4),device='cuda')
    beta_quat[:,[0,3]]=torch.randn((size, 2), device='cuda')
    beta_quat = F.normalize(beta_quat, dim=-1)

    approach=torch.rand((size, 3), device='cuda')-0.5
    approach[:,-1]=torch.abs(approach[:,-1]*2)
    approach[:, :2]*=0.5
    approach = F.normalize(approach, dim=-1)

    approach_quat=quat_between_batch(torch.tensor([0.0, 0.0, 1.0],device='cuda'),approach)
    approach_quat = F.normalize(approach_quat, dim=-1)

    quat=bulk_quat_mul(beta_quat,ref_quat)

    quat=bulk_quat_mul(approach_quat,quat)
    quat = F.normalize(quat, dim=-1)

    return quat

def sample_ch_quat(size):
    ref_quat = torch.tensor([[1., 1., 0., 0.]],device='cuda')

    beta_quat=torch.zeros((size,4),device='cuda')
    beta_quat[:,[0,3]]=torch.randn((size, 2), device='cuda')
    beta_quat = F.normalize(beta_quat, dim=-1)

    approach=(torch.rand((size, 3), device='cuda'))

    approach[:,[0,2]]=2*(approach[:,[0,2]]-0.5)
    U=approach[:,1]
    k=2
    approach[:, 1] = (1 + torch.sign(2*U - 1) * torch.abs(2*U - 1) ** (1 / (k + 1))) / 2 # this is the CDF inversion of the Probability function defined as (x/0.5-1)^k
    approach = F.normalize(approach, dim=-1)

    approach_quat=quat_between_batch(torch.tensor([0.0, 1.0, 0.0],device='cuda'),approach)
    approach_quat = F.normalize(approach_quat, dim=-1)

    quat=bulk_quat_mul(beta_quat,ref_quat)

    quat=bulk_quat_mul(approach_quat,quat)
    quat = F.normalize(quat, dim=-1)

    return quat


def sample_mixed_tensors(base, n, device='cuda', smooth=False, noise_rate=0.0):
    """
    Sample a tensor of shape [n, 4] by mixing given base vectors equally,
    with optional smooth interpolation and controllable random noise.

    Args:
        base_list (list[list[float]]): list of lists, each inner list of size 4
        n (int): number of samples
        device (str): 'cuda' or 'cpu'
        smooth (bool): if True, interpolate between random pairs of base vectors
        noise_rate (float): standard deviation of added Gaussian noise (0 = none)

    Returns:
        torch.Tensor: [n, 4] sampled tensor
    """
    # Convert list of lists to tensor
    # base = torch.tensor(base_list, dtype=torch.float32, device=device)  # [m, 4]
    m = base.shape[0]

    if not smooth:
        # Discrete selection with equal probability
        idx = torch.randint(0, m, (n,), device=device)
        samples = base[idx]
    else:
        # Smooth interpolation between two random base vectors
        idx1 = torch.randint(0, m, (n,), device=device)
        idx2 = torch.randint(0, m, (n,), device=device)
        alpha = torch.rand(n, 1, device=device)
        samples = alpha * base[idx1] + (1 - alpha) * base[idx2]

    # Add controllable Gaussian noise
    if noise_rate > 0:
        noise = torch.randn_like(samples) * noise_rate
        samples = samples + noise

    samples = F.normalize(samples, dim=-1)

    return samples


def sample_from_two(A, B, ratio_of_A):
    """
    Select rows from A or B based on probability p.

    Args:
        A (torch.Tensor): tensor of shape [n, 4]
        B (torch.Tensor): tensor of shape [n, 4]
        p (float or torch.Tensor): probability of selecting from A
                                   (if tensor, must be shape [n] or [n,1])

    Returns:
        torch.Tensor: selected tensor of shape [n, 4]
    """
    n = A.shape[0]

    # Convert scalar p to tensor if needed
    if not torch.is_tensor(ratio_of_A):
        p = torch.full((n,), ratio_of_A, device=A.device)
    if p.dim() == 1:
        p = p.unsqueeze(1)  # [n,1]

    # Random uniform draw per sample
    rand = torch.rand(n, 1, device=A.device)

    # Boolean mask where to pick from A
    mask = (rand < p).float()  # [n,1]

    # Select entire rows from A or B
    result = mask * A + (1 - mask) * B
    return result

def sample_closets_quat(base_quat,quat_centers,noise_rate=0.):
    dots = torch.abs(torch.matmul(base_quat, quat_centers.t()))
    closest_idx = torch.argmax(dots, dim=1)
    q_assigned = quat_centers[closest_idx]

    random_quat=torch.randn_like(q_assigned)
    # if noise_rate > 0:
    samples = q_assigned*(1-noise_rate) +   random_quat* noise_rate

    samples = F.normalize(samples, dim=-1)

    return samples

def nearest_replace(x: torch.Tensor, y: torch.Tensor,noise_rate=0.) -> torch.Tensor:
    """
    x : [N, 3]  – query points
    y : [M, 3]  – code-book
    return : [N, 3] – for every row in x pick the closest row in y (L2)
    """
    # pairwise squared distances: [N, M]
    dist = (x.unsqueeze(1) - torch.clip(y,0,1).unsqueeze(0)).pow(2).sum(dim=2)
    # index of nearest neighbour in y for every x
    idx = dist.argmin(dim=1)          # [N]

    result=y[idx]


    result = result *(1-noise_rate) + noise_rate*(torch.randn_like(result)+0.5)

    return result


def generate_random_CH_poses(ref_pose,sampling_centroid,noise_ratios,quat_centers=None,finger_centers=None):
    size=ref_pose[:,  0].numel()


    base_quat=ref_pose[0,0:4].reshape(4,-1).T
    base_fingers=ref_pose[0,4:4+3].reshape(3,-1).T
    base_fingers=torch.clip(base_fingers,0,1)
    base_transition=ref_pose[0,4+3:].reshape(1,-1).T

    # quat=torch.randn(size=(size,4),device='cuda')
    #

    if quat_centers is not None:
        # sampled_taxonomies=sample_mixed_tensors(quat_centers,size,smooth=False,noise_rate=0.1)
        quat=sample_closets_quat(base_quat,quat_centers,noise_rate=noise_ratios[:,0:4])
        # quat=sample_from_two(A=quat, B=sampled_taxonomies, ratio_of_A=0.1)
    else:
        quat = sample_ch_quat(size)

    quat = F.normalize(quat, dim=-1)

    # fingers=beta_peak_intensity_tensor(n=size,c=3,data_range=[0,1],centers=torch.tensor([0.5,0.5,0.5],device='cuda'))
    # fingers=(torch.randn((size,3),device='cuda')+0.5)
    if finger_centers is not None:
        fingers=nearest_replace(base_fingers, finger_centers, noise_rate = noise_ratios[:,4:4+3])
        # sampled_fingers=sample_mixed_tensors(finger_centers,size,smooth=False,noise_rate=0.1)
        # fingers=sample_from_two(A=fingers, B=sampled_fingers, ratio_of_A=0.3)
    else:
        fingers = (torch.randn((size, 3), device='cuda') + 0.5)

    # transition=beta_peak_intensity_tensor(n=size,c=1,data_range=[-1,1],centers=sampling_centroid[-1:],peak_intensity=5)
    transition=torch.randn((size,1),device='cuda')*noise_ratios[:,4+3:]+base_transition*(1-noise_ratios[:,4+3:])
    sampled_pose = torch.cat([quat, fingers, transition], dim=1)

    return sampled_pose

def ch_pose_interpolation( gripper_pose,sampling_centroid, annealing_factor,quat_centers=None,finger_centers=None):

    ref_pose = gripper_pose.detach().clone()

    assert ref_pose.shape[0]==1

    sampling_ratios = torch.clip(annealing_factor,0.0,1.0)
    sampling_ratios[sampling_ratios>0.95]=1.
    sampling_ratios1 = 1/(1+((1-sampling_ratios)*torch.rand_like(ref_pose)) /(sampling_ratios*torch.rand_like(ref_pose)+1e-5))[0].permute(1,2,0).view(-1,8)
    sampling_ratios2 = 1/(1+((1-sampling_ratios)*torch.rand_like(sampling_ratios)) /(sampling_ratios*torch.rand_like(sampling_ratios)+1e-5))

    sampled_pose = generate_random_CH_poses(ref_pose,sampling_centroid,noise_ratios=sampling_ratios1,quat_centers=quat_centers,finger_centers=finger_centers).reshape(600,600,8).permute(2,0,1)[None,...]

    '''process quat sign'''
    dist1=signed_cosine_distance(sampled_pose[:,0:4],ref_pose[:,0:4])
    dist2=signed_cosine_distance(-sampled_pose[:,0:4],ref_pose[:,0:4])
    f=torch.ones_like(ref_pose[:,0:1])
    f[dist2<dist1]*=-1
    sampled_pose[:,0:4]*=f


    sampled_pose = sampled_pose * sampling_ratios2 + (1 - sampling_ratios2) * ref_pose
    assert not torch.isnan(sampled_pose).any(), f'{sampled_pose}, {sampling_ratios.min()}, {sampled_pose.max()}'

    sampled_pose[:, :4] = F.normalize(sampled_pose[:, :4], dim=1)

    # '''clip fingers to scope'''
    # sampled_pose[:,4:4+3]=torch.clamp(sampled_pose[:,4:4+3],min=0.01,max=0.99)


    return sampled_pose

def generate_random_SH_poses(size,sampling_centroid):

    quat = sample_sh_quat(size)


    fingers_beta=beta_peak_intensity_tensor(n=size,c=3,data_range=[-1,1],centers=torch.tensor([0,0,0],device='cuda'))

    fingers_main_joints=beta_peak_intensity_tensor(n=size,c=3,data_range=[-1,1],centers=torch.tensor([-1,-1,-1],device='cuda'))
    fingers_tip_joints=beta_peak_intensity_tensor(n=size,c=3,data_range=[-1,1],centers=torch.tensor([-1,-1,-1],device='cuda'))


    transition=beta_peak_intensity_tensor(n=size,c=1,data_range=[sampling_centroid[-1].item()-1,sampling_centroid[-1].item()+1],centers=torch.tensor([0],device='cuda'))

    sampled_pose = torch.cat([quat, fingers_beta,fingers_main_joints,fingers_tip_joints, transition], dim=1)



    return sampled_pose

def sh_pose_interpolation( gripper_pose,sampling_centroid, annealing_factor=0.99):

    ref_pose = gripper_pose.detach().clone()

    p = max(min(annealing_factor,1.),0.0)

    print(f'p= {p}')

    sampled_pose = generate_random_SH_poses(ref_pose[:,  0].numel(),sampling_centroid).reshape(600,600,14).permute(2,0,1)[None,...]

    sampling_ratios = 1/(1+((1-p)*torch.rand_like(ref_pose)) /(p*torch.rand_like(ref_pose)+1e-5))

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

    annealing_factor = max(min(annealing_factor,1.),0.0)
    p1=annealing_factor#**2
    p2=annealing_factor**0.5


    print(f'p1= {p1}')

    sampled_pose = generate_random_beta_dist_widh(ref_pose[:,  0].numel())

    sampling_ratios1 = 1/(1+((1-p1)*torch.rand_like(ref_pose[:,0:5])) /(p1*torch.rand_like(ref_pose[:,0:5])+1e-5))
    sampling_ratios2 = 1/(1+((1-p2)*torch.rand_like(ref_pose[:,5:])) /(p2*torch.rand_like(ref_pose[:,5:])+1e-5))
    sampling_ratios=torch.cat([sampling_ratios1,sampling_ratios2],dim=1)

    sampled_pose = sampled_pose.detach().clone() * sampling_ratios + (1 - sampling_ratios) * ref_pose
    assert not torch.isnan(sampled_pose).any(), f'{sampled_pose}, {sampling_ratios1.min()}, {sampled_pose.min()}, {ref_pose.min()}, {sampling_ratios1.max()}, {sampled_pose.max()}, {ref_pose.max()}, {p1}'

    sampled_pose[:, 3:5] = F.normalize(sampled_pose[:, 3:5], dim=1)
    sampled_pose[:, 0:3] = F.normalize(sampled_pose[:, 0:3], dim=1)

    sampled_pose[:, 5:] = torch.clamp(sampled_pose[:, 5:], 0.01, 0.99)

    return sampled_pose[None,...]