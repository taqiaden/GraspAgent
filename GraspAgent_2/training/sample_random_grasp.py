import numpy as np
import torch
import torch.nn.functional as F
from GraspAgent_2.hands_config.sh_config import fingers_range, fingers_min, fingers_max
from GraspAgent_2.utils.Online_clustering import OnlingClustering
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

def pose_interpolation( gripper_pose, objects_mask,annealing_factor,tou=1.):
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

    # max_angle_rad=2*np.pi*tou
    # sampled_pose[:, 0:3] =clip_vectors_angle_batch(sampled_pose[:, 0:3] ,gripper_pose[:, 0:3] ,max_angle_rad/2)
    # sampled_pose[:, 3:5] =clip_vectors_angle_batch_2d(sampled_pose[:, 3:5] ,gripper_pose[:, 3:5] ,max_angle_rad)

    sampled_pose[:, 0:3] = F.normalize(sampled_pose[:, 0:3], dim=1)
    sampled_pose[:, 3:5] = F.normalize(sampled_pose[:, 3:5], dim=1)

    # sampled_pose[:, 5:]=clip_scalars_batch(sampled_pose[:, 5:],gripper_pose[:, 5:],max_dist=tou)

    sampled_pose[:, 5:] = torch.clamp(sampled_pose[:, 5:], 0.01, 0.99)

    return sampled_pose


def clip_scalars_batch(A, B, max_dist):
    """
    Clip each element of A so that the distance to B <= max_dist.
    Operates element-wise independently.

    Args:
        A: Tensor [B, dim, H, W]
        B: Tensor [B, dim, H, W]
        max_dist: float

    Returns:
        A_clipped: Tensor [B, dim, H, W]
    """
    # Element-wise difference
    diff = A - B

    # Clip each element independently
    diff_clipped = diff.clamp(min=-max_dist, max=max_dist)

    # Reconstruct clipped tensor
    A_clipped = B + diff_clipped
    return A_clipped

def clip_vectors_angle_batch_2d(A, B, max_angle_rad):
    """
    Clip batch of 2D vectors A so that angle with B <= max_angle_rad.
    Supports [n, 2] or [B, 2, H, W]
    """
    is_4d = A.ndim == 4
    if is_4d:
        BATCH, VEC_DIM, H, W = A.shape
        A_flat = A.permute(0, 2, 3, 1).reshape(-1, 2)
        B_flat = B.permute(0, 2, 3, 1).reshape(-1, 2)
    else:
        A_flat = A
        B_flat = B

    # Normalize
    A_norm = A_flat / A_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    B_norm = B_flat / B_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Cosine of angles
    cos_theta = (A_norm * B_norm).sum(dim=1).clamp(-1.0, 1.0)
    angles = torch.acos(cos_theta)

    # Mask: need to clip
    outside_mask = angles > max_angle_rad

    A_clipped_flat = A_flat.clone()

    if outside_mask.any():
        A_out = A_norm[outside_mask]
        B_out = B_norm[outside_mask]

        # Compute rotation matrix for 2D
        # Angle to rotate: clipped_angle = max_angle_rad
        # Determine sign of rotation (cross product scalar in 2D)
        cross = B_out[:, 0]*A_out[:, 1] - B_out[:, 1]*A_out[:, 0]
        sign = torch.sign(cross).unsqueeze(1)  # +1 or -1

        max_angle_rad_tensor = torch.tensor(max_angle_rad, device=A.device, dtype=A.dtype)

        cos_clip = torch.cos(max_angle_rad_tensor)
        sin_clip = torch.sin(max_angle_rad_tensor)

        # Rotation matrix application
        x_new = cos_clip * B_out[:, 0] - sign[:,0] * sin_clip * B_out[:, 1]
        y_new = sign[:,0] * sin_clip * B_out[:, 0] + cos_clip * B_out[:, 1]
        v_rot = torch.stack([x_new, y_new], dim=1)

        # Scale to original magnitude
        A_mag = A_flat[outside_mask].norm(dim=1, keepdim=True)
        v_rot = v_rot / v_rot.norm(dim=1, keepdim=True).clamp(min=1e-8) * A_mag

        A_clipped_flat[outside_mask] = v_rot

    # Reshape back if 4D
    if is_4d:
        A_clipped = A_clipped_flat.reshape(BATCH, H, W, 2).permute(0, 3, 1, 2)
    else:
        A_clipped = A_clipped_flat

    return A_clipped
def clip_vectors_angle_batch(A, B, max_angle_rad):
    """
    Clip batch of vectors A so that angle with B <= max_angle_rad.
    Supports shapes [n, vec_dim] or [B, vec_dim, H, W].

    Args:
        A: Tensor of shape [n, vec_dim] or [B, vec_dim, H, W]
        B: Tensor of same shape as A
        max_angle_rad: float, max allowed angle in radians

    Returns:
        A_clipped: same shape as A
    """
    original_shape = A.shape
    vec_dim = A.shape[1] if A.ndim == 2 else A.shape[1]

    # Flatten spatial dimensions if necessary
    if A.ndim == 4:  # [B, vec_dim, H, W]
        BATCH, VEC_DIM, H, W = A.shape
        A_flat = A.permute(0, 2, 3, 1).reshape(-1, VEC_DIM)
        B_flat = B.permute(0, 2, 3, 1).reshape(-1, VEC_DIM)
    else:
        A_flat = A
        B_flat = B

    # Normalize
    A_norm = A_flat / A_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    B_norm = B_flat / B_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Compute angles
    cos_theta = (A_norm * B_norm).sum(dim=1).clamp(-1.0, 1.0)
    angles = torch.acos(cos_theta)

    # Masks
    outside_mask = angles > max_angle_rad

    # Prepare output
    A_clipped_flat = A_flat.clone()

    if outside_mask.any():
        A_out = A_norm[outside_mask]
        B_out = B_norm[outside_mask]

        # Rotation axis (cross product)
        axis = torch.cross(B_out, A_out)
        axis_norm = axis.norm(dim=1, keepdim=True)
        axis = torch.where(axis_norm > 1e-8, axis / axis_norm, torch.zeros_like(axis))

        # Convert max_angle_rad to tensor if needed
        max_angle_rad_tensor = torch.tensor(max_angle_rad, device=A.device, dtype=A.dtype)
        cos_clip = torch.cos(max_angle_rad_tensor)
        sin_clip = torch.sin(max_angle_rad_tensor)

        # Rodrigues formula
        v_rot = B_out * cos_clip + torch.cross(axis, B_out) * sin_clip

        # Scale to original magnitude
        A_mag = A_flat[outside_mask].norm(dim=1, keepdim=True)
        v_rot = v_rot / v_rot.norm(dim=1, keepdim=True).clamp(min=1e-8) * A_mag

        A_clipped_flat[outside_mask] = v_rot

    # Reshape back if necessary
    if A.ndim == 4:
        A_clipped = A_clipped_flat.reshape(BATCH, H, W, VEC_DIM).permute(0, 3, 1, 2)
    else:
        A_clipped = A_clipped_flat

    return A_clipped

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

def rand_unit_quaternion(n=1, device='cuda', dtype=torch.float32):
    """
    Generate n uniformly-distributed *unit* quaternions.
    Output shape: (n, 4)  [w, x, y, z]
    """
    u1, u2, u3 = torch.rand(3, n, device=device, dtype=dtype)
    q = torch.empty(n, 4, device=device, dtype=dtype)
    q[:, 0] = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q[:, 1] = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q[:, 2] = torch.sqrt(u1)     * torch.sin(2 * torch.pi * u3)
    q[:, 3] = torch.sqrt(u1)     * torch.cos(2 * torch.pi * u3)
    return q

def sample_closets_quat(base_quat,quat_centers,noise_rate=0.,noise=None):
    dots = torch.abs(torch.matmul(base_quat, quat_centers.t()))
    closest_idx = torch.argmax(dots, dim=1)
    q_assigned = quat_centers[closest_idx]

    noise=rand_unit_quaternion(n=q_assigned.shape[0]) if noise is None else noise
    noise = F.normalize(noise, dim=-1)

    # if noise_rate > 0:
    samples = q_assigned*(1-noise_rate) +   noise* noise_rate

    samples = F.normalize(samples, dim=-1)

    return samples

def nearest_replace(x: torch.Tensor, y: torch.Tensor,noise_rate=0.,noise=None) -> torch.Tensor:
    """
    x : [N, 3]  – query points
    y : [M, 3]  – code-book
    return : [N, 3] – for every row in x pick the closest row in y (L2)
    """
    # pairwise squared distances: [N, M]
    dist = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=2)
    # index of nearest neighbour in y for every x
    idx = dist.argmin(dim=1)          # [N]

    result=y[idx]

    r=torch.randn_like(result) if noise is None else noise
    result = result *(1-noise_rate) + noise_rate*(r)

    return result

def nearest_replace_cosine(x: torch.Tensor, y: torch.Tensor,
                           noise_rate= 0.,
                           noise= None) -> torch.Tensor:
    """
    x : [N, 3]  – query points
    y : [M, 3]  – code-book
    return : [N, 3] – for every row in x pick the closest row in y (COSINE)
    """
    x_n = F.normalize(x, dim=1)
    y_n = F.normalize(y, dim=1)

    # cosine similarity: [N, M]
    sim = x_n @ y_n.T

    # cosine distance = 1 - similarity
    idx = sim.argmax(dim=1)   # max similarity = min distance

    result=y[idx]

    r=torch.randn_like(result) if noise is None else noise
    result = result *(1-noise_rate) + noise_rate*(r)
    return result
def generate_random_CH_poses(ref_pose,noise_ratios,alpha: OnlingClustering,beta: OnlingClustering,fingers: OnlingClustering,transition: OnlingClustering):
    size=ref_pose[:,  0].numel()

    base_alpha=ref_pose[:,0:3]
    base_beta=ref_pose[:,3:5]
    base_fingers=ref_pose[:,5:5+3]
    base_fingers=torch.clip(base_fingers,0,1)
    base_transition=ref_pose[:,5+3:]

    noise = torch.randn_like(base_alpha)
    noise[:, -1] = -1
    noise = F.normalize(noise, dim=-1)
    if alpha.centers is not None:
        indices = torch.randint(0, alpha.centers.shape[0], (size,))
        alpha_=alpha.centers[indices]*0.9+noise*0.1
        # alpha_=nearest_replace_cosine(base_alpha,alpha.centers,noise_rate=noise_ratios[:,0:3],noise=noise)
    else:
        alpha_ = noise
    alpha_ = F.normalize(alpha_, dim=-1)

    noise = torch.randn_like(base_beta)
    noise = F.normalize(noise, dim=-1)
    if beta.centers is not None:
        indices = torch.randint(0, beta.centers.shape[0], (size,))
        beta_=beta.centers[indices]*0.9+noise*0.1
        # beta_=nearest_replace_cosine(base_beta,beta.centers,noise_rate=noise_ratios[:,3:5],noise=noise)
    else:
        beta_ =noise
    beta_ = F.normalize(beta_, dim=-1)

    noise = (torch.randn((size, 3), device='cuda')) + 0.5
    noise=torch.clip(noise,max=1.0)
    if fingers.centers is not None:
        indices = torch.randint(0, fingers.centers.shape[0], (size,))
        fingers_=fingers.centers[indices]*0.9+noise*0.1
        # fingers_=nearest_replace(base_fingers, fingers.centers, noise_rate = noise_ratios[:,5:5+3],noise=noise)
    else:
        fingers_ = noise
    fingers_=torch.clip(fingers_,max=1.0)

    noise = torch.randn((size, 1), device='cuda') -0.5
    if transition.centers is not None:
        indices = torch.randint(0, transition.centers.shape[0], (size,))
        transition_=transition.centers[indices]*0.9+noise*0.1
        # transition_=nearest_replace(base_transition, transition.centers, noise_rate = noise_ratios[:,5+3:],noise=noise)
    else:
        transition_=noise

    sampled_pose = torch.cat([alpha_,beta_, fingers_, transition_], dim=1)
    return sampled_pose

def fill_tensor_by_rate(X, rates, m):
    """
    X: [n, c_dim]
    rates: [n] (non-negative)
    m: target number of rows
    """

    n, c_dim = X.shape

    # normalize rates
    rates = rates / rates.sum()

    # ideal (fractional) counts
    counts = rates * m

    # integer part
    int_counts = torch.floor(counts).long()

    # fix rounding error
    remainder = m - int_counts.sum()
    if remainder > 0:
        # distribute leftover by largest fractional parts
        frac = counts - int_counts.float()
        _, idx = torch.topk(frac, remainder)
        int_counts[idx] += 1

    # repeat rows
    Y = torch.repeat_interleave(X, int_counts, dim=0)

    return Y  # [m, c_dim]
def generate_random_CH_poses2(ref_pose,noise_ratios,taxonomies: OnlingClustering,alphas: OnlingClustering,betas: OnlingClustering,fingerss: OnlingClustering,transitions: OnlingClustering):
    size=ref_pose[:,  0].numel()

    taxonomies_centers=taxonomies.centers #[n,c_dim]
    if taxonomies_centers is not None:
        taxonomies_centers_rates=taxonomies.update_rates[:,0] #[n]
        buffer_size_ratio=taxonomies_centers.shape[0]/taxonomies.N
    else:
        buffer_size_ratio=0

    ''''random noise'''
    alpha_noise=torch.randn_like(ref_pose[:,0:3])
    alpha_noise[:,-1]=-1
    alpha_noise = F.normalize(alpha_noise, dim=-1)
    beta_noise=torch.randn_like(ref_pose[:,0:2])
    beta_noise = F.normalize(beta_noise, dim=-1)
    fingers_noise = 0.75 + torch.randn_like(ref_pose[:,0:3])
    transition_noise = torch.randn_like(ref_pose[:,0:1]) -0.5
    random_noise=torch.cat([alpha_noise,beta_noise,fingers_noise,transition_noise],dim=1)
    if buffer_size_ratio<0.3:
        sampled_pose=random_noise
    else:
        # pick randomly
        # indices = torch.randint(0, taxonomies_centers.shape[0], (size,))
        # noise=taxonomies_centers[indices]
        # pick based on rate
        noise=fill_tensor_by_rate(X=taxonomies_centers, rates=taxonomies_centers_rates, m=size)*0.9+random_noise*0.1


        '''closest taxonomy ids'''
        clipped_tax=taxonomies_centers.clone()
        clipped_tax[:,5:5+3]=torch.clip(clipped_tax[:,5:5+3],0,1)
        dist = (ref_pose.unsqueeze(1) - clipped_tax.unsqueeze(0)).pow(2).sum(dim=2)
        # index of nearest neighbour in y for every x
        idx = dist.argmin(dim=1)  # [N]
        result = taxonomies_centers[idx]
        sampled_pose = result * (1 - noise_ratios) + noise_ratios * noise



    return sampled_pose

def ch_pose_interpolation( gripper_pose, annealing_factor,taxonomies=None,alpha=None,beta=None,fingers=None,transition=None,tou=1.):

    ref_pose = gripper_pose.detach().clone()

    assert ref_pose.shape[0]==1

    assert not torch.isnan(ref_pose).any(), f'{ref_pose}'

    sampling_ratios = torch.clip(annealing_factor,0.01,0.99)
    sampling_ratios[sampling_ratios>0.95]=1.
    sampling_ratios1 = 1/(1+((1-sampling_ratios)*torch.rand_like(ref_pose)) /(sampling_ratios*torch.rand_like(ref_pose)+1e-5))[0].permute(1,2,0).view(-1,9)
    sampling_ratios2 = 1/(1+((1-sampling_ratios)*torch.rand_like(sampling_ratios)) /(sampling_ratios*torch.rand_like(sampling_ratios)+1e-5))
    '''clip minimum update'''
    sampling_ratios1=torch.clip(sampling_ratios1,min=0.1)
    sampling_ratios2=torch.clip(sampling_ratios2,min=0.1)

    sampled_pose=generate_random_CH_poses(ref_pose[0,:].reshape(9,-1).T, sampling_ratios1, alpha=alpha,beta=beta,fingers=fingers,transition=transition).reshape(600,600,9).permute(2,0,1)[None,...]

    # '''process quat sign'''
    # dist1=signed_cosine_distance(sampled_pose[:,0:4],ref_pose[:,0:4])
    # dist2=signed_cosine_distance(-sampled_pose[:,0:4],ref_pose[:,0:4])
    # f=torch.ones_like(ref_pose[:,0:1])
    # f[dist2<dist1]*=-1
    # sampled_pose[:,0:4]*=f

    sampled_pose = sampled_pose * sampling_ratios2 + (1 - sampling_ratios2) * ref_pose
    assert not torch.isnan(sampled_pose).any(), f'{sampled_pose}, {sampling_ratios.min()}, {sampled_pose.max()}'


    # max_angle_rad=2*np.pi*tou
    # sampled_pose[:, 0:3] =clip_vectors_angle_batch(sampled_pose[:, 0:3] ,gripper_pose[:, 0:3] ,max_angle_rad/2)
    # sampled_pose[:, 3:5] =clip_vectors_angle_batch_2d(sampled_pose[:, 3:5] ,gripper_pose[:, 3:5] ,max_angle_rad)

    sampled_pose[:, :3] = F.normalize(sampled_pose[:, :3], dim=1)
    sampled_pose[:, 3:5] = F.normalize(sampled_pose[:, 3:5], dim=1)

    # sampled_pose[:, 5:]=clip_scalars_batch(sampled_pose[:, 5:],gripper_pose[:, 5:],max_dist=tou)


    # '''clip fingers to scope'''
    sampled_pose[:,5:5+3]=torch.clamp(sampled_pose[:,5:5+3],max=0.99)

    # sampled_pose[:,0:4] = torch.where(sampled_pose[:, 0:1] >= 0, sampled_pose[:,0:4], -sampled_pose[:,0:4])


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