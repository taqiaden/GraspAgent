import numpy as np
import torch
import torch.nn.functional as F
from lib.grasp_utils import shift_a_distance
from Configurations import config
from lib.bbox import rotation_matrix_to_angles, angles_to_rotation_matrix, \
    construct_transformation
from lib.one_hot_utiles import unit_regression_to_one_hot, clean_one_hot, one_hot_to_unit_regression, \
    unit_regression_to_indexes, one_hot_to_indexes


def pose_7_to_pose_5(pose_7):
    target_pose=pose_7.clone()
    approach = target_pose[0:3]
    theta, phi_sin, phi_cos = approach_vec_to_theta_phi(approach[None, :])
    target_pose[0:1] = theta
    target_pose[1:2] = phi_sin
    target_pose[2:3] = phi_cos
    pose_5 = output_processing(target_pose[None, :, None]).squeeze(-1)
    return pose_5



def pose_7_to_transformation(pose_7,target_point,clip=True):

    # relative_pose_5 = pose_7_to_pose_5(pose_7)
    # T_d, width, distance = convert_angles_to_transformation_form(pose_7, target_point,approach=pose_7[0:3])
    pose_7=pose_7.squeeze()
    beta= angle_ratio_from_sin_cos(pose_7[ 3:4], pose_7[ 4:5])*config.beta_scope

    if clip:
        distance=torch.clip(pose_7[ 5:6 ],0,1)*config.distance_scope
        width=torch.clip(pose_7[6:7],0,1)*config.width_scope
    else:
        distance = pose_7[ 5:6 ]*config.distance_scope
        width = pose_7[6:7]*config.width_scope

    approach=pose_7[0:3]


    '''angles to rotation matrix'''
    rotation_matrix=angles_to_rotation_matrix(approach,beta.squeeze())

    '''transformation matrix'''
    T_0=construct_transformation(target_point, rotation_matrix)

    '''adjust the penetration distance for the transformation'''
    # print(distance)
    # print(T_0)
    T_d = shift_a_distance(T_0, distance)
    assert T_d[0:3, 3].shape == target_point.shape,f'{T_d[0:3, 3].shape},  {target_point.shape}'
    assert T_d[0:3, 0:3].shape == rotation_matrix.shape
    return T_d, width, distance

def label_to_transformation(label):
    distance = label[22]
    width = label[21] / config.width_scale
    transformation = label[5:21].copy().reshape(-1, 4)
    return transformation,width,distance


def encode_gripper_pose_npy(distance, width, rotation_matrix):
    width =width/ config.width_scope
    distance =distance/ config.distance_scope
    theta2, phi2, beta2 = rotation_matrix_to_angles(rotation_matrix)
    theta2= theta2 / config.theta_scope
    phi2= phi2 / config.phi_scope
    beta2 = beta2 / config.beta_scope
    beta_sin_360, beta_cos_360 = sin_cos_encoding(beta2, 2 * np.pi)
    approach=unit_theta_phi_to_vector(theta2,phi2)
    pose=np.array([approach[0],approach[1],approach[2],beta_sin_360,beta_cos_360,distance, width],dtype=float)
    # pose = pose[np.newaxis, :]

    return pose
def soft_label_template_(size,k=1):
    soft_label_template=[]
    factor=-100/(k*(size**2))
    for i in range(size):
        exp_=factor*((i-size/2)**2)
        soft_label_template.append(2**exp_)
    return np.asarray(soft_label_template)
def indep_roll(arr, shifts, axis=1):
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted.
    """
    arr = np.swapaxes(arr,axis,-1)
    all_idcs = np.ogrid[[slice(0,n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result,-1,axis)
    return arr
def sof_label_(label,size,k=1.0,circular_scope=False):

    n=len(label)

    soft_one_hot_label=soft_label_template_(size,k)
    soft_one_hot_label=np.tile(soft_one_hot_label,(n,1))

    roll_values=(label-size/2).cpu().numpy()
    roll_values=np.floor(roll_values).astype(int)
    soft_one_hot_label=indep_roll(soft_one_hot_label,roll_values.copy())

    # exit()
        # np.roll(soft_one_hot_label, roll_value))
    if  circular_scope==False: # the beginning and end of the scope are not connected
        for i in range(n):
            roll_value_=roll_values[i]
            if roll_value_>0:
                soft_one_hot_label[i,:roll_value_]=0.
            elif roll_value_<0:
                soft_one_hot_label[i,roll_value_:] = 0.

    # soft_label_dict[key]=soft_one_hot_label
    soft_one_hot_label=torch.from_numpy(soft_one_hot_label).cuda().float()
    return (soft_one_hot_label+0.12)/1.2

def unit_theta_phi_to_vector(theta_,phi_):
    theta=config.theta_scope_rad*theta_
    phi=config.phi_scope_rad*phi_
    if torch.is_tensor(theta):
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        approach_direction = torch.stack([x, y, z], axis=-1)
    else:
        x =  np.sin( theta ) * np.cos( phi )
        y = np.sin( theta ) * np.sin( phi )
        z =  np.cos( theta )
        approach_direction=np.stack([x,y,z],axis=-1)
    return approach_direction

def angle_ratio_from_sin_cos(_sin,_cos):
    _angle=sin_cos_to_angle(_sin, _cos)
    _angle=_angle/(2*np.pi)
    return _angle

def output_processing(output):
    theta = output[:, 0:1, :]
    phi_ratio = angle_ratio_from_sin_cos(output[:, 1:2, :], output[:, 2:3, :])

    beta_ratio = angle_ratio_from_sin_cos(output[:, 3:4, :], output[:, 4:5, :])
    dist = output[:, 5:6, :]
    width = output[:, 6:7, :]
    # dist_width=output[:,6:,:]*scale
    output = torch.cat([theta, phi_ratio, beta_ratio, dist, width], dim=1)
    return output

def sin_cos_encoding(data,scope_rad=None):
    if scope_rad is None:
        scope_rad=2*np.pi
    angle=data*scope_rad
    if torch.is_tensor(angle):
        sin_val = torch.sin(angle)
        cos_val = torch.cos(angle)
    else:
        sin_val=np.sin(angle)
        cos_val=np.cos(angle)
    return sin_val,cos_val

def dense_pose_to_sin_cos(dense_pose_ratio):
    theta_ratio = dense_pose_ratio[:, 0:1, :]
    phi_ratio = dense_pose_ratio[:, 1:2, :]
    phi_sin, phi_cos = sin_cos_encoding(phi_ratio,2*np.pi)
    beta_ratio = dense_pose_ratio[:, 2:3, :]
    beta_sin_360, beta_cos_360 = sin_cos_encoding(beta_ratio, 2*np.pi) # when encoding beta to sin and cos it is considered to be in scope of 360 although the actual scope is 180
    dist = dense_pose_ratio[:, 3:4, :]
    width=dense_pose_ratio[:,4:5,:]
    dense_pose_7 = torch.cat([theta_ratio, phi_sin, phi_cos, beta_sin_360, beta_cos_360, dist,width], dim=1)
    return dense_pose_7

def sin_cos_to_angle(sin_data,cos_data,output_ratio=False):
    angle = torch.atan2(sin_data, cos_data)
    angle[angle < 0] = 2 * np.pi + angle[angle < 0]
    if output_ratio:angle = angle / (np.pi * 2)
    return angle

def approach_vec_to_theta_phi(xyz):
    # takes list xyz (single coord)
    # print(xyz.shape)
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]
    r = torch.sqrt(x * x + y * y + z * z)
    theta = torch.arccos(z / r)
    phi = torch.atan2(y, x)
    phi[phi < 0] = (2 * np.pi) + phi[phi < 0]
    theta_ratio=theta/config.theta_scope_rad

    phi_sin, phi_cos = sin_cos_encoding(phi)
    return theta_ratio, phi_sin, phi_cos

def vectors_to_ratio_metrics(poses):
    x = poses[:, 0:1]
    y = poses[:, 1:2]
    z = poses[:, 2:3]
    r=torch.norm(poses[:,0:3],dim=-1,keepdim=True)

    theta = torch.arccos(z / r)
    phi = torch.atan2(y, x)
    phi[phi < 0] = (2 * np.pi) + phi[phi < 0]
    phi_ratio=phi/config.phi_scope_rad
    theta_ratio = theta / config.theta_scope_rad
    phi_ratio=torch.clip(phi_ratio,0,1.0)
    theta_ratio=torch.clip(theta_ratio,0,1.0)
    beta_ratio = angle_ratio_from_sin_cos(poses[:, 3:4], poses[:, 4:5])
    dist_ratio = poses[:, 5:6]
    width_ratio = poses[:, 6:7]
    dist_ratio=torch.clip(dist_ratio,0,1.0)
    width_ratio=torch.clip(width_ratio,0,1.0)

    output = torch.cat([theta_ratio, phi_ratio, beta_ratio, dist_ratio, width_ratio], dim=1)
    return output

def theta_phi_to_approach_vec(theta_ratio,phi):
    theta=theta_ratio*config.theta_scope_rad
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    approach_direction = torch.stack([x, y, z], axis=-1)
    return approach_direction

class param():
    def __init__(self,upper_bound,bin_size,start_index,is_circular,param=None):
        self.val=None
        self.one_hot=None
        self.index=None
        self.lower_bound=0
        self.upper_bound=upper_bound
        self.bin_size=bin_size
        self.bins=int((self.upper_bound-self.lower_bound)/self.bin_size)
        self.start_index=start_index
        self.end_index=self.start_index+self.bins
        self.is_circular=is_circular
        if param is not None: self.upper_bound(param)
    def update(self,data):
        if data.shape[-1]==1:
            self.val=data
            self.one_hot=None
            self.index=None
        elif data.shape[-1]==self.bins:
            self.one_hot=data
            self.val=None
            self.index=None

    def get_one_hot(self,input=None,clean=False):
        assert input is not None or self.one_hot is not None or self.val is not None or self.index is  not None
        if input is not None : self.update(input)
        if self.one_hot is not None:
            if clean: self.one_hot= clean_one_hot(self.one_hot)
            return self.one_hot
        else:
            self.one_hot= unit_regression_to_one_hot(self.val,self.bins)
            return self.one_hot

    def get_val(self,input=None):
        assert input is not None or self.one_hot is not None or self.val is not None or self.index is  not None
        if input is not None : self.update(input)
        if self.val is not None: return self.val
        else:
            self.val= one_hot_to_unit_regression(self.one_hot,self.bins)
            return self.val

    def get_indexes(self,input=None):
        assert input is not None or self.one_hot is not None or self.val is not None or self.index is not None
        if input is not None: self.update(input)
        if self.val is not None:
            indexes=unit_regression_to_indexes(self.val, self.bins)
        else:
            indexes=one_hot_to_indexes(self.one_hot)
        return indexes

class pose_obj():
    def __init__(self,pose=None):
        self.one_hot = None
        self.val = None
        self.indexes=None
        self.theta=param(upper_bound=config.theta_scope,bin_size=config.theta_bins_size,is_circular=False,start_index=0)
        self.phi=param(upper_bound=config.phi_scope,bin_size=config.phi_bins_size,is_circular=True,start_index=self.theta.end_index)
        self.beta=param(upper_bound=config.beta_scope,bin_size=config.beta_bins_size,is_circular=True,start_index=self.phi.end_index)
        self.dist=param(upper_bound=config.distance_scope,bin_size=config.distance_bin_size,is_circular=False,start_index=self.beta.end_index)
        self.width=param(upper_bound=config.width_scope,bin_size=config.width_bin_size,is_circular=False,start_index=self.dist.end_index)
        self.param_list=[self.theta,self.phi,self.beta,self.dist,self.width]


        if pose is not None:
            self.update_pose(pose)

    def update_pose(self,pose):
        if pose.shape[-1] == self.pose_bins:

            self.one_hot = pose
            self.val=None
            self.indexes=None
            for i in range(5):
                self.param_list[i].one_hot=pose[:, self.param_list[i].start_index:self.param_list[i].end_index]
                self.param_list[i].val=None
                self.param_list[i].index=None



        elif pose.shape[-1] == 5:
            self.val = pose
            self.one_hot=None
            self.indexes=None
            for i in range(5):
                self.param_list[i].val=pose[:, i:i+1]
                self.param_list[i].one_hot=None
                self.param_list[i].index=None

        else: assert 1==2, 'Wrong shape'

    def get_one_hot(self,pose=None,clean=False):
        assert pose!=None or self.one_hot != None or self.val!=None or self.indexes!=None
        if pose is not None : self.update_pose(pose)
        if self.one_hot is not None and clean==False: return self.one_hot
        else:
            one_hot_list=[]
            for i in range(5):
                self.param_list[i].get_one_hot(clean=clean)
                one_hot_list.append(self.param_list[i].one_hot)
            self.one_hot=torch.cat(one_hot_list,dim=-1)
            return self.one_hot
    def get_val(self, pose=None):
        assert pose != None or self.one_hot != None or self.val != None or self.indexes != None
        if pose is not None: self.update_pose(pose)
        if self.val is not None:
            return self.val
        else:
            val_list = []
            for i in range(5):
                self.param_list[i].get_val()
                val_list.append(self.param_list[i].val)
            self.val = torch.cat(val_list, dim=-1)
            return self.val

    def get_soft_one_hot(self,pose=None,decayed=False,soft_max=False):
        self.get_one_hot(pose)
        soft_one_hot_list=[]
        for i in range(5):
            if decayed:
                indexes=self.param_list[i].get_indexes()
                k=0.5
                label=sof_label_(indexes, size=self.param_list[i].bins,k=k, circular_scope=self.param_list[i].is_circular)
            else: label=self.param_list[i].get_one_hot()

            soft_one_hot_=F.softmax(label,dim=-1) if soft_max else label
            soft_one_hot_list.append(soft_one_hot_)
        soft_one_hot=torch.cat(soft_one_hot_list,dim=-1)
        return soft_one_hot
