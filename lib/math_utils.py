import math
import time
import random
import numpy as np
import torch
from sklearn.preprocessing import normalize
import transforms3d.euler as eul
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from Configurations import config


def seeds(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
def fibonacci_sphere(samples=1000,angle_limit=None):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment
        # print(theta)
        # if theta>60:continue
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        if angle_limit:
            angle=(math.atan2(math.sqrt(x**2+y**2),z)*180)/np.pi
            if angle>angle_limit :
                continue

        # print(math.sqrt(x**2+y**2+z**2))
        points.append((x, y, z))

    return points
semi_sphere_of_points=np.asarray( fibonacci_sphere(samples=1000,angle_limit=config.theta_scope))

def asSpherical_b(xyz):
    # takes list xyz (single coord)
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r) * 180 / np.pi  # to degrees
    phi = np.arctan2(y, x) * 180 / np.pi
    return  theta, phi

def asCartesian_b( theta, phi):
    #takes list rthetaphi (single coord)
    r       = np.full_like(theta,1.)
    theta   = theta* np.pi/180 # to radian
    phi     = phi* np.pi/180
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    points=np.stack([x,y,z],axis=-1)
    return points
def custom_spherical_coordinate(direction):
    theta, phi = asSpherical_b(direction)

    phi[phi<0]=360.+phi[phi<0]
    # if phi < 0: phi = 360. + phi
    return theta,phi
def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1]* np.pi/180 # to radian
    phi     = rthetaphi[2]* np.pi/180
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [x,y,z]

def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arccos(z/r)*180/ np.pi #to degrees
    phi     =  np.arctan2(y,x)*180/ np.pi
    return [r,theta,phi]
def statistical_info_(npy,array_name):
    std=np.std(npy)
    ave=np.average(npy)
    max_=np.max(npy)
    min_=np.min(npy)
    print(f'{array_name} statistical: ave={ave},std={std},max={max_},min={min_}')
def rotation_matrix_from_vectors(source_vec, destination_vec):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    no_rotation_mat=np.array([[1.,0.,0.],[0.,1.,.0],[0.,0.,1.]])
    a, b = (source_vec / np.linalg.norm(source_vec)).reshape(3), (destination_vec / np.linalg.norm(destination_vec)).reshape(3)
    v = np.cross(a, b)
    if np.sum(v)==0.0: return no_rotation_mat
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
def standardization(data,ave=None,std=None,axis=0):
    average = np.average(data, axis=axis) if ave is None else ave
    std = np.std(data, axis=axis) if std is None else ave
    for i in range(std.shape[0]):
        if std[i] < 0.000001: std[i] = 1
    result = (data - average) / std
    return result,average,std

def one_hot(bins_size,one_id,rows=1):
    one_hot_array=np.full((rows,bins_size),0)
    one_hot_array[:,one_id]=1
    return one_hot_array

def vector_length(vector):
    sum=0
    for i in vector:
        sum+=i*i
    return math.sqrt(sum)

def random_point_within_a_sphere(radius,ref_xyz=[0,0,0],only_negative_z=False):
    delta= random_point_on_a_sphere(random()*radius,only_negative_z=only_negative_z)
    return delta+ref_xyz
def random_point_on_a_sphere(radius,only_negative_z=False,ref_xyz=[0,0,0]):
    x, y, z = random(), random(), random()
    x = -x if random() > 0.5 else x
    y = -y if random() > 0.5 else y
    z = -z if random() > 0.5 or only_negative_z  else z

    denominator = math.sqrt(x * x + y * y + z * z)
    x = (x / denominator) * radius
    y = (y / denominator) * radius
    z = (z / denominator) * radius
    return np.asarray([x,y,z])+np.asarray(ref_xyz)

def max_normalization(list,scale=1):
    normalized_list=[(i-min(list))*scale/(max(list)-min(list)) for i in list]
    return normalized_list

def data_entropy(data,n_bins):
    # if not isinstance(data,np.array()):
    #     data=np.array(data)
    hist = np.histogramdd(data, bins=n_bins)[0]
    hist /= hist.sum()
    hist = hist.flatten()
    hist = hist[hist.nonzero()]
    entropy = -0.5 * np.sum(hist * np.log2(hist))
    return entropy

def rot2eul(R):
    # method 1
    # beta = -np.arcsin(R[2,0])
    # alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    # gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    # return np.array((alpha, beta, gamma))
    # method 2 : faster
    ang = eul.mat2euler(R, axes='sxyz')
    return np.array(ang)

def change_angle(Homogenous_matrix, angle_index, value):
    rotation_mat = Homogenous_matrix[0:3, 0:3]
    angles = rot2eul(rotation_mat)
    angles[angle_index] += value
    new_rot = eul2rot(angles)
    Homogenous_matrix[0:3, 0:3] = new_rot
    return Homogenous_matrix


def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3\

    return matrix
def eul2rot(theta) :
    # method 1
    # R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
    #               [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
    #               [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
    # return R
    # method 2 : faster
    R_=eul.euler2mat(theta[0], theta[1], theta[2], axes='sxyz')
    return np.array(R_)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def fps(points, n_samples,first_point_idx=None):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    print(f'points dimension for FPS calculation = {points.shape[-1]}')
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int')  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf')  # [P]

    # Select a point from points by its index, save it

    selected = 0 if first_point_idx is None else first_point_idx
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
                (points[last_added] - points[points_left]) ** 2).sum(-1)  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return sample_inds

def n_mask_(npy,n,k):
    assert k==1 or k==-1
    return (k*npy).argsort().argsort() < n
def maximum_n_mask(npy,n):
    return n_mask_(npy,n,-1)
def minimum_n_mask(npy,n):
    return n_mask_(npy,n,1)

def cv(array):
    return np.std(array) / np.average(array)

def min_max_normalization(npy,epsilon=0.,axis=0):
    max_s = np.max(npy,axis=axis)
    min_s = np.min(npy,axis=axis)-epsilon
    return (npy - min_s) / (max_s - min_s)

def rotationMatrixToQuaternion1(m):
    #q0 = qw
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if(t > 0):
        t = np.sqrt(t + 1)
        q[0] = 0.5 * t
        t = 0.5/t
        q[1] = (m[2,1] - m[1,2]) * t
        q[2] = (m[0,2] - m[2,0]) * t
        q[3] = (m[1,0] - m[0,1]) * t
    else:
        i = 0
        if (m[1,1] > m[0,0]):
            i = 1
        if (m[2,2] > m[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3
        t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[0] = (m[k,j] - m[j,k]) * t
        q[j] = (m[j,i] + m[i,j]) * t
        q[k] = (m[k,i] + m[i,k]) * t
    return q
def rot_to_quat(rot_mat):
    R.from_matrix()
def tanh_to_unity(data):
    data=(data+1.)/2
    return data

def line_distance(points,start,end):
    if np.all(start==end):
        return np.linalg.norm(points-start,axis=1)

    vec=end-start
    cross=np.cross(vec,start-points)
    return np.divide(abs(cross),np.linalg.norm(vec))


def distance_point_clouds_to_vector(point_clouds,vector):
    normalized_vector=vector/np.linalg.norm(vector)
    # return np.dot(point_clouds,normalized_vector)
    projections=np.dot(point_clouds,normalized_vector)*normalized_vector
    return projections
    distance=np.linalg.norm(point_clouds-projections,axis=1)

    return distance

if __name__ == "__main__":
    l=[1,2,3]
    print(max_normalization(l,10))
    exit()

    for i in range(100):
        p = (random_point_within_a_sphere(20,only_negative_z=True))
        print(p)
        # print(vector_length(p))