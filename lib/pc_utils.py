import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from Configurations.ENV_boundaries import ref_pc_center


def closest_point(point_data, center_point, radius=0.0025,warning_limit=0.005):
    # Find the neighbour points of the center_point
    distance = point_data - center_point
    distance = np.linalg.norm(distance, axis=1)
    index = np.argmin(distance)
    min_d = distance[index]

    if min_d > radius:
        return None
    return index

def point_index(point_data, center_point):
    # Find the neighbour points of the center_point
    distance = point_data - center_point
    distance = np.linalg.norm(distance, axis=1)
    index = np.argmin(distance)
    min_d = distance[index]
    if min_d ==0.:
        return index
    return None

def refine_point_cloud(point_cloud):
    pcd = numpy_to_o3d(point_cloud)
    # print(np.max(point_cloud))
    # view_o3d(pcd, view_coordinate=True)
    # print(len(pcd.points))
    # pcd, indexes = pcd.remove_radius_outlier(nb_points=16, radius=0.05,print_progress=True) # slow
    pcd, indexes = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.5)
    pcd, indexes = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.5)

    # print(len(pcd.points))

    # view_o3d(pcd, view_coordinate=True)
    point_cloud = np.asarray(pcd.points)
    return point_cloud


def numpy_to_o3d( pc,normals=None,color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if color is not None: pcd.colors = o3d.utility.Vector3dVector(color)
    if normals is not None: pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def scene_point_clouds_mask(point_data):
    xy_mask= np.logical_and(abs(point_data[:, 1] - 0.0025) < 0.30, abs(point_data[:, 0] - 0.43) < 0.20)
    z_mask = np.logical_and(point_data[:, 2] < 0.20, point_data[:, 2] > 0.04)
    mask = xy_mask & z_mask
    return point_data[mask]
def get_o3d_norms(pcd):

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd
def get_npy_norms(pc):
    pcd=numpy_to_o3d(pc=pc)
    pcd=get_o3d_norms(pcd)
    return pcd

def random_down_sampling(point_data,number_of_points):
    choices = np.random.choice(point_data.shape[0], number_of_points, replace=False)
    return point_data[choices, :]

def random_transformation(pc):
    # random shift
    shift=torch.rand(size=(1,3)).to('cuda')-0.5
    shift[:,2]/=5
    transformed_pc=pc-shift
    #random rotation
    rotation=R.random().as_matrix()
    rotation=torch.from_numpy(rotation).to('cuda')[None,None,:,:]
    transformed_pc=rotation*transformed_pc[:,:,:,None]
    transformed_pc=transformed_pc.sum(-2).float()
    return transformed_pc
def random_transition(pc):
    # random shift
    shift=torch.rand(size=(1,3)).to('cuda')-0.5
    shift[:, 0:2] /= 2
    shift[:,2]/=4
    transitioned_pc=pc-shift
    return transitioned_pc

def unit_sphere_pc_normalization( pc_data,constant_scale=True,constant_shift=True):
    pc_center = torch.mean(pc_data, dim=1,keepdim=True)

    # if not (torch.all(pc_center[:,:,0]<0.5) and torch.all(pc_center[:,:,0]>0.4) and torch.all(pc_center[:,:,1]<0.1) and torch.all(pc_center[:,:,1]>-0.1)
    #         and torch.all(pc_center[:,:,2]<0.1) and torch.all(pc_center[:,:,2]>0.0)):
    #     print(Fore.RED, 'point center out of range=', pc_center, Fore.RESET)
    if constant_shift:pc_center=ref_pc_center
    pc_data_new = pc_data - pc_center
    scale,ids_ = torch.max(torch.sqrt(torch.sum(pc_data_new ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)
    if constant_scale: pc_data_new /=0.36
    else:
        pc_data_new /= scale

    return pc_data_new,scale,pc_center

def circle_to_points(radius = 1,number_of_points = 30,x=0,y=0,z=None):
    interval_radian = (2 * np.pi) / number_of_points
    if z is None:
        point_cloud = [np.array([np.cos(interval_radian * i)+x,np.sin(interval_radian * i)+y]) for i in range(number_of_points)]
    else:
        point_cloud = [np.array([(np.cos(interval_radian * i)*radius) + x, (np.sin(interval_radian * i)*radius) + y,z]) for i in
                   range(number_of_points)]

    point_cloud=np.array(point_cloud)
    return point_cloud

def compute_curvature(points, radius=0.5):
    pcd = numpy_to_o3d(points)

    # Build a KDTree for efficient neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    curvature = []
    for point in points:
        # Find neighbors within the specified radius
        [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)

        # If not enough neighbors, set curvature to 0
        if k < 3:
            curvature.append(0)
            continue

        # Compute the covariance matrix of the neighborhood
        M = np.cov(np.asarray(pcd.points)[idx].T)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, _ = np.linalg.eigh(M)

        # Curvature is approximated by the ratio of the smallest eigenvalue to the sum of eigenvalues
        curvature.append(eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]))

    return curvature

def rotate_point_clouds(point_clouds,rotation_matrix):
    return np.matmul(rotation_matrix,point_clouds.T).T

def circle_points(r, n,x_center,y_center,z_center):
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = r * np.cos(t)+x_center
        y = r * np.sin(t)+y_center
        z=np.zeros_like(x)+z_center
        if z_center is not None:

            circles.append(np.c_[x, y,z])
        else:
            circles.append(np.c_[x, y,0])
    return circles

def circle_cavity_to_point(radius=1.0,n_circles=10,plot=False,x_center=0,y_center=0,z_center=0):
    r=[]
    n=[]
    for i in range(n_circles):
        current_radius  = (i/(n_circles-1))    *   radius
        r.append(current_radius)
        n.append(1+int(2*np.pi*i))

    circles = circle_points(r, n,x_center,y_center,z_center)

    if plot:
        fig, ax = plt.subplots()
        for circle in circles:
            ax.scatter(circle[:, 0], circle[:, 1])
        ax.set_aspect('equal')
        plt.show()
    return np.concatenate(circles,axis=0)

if __name__ == "__main__":
    circle_cavity_to_point(radius=0.7,n_circles=15,plot=True)

    # points=circle_to_points(radius=1,number_of_points=2,x=2,y=3,z=10)
    # print(points)
    # print(points.shape)