import os
import random
import time

import numpy as np

from lib.IO_utils import save_pickle, load_pickle
from lib.Multible_planes_detection.utils import *
from lib.math_utils import angle_between_vectors_cross
from lib.pc_utils import refine_point_cloud
from visualiztion import view_npy_open3d

'''Note: Plane detection method is borrowed from (https://github.com/yuecideng/Multiple_Planes_Detection)'''

cache_dir = os.getcwd() + r'/cache/'

x_vec = np.array([1., 0., 0.])
y_vec = np.array([0., 1., 0.])
z_vec = np.array([0., 0., 1.])

def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    """ Detect multiple planes from given point clouds

    Args:
        points (np.ndarray):
        min_ratio (float, optional): The minimum left points ratio to end the Detection. Defaults to 0.05.
        threshold (float, optional): RANSAC threshold in (m). Defaults to 0.01.

    Returns:
        [List[tuple(np.ndarray, List)]]: Plane equation and plane point index
    """

    plane_list = []
    N = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        w, index = PlaneRegression(
            target, threshold=threshold, init_n=3, iter=iterations)

        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)

    return plane_list

def plane_masked_view(points_,plane_equ_,z_max_,direction_):
    # p_x = sub_points[:] * plane_equ_[0:3]
    # p_x = np.sum(p_x, axis=-1)
    # p_x+= plane_equ_[3]
    # print(p_x)
    p_ = points_[:] * plane_equ_[0:3]
    p_ = np.sum(p_, axis=-1)
    p_ += plane_equ_[3]
    if  direction_>0:
        mask=(p_>-0.005) & (points_[:,2]<=z_max_)
    else:
        mask=(p_ < 0.005) & (points_[:,2]<=z_max_)

    view_npy_open3d(points_[mask])

def refine_floor_plane(plane):
    '''refine floor'''
    floor_results = DetectMultiPlanes(plane, min_ratio=0.01, threshold=0.001, iterations=2000)

    best_fit_floor_plane_equ = None
    best_floor_set = None
    s = 0
    for floor_plane_equ, floor_plane in floor_results:
        if floor_plane.shape[0] > s:
            s = floor_plane.shape[0]
            best_floor_set = floor_plane
            best_fit_floor_plane_equ = floor_plane_equ

    # if view:
    #     view_npy_open3d(best_floor_set)
    return best_fit_floor_plane_equ

def view_statistics(plane_equ,x_deg,y_deg,z_deg,plane,x_range,y_range,z_range,x_center,y_center,z_center,z_max,z_min):
    print('--------------------------------------------------------------')
    print('plane equation: ', plane_equ)
    print(f'angles with x,y,z vectors: {x_deg}, {y_deg}, {z_deg}')
    print(f'n points: ', plane.shape[0])
    print(f'x range={x_range}, y range={y_range}, z range={z_range}')
    print(f'x center={x_center}, y center={y_center}, z center={z_center}')
    print('z max = ', z_max)
    print('z min = ', z_min)


def collect_bin_planes(pc,view=False,min_ratio=0.01, threshold=0.005, iterations=2000):

    # t0 = time.time()

    # plane_list = []
    N = len(pc)
    target = pc.copy()
    count = 0

    all_planes = []
    detected_bin_tuple = []
    while count < (1 - min_ratio) * N:
        plane_equ, index = PlaneRegression(
            target, threshold=threshold, init_n=3, iter=iterations)

        count += len(index)
        plane=target[index]

        '''angels to xyz'''
        plane_normal = plane_equ[0:3]
        _, x_deg = angle_between_vectors_cross(plane_normal, x_vec)
        _, y_deg = angle_between_vectors_cross(plane_normal, y_vec)
        _, z_deg = angle_between_vectors_cross(plane_normal, z_vec)

        '''points range'''
        x_range = np.max(plane[:, 0]) - np.min(plane[:, 0])
        y_range = np.max(plane[:, 1]) - np.min(plane[:, 1])
        z_range = np.max(plane[:, 2]) - np.min(plane[:, 2])

        '''center'''
        x_center = np.min(plane[:, 0]) + x_range / 2
        y_center = np.min(plane[:, 1]) + y_range / 2
        z_center = np.min(plane[:, 2]) + z_range / 2

        '''elevation boundaries'''
        z_max = np.max(plane[:, 2])
        z_min = np.min(plane[:, 2])


        if view:
            view_statistics(plane_equ, x_deg, y_deg, z_deg, plane, x_range, y_range, z_range, x_center, y_center,
                            z_center, z_max, z_min)

        if x_deg > 88 and y_deg > 88 and z_deg < 2 and x_range > 0.25 and y_range > 0.4 and z_min < 0.05 and z_max<0.06:
            best_fit_floor_plane_equ=refine_floor_plane(plane)
            '''floor'''
            if view:
                print('bin floor detected')
            detected_bin_tuple.append((best_fit_floor_plane_equ, z_max, -1))

        elif x_deg > 88 and 50 > y_deg > 42 and 47 > z_deg > 40 and x_range > 0.35 :
            '''sides along the y axis'''
            if plane_normal[-1]*plane_normal[-2] < 0 and y_center>0.2:
                if view:
                    print('bin side along positive y axis is detected')
                if plane_normal[-2]>0:
                    detected_bin_tuple.append((plane_equ, z_max, 1))
                else:
                    detected_bin_tuple.append((plane_equ, z_max, -1))

            elif y_center<-0.2:
                if view:
                    print('bin side along negative y axis is detected')
                detected_bin_tuple.append((plane_equ, z_max, -1))

        elif 30 > x_deg > 24 and y_deg > 85 and 65 > z_deg > 58 and y_range > 0.5:
            '''sides along the x axis'''
            if plane_normal[-1] > 0 and x_center<0.3:
                if view:
                    print('bin side along negative x axis is detected')
                detected_bin_tuple.append((plane_equ, z_max, -1))
            elif x_center>0.5:
                if view:
                    print('bin side along positive x axis is detected')
                detected_bin_tuple.append((plane_equ, z_max, 1))

        all_planes.append(plane)
        if len(detected_bin_tuple)>=5:break
        target = np.delete(target, index, axis=0)

    # if view:
    #     print('Time:', time.time() - t0)
    return detected_bin_tuple,all_planes

def get_bin_planes_equations(pc,view=False):
    detected_bin_tuple,all_planes=collect_bin_planes(pc,view=view)

    if view:
        colors = []
        for i in range(len(all_planes)):
            color = np.zeros((all_planes[i].shape[0], all_planes[i].shape[1]))
            r = random.random()
            g = random.random()
            b = random.random()
            color[:, 0] = r
            color[:, 1] = g
            color[:, 2] = b
            colors.append(color)

        planes = np.concatenate(all_planes, axis=0)
        colors = np.concatenate(colors, axis=0)
        DrawResult(planes, colors)
    if len(detected_bin_tuple) < 5: return None
    else:
        return detected_bin_tuple

def distance_to_plane(pc,plane_equ):
    plane_normal=plane_equ[0:3]
    p_ = pc[:] * plane_normal
    p_ = np.sum(p_, axis=-1)
    p_ += plane_equ[3]
    return p_

def get_edge_mask(pc,mask_,edge_threshold,disregarded_dimension=0):
    masked_pc = pc[mask_]
    max_elevated_point_arg = np.argmax(masked_pc[:, 2])
    max_elevated_point = masked_pc[max_elevated_point_arg]
    if disregarded_dimension==0:
        two2_pc = pc[:, [1,2]]
        dist = np.linalg.norm(two2_pc - max_elevated_point[np.newaxis, [1,2]],axis=-1)
    elif disregarded_dimension==1:
        two2_pc = pc[:, [0,2]]
        dist = np.linalg.norm(two2_pc - max_elevated_point[np.newaxis, [0,2]],axis=-1)
    elif disregarded_dimension == 2:

        two2_pc = pc[:, [0, 1]]
        dist = np.linalg.norm(two2_pc - max_elevated_point[np.newaxis, [0, 1]], axis=-1)

    edge_mask_ = dist < edge_threshold
    return edge_mask_

def bin_planes_detection(pc,sides_threshold = 0.0035,floor_threshold=0.0015,edge_threshold=0.005,view=False,file_index=None,cache_name='bin_planes'):

    if file_index is not None:
        file_path = cache_dir+cache_name+'/' + file_index + '.pkl'
        if os.path.exists(file_path):
            detected_bin_tuple=load_pickle(file_path)

        else:
            detected_bin_tuple=get_bin_planes_equations(pc, view=view)
            if file_index is not None:
                file_path = cache_dir +cache_name+'/' +  file_index + '.pkl'
                save_pickle(file_path, detected_bin_tuple)
    else:
        detected_bin_tuple = get_bin_planes_equations(pc, view=view)
        if file_index is not None:
            file_path = cache_dir + cache_name+'/' + file_index + '.pkl'
            save_pickle(file_path, detected_bin_tuple)

    '''get bin mask'''
    if detected_bin_tuple is None: return None
    masks_list = []

    for plane_equ, z_max, direction in detected_bin_tuple:
        dist_ = distance_to_plane(pc,plane_equ)
        if -0.3<plane_equ[0]<.3 and -0.3<plane_equ[1]<0.3 and plane_equ[2] >0.95  :
            t=floor_threshold
        else:
            t = sides_threshold

        if direction > 0:
            mask_=(dist_ > -t) & (pc[:, 2] <= z_max+t)
        else:
            mask_=(dist_ < t) & (pc[:, 2] <= z_max+t)

        if -0.3<plane_equ[0]<0.3 and -0.3<plane_equ[1]<0.3 and plane_equ[2] >0.95  :
            '''floor'''
            pass
        else:

            '''mask upper edges'''
            if -0.3<plane_equ[0]<0.3:
                '''sides along  y'''
                edge_mask = get_edge_mask(pc, mask_, edge_threshold, disregarded_dimension=0)
                mask_ = mask_ | edge_mask
            else:
                '''sides along  x'''
                edge_mask = get_edge_mask(pc, mask_, edge_threshold, disregarded_dimension=1)
                mask_ = mask_ | edge_mask

            if view:
                colors = np.zeros_like(pc)
                colors[edge_mask, 0] += 1.
                view_npy_open3d(pc, color=colors)


        if view:
            colors = np.zeros_like(pc)
            colors[mask_, 0] += 1.
            view_npy_open3d(pc, color=colors)
        masks_list.append(mask_)

    bin_mask = masks_list[0] | masks_list[1] | masks_list[2] | masks_list[3] | masks_list[4]

    if view:
        colors = np.zeros_like(pc)
        colors[bin_mask, 0] += 1.
        view_npy_open3d(pc, color=colors)

    return bin_mask

if __name__ == "__main__":
    points = np.load('pc_tmp_data.npy')
    points = refine_point_cloud(points)

    bin_planes_detection(points,view=True)
