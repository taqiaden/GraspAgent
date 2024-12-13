import os

import numpy as np
import random
import time

from colorama import Fore

from lib.IO_utils import save_pickle, load_pickle
from lib.math_utils import angle_between_vectors_cross
from lib.pc_utils import refine_point_cloud
from lib.Multible_planes_detection.utils import *
from visualiztion import view_npy_open3d

'''Note: Plane detection method is borrowed from (https://github.com/yuecideng/Multiple_Planes_Detection)'''

cache_dir = os.getcwd() + r'/training/cache/bin_planes/'

x_vec = np.array([1., 0., 0.])
y_vec = np.array([0., 1., 0.])
z_vec = np.array([0., 0., 1.])

color1 = np.array([1., 0., 0.])
color2_1 = np.array([0., 1., 0.])
color2_2 = np.array([0.7, 1., 0.7])

color3_1 = np.array([0., 0., 1.])
color3_2 = np.array([0.7, 0.7, 1.])

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
def get_bin_planes_equations(pc,view=False):
    t0 = time.time()
    results = DetectMultiPlanes(pc, min_ratio=0.01, threshold=0.005, iterations=2000)
    if view:
        print('Time:', time.time() - t0)
    planes = []
    colors = []
    bin_sides = []
    for plane_equ, plane in results:

        color = np.zeros((plane.shape[0], plane.shape[1]))

        '''metrics'''
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

        z_max = np.max(plane[:, 2])
        z_min = np.min(plane[:, 2])

        if view:
            print('--------------------------------------------------------------')
            print('plane equation: ', plane_equ)
            print(f'angles with x,y,z vectors: {x_deg}, {y_deg}, {z_deg}')
            print(f'n points: ', plane.shape[0])
            print(f'x range={x_range}, y range={y_range}, z range={z_range}')
            print(f'x center={x_center}, y center={y_center}, z center={z_center}')
            print('z max = ', z_max)
            print('z min = ', z_min)

        '''floor detection'''
        if x_deg > 88 and y_deg > 88 and z_deg < 2 and x_range > 0.25 and y_range > 0.4 and z_min < 0.05:
            '''floor'''
            if view:
                color[:] = color1
                print('bin floor detected')
                # plane_masked_view(points, plane_equ, z_max, -1)
            bin_sides.append((plane_equ, z_max, -1))

        elif x_deg > 88 and 45 > y_deg > 42 and 47 > z_deg > 45 and x_range > 0.38:
            '''sides along the y axis'''
            if plane_normal[-1] < 0:
                if view:
                    color[:] = color2_1
                    print('bin side along positive y axis is detected')
                    # plane_masked_view(points, plane_equ, z_max, 1)
                bin_sides.append((plane_equ, z_max, 1))

            else:
                if view:
                    color[:] = color2_2
                    print('bin side along negative y axis is detected')
                    # plane_masked_view(points, plane_equ, z_max, -1)
                bin_sides.append((plane_equ, z_max, -1))

        elif 30 > x_deg > 27 and y_deg > 88 and 63 > z_deg > 60 and y_range > 0.55:
            '''sides along the x axis'''
            if plane_normal[-1] > 0:
                if view:
                    color[:] = color3_1
                    print('bin side along negative x axis is detected')
                    # plane_masked_view(points, plane_equ, z_max, -1)
                bin_sides.append((plane_equ, z_max, -1))

            else:
                if view:
                    color[:] = color3_2
                    print('bin side along positive x axis is detected')
                    # plane_masked_view(points, plane_equ, z_max, 1)
                bin_sides.append((plane_equ, z_max, 1))

        elif view:
            r = random.random()
            g = random.random()
            b = random.random()
            color[:, 0] = r
            color[:, 1] = g
            color[:, 2] = b

        planes.append(plane)
        colors.append(color)
        if len(bin_sides) >= 5: break

    if view:
        planes = np.concatenate(planes, axis=0)
        colors = np.concatenate(colors, axis=0)
        DrawResult(planes, colors)

    if len(bin_sides) < 5: return None
    else:
        return bin_sides


def bin_planes_detection(points,threshold = 0.001,view=False,file_index=None):
    if file_index is not None:
        file_path = cache_dir + file_index + '.pkl'
        if os.path.exists(file_path):
            bin_sides=load_pickle(file_path)

        else:
            bin_sides=get_bin_planes_equations(points, view=view)
            if file_index is not None:
                file_path = cache_dir + file_index + '.pkl'
                save_pickle(file_path, bin_sides)
    else:
        bin_sides = get_bin_planes_equations(points, view=view)
        if file_index is not None:
            file_path = cache_dir + file_index + '.pkl'
            save_pickle(file_path, bin_sides)


    '''get bin mask'''
    if bin_sides is None: return None
    masks_list = []

    for plane_equ, z_max, direction in bin_sides:
        p_ = points[:] * plane_equ[0:3]
        p_ = np.sum(p_, axis=-1)
        p_ += plane_equ[3]
        if direction > 0:
            masks_list.append((p_ > -threshold) & (points[:, 2] <= z_max))
        else:
            masks_list.append((p_ < threshold) & (points[:, 2] <= z_max))

    bin_mask = masks_list[0] | masks_list[1] | masks_list[2] | masks_list[3] | masks_list[4]

    if view:
        view_npy_open3d(points[bin_mask])
    return bin_mask

if __name__ == "__main__":


    # points = ReadPlyPoint('Data/test1.ply')
    points = np.load('pc_tmp_data.npy')
    points = refine_point_cloud(points)

    # spatial_mask = estimate_object_mask(points, custom_margin=0.03)
    # points=points[~spatial_mask]
    # pre-processing
    # points = RemoveNan(points)
    # points = DownSample(points,voxel_size=0.003)
    # points = RemoveNoiseStatistical(points, nb_neighbors=50, std_ratio=0.5)

    # DrawPointCloud(points, color=(0.4, 0.4, 0.4))
    bin_planes_detection(points,view=True)
