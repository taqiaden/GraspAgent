import os
import subprocess

import cv2
import cv2 as cv
import numpy as np
from colorama import Fore

from Configurations import config
from Configurations.config import home_dir
from Configurations.run_config import simulation_mode, offline_point_cloud, activate_refine_point_cloud
from lib.depth_map import transform_to_camera_frame, point_clouds_to_depth
from lib.image_utils import view_image
from lib.pc_utils import refine_point_cloud, random_down_sampling, closest_point, scene_point_clouds_mask
from lib.report_utils import wait_indicator as wi
from registration import crop_scene_image, camera
from visualiztion import view_npy_open3d

sensory_pc_path = home_dir+'pc_tmp_data.npy'
sensory_RGB_path = home_dir+'RGB_tmp_data.npy'
sensory_depth_path = home_dir+'depth_tmp_data.npy'

get_point_bash='./bash/get_point.sh'
texture_image_path = config.home_dir + 'texture_image.jpg'
get_rgb_bash='./bash/get_rgb.sh'
rgb_path='Frame_0.ppm'

last_rgb=None
last_depth=None

def trigger_new_perception():

    if simulation_mode and offline_point_cloud == True:
        # get new data from data pool
        from lib.dataset_utils import online_data
        online_data=online_data()
        pc=online_data.load_random_pc()
        np.save(sensory_pc_path,pc)
        return

    ctime_stamp_rgb = os.path.getctime(rgb_path)
    ctime_stamp_texture = os.path.getctime(texture_image_path)
    ctime_stamp_pc = os.path.getctime(sensory_pc_path)

    subprocess.run(get_point_bash)
    subprocess.run(get_rgb_bash)
    wait = wi('Waiting for new perception data')
    i=0
    while (os.path.getctime(rgb_path)==ctime_stamp_rgb or
           ctime_stamp_pc == os.path.getctime(sensory_pc_path) or
           os.path.getctime(texture_image_path)==ctime_stamp_texture):

        if  offline_point_cloud == True:
            print(Fore.RED, 'Camera is not available. Load data from pool', Fore.RESET)
            break
        wait.step(0.01)
        i+=1
    wait.end()

def crop_side_tray_image(img):
    img_suction = img[200:540, 0:160, 0]
    img_grasp = img[180:570, 780:1032, 0]
    img_main=img[190:560, 170:770, 0]
    return img_suction, img_grasp,img_main

def crop_side_tray_RGB_image(img):
    img_suction = img[220:700, 0:240, :]
    img_grasp = img[220:700, 1070:1400, :]
    img_main = img[220:700, 295:1007, :]
    return img_suction, img_grasp,img_main

def get_side_bins_images():
    im = cv.imread(texture_image_path)
    im=cv2.GaussianBlur(im,(5,5),0)
    img_suction, img_grasp ,img_main= crop_side_tray_image(im)
    # view_image(img_main)
    # view_image(img_suction)
    # view_image(img_grasp)
    return img_suction, img_grasp,img_main

# def get_side_bins_RGB_images():
#     im = cv.imread(rgb_path)
#     im = cv2.rotate(im, cv2.ROTATE_180)
#     # view_image(im)
#     img_suction, img_grasp ,img_main= crop_side_tray_RGB_image(im)
#     # view_image(img_main)
#     # view_image(img_suction)
#     # view_image(img_grasp)
#     return img_suction, img_grasp,img_main

def get_scene_point_clouds():
    point_data = np.load(sensory_pc_path) # (<191000, 3) number of points is not constant

    point_data=refine_point_cloud(point_data)

    np.save(sensory_pc_path, point_data)

    full_point_clouds = scene_point_clouds_mask(point_data)

    return full_point_clouds

def get_scene_RGB():
    '''load and crop'''
    full_RGB=cv2.imread(rgb_path)
    full_RGB=cv2.cvtColor(full_RGB, cv2.COLOR_BGR2RGB)/255
    assert full_RGB.shape == (1200, 1920, 3), f'{full_RGB.shape}'
    scene_RGB = crop_scene_image(full_RGB)
    # view_image(scene_RGB)

    '''save'''
    np.save(sensory_RGB_path, scene_RGB)
    return scene_RGB

def get_scene_depth():
    '''load transform, and convert'''
    point_clouds = np.load(sensory_pc_path) # (<191000, 3) number of points is not constant
    # view_npy_open3d(point_clouds)
    if activate_refine_point_cloud:
        point_clouds = refine_point_cloud(point_clouds)
    # view_npy_open3d(point_clouds)
    transformed_point_clouds = transform_to_camera_frame(point_clouds)
    depth = point_clouds_to_depth(transformed_point_clouds, camera)

    '''save'''
    np.save(sensory_depth_path, depth)
    return depth

def random_sampling_augmentation(center_point, point_data, number_of_points):
    i=0
    while True:
        i+=1
        point_data_=random_down_sampling(point_data,number_of_points)
        if i>10:
            print(Fore.RED,'Warning: Unable to find closest point, after down sampling   ',Fore.RESET )
            return point_data_, None
        index = closest_point(point_data_, center_point)
        if index:
            return point_data_, index