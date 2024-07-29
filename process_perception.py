import os
import subprocess
from colorama import Fore

from Configurations.config import home_dir
from lib.pc_utils import refine_point_cloud, apply_mask, random_down_sampling, closest_point
from lib.report_utils import wait_indicator as wi
import numpy as np
from Configurations import config
from Run import simulation_mode
import cv2 as cv
sensory_pc_path = home_dir+'pc_tmp_data.npy'
get_point_bash='./bash/get_point.sh'
texture_image_path = config.home_dir + 'texture_image.jpg'
get_rgb_bash='./bash/get_rgb.sh'
rgb_path='Frame_0.ppm'
offline_point_cloud= True

def get_new_perception():
    ctime_stamp = os.path.getctime(texture_image_path)
    ctime_stamp_pc = os.path.getctime(sensory_pc_path)
    if simulation_mode and offline_point_cloud == True:
        # get new data from data pool
        from lib.dataset_utils import online_data
        online_data=online_data()
        pc=online_data.load_random_pc()
        np.save(sensory_pc_path,pc)

    subprocess.run(get_point_bash)
    # os.system(get_point_bash)
    wait = wi('Waiting for new perception data')
    i=0
    while os.path.getctime(texture_image_path)==ctime_stamp or ctime_stamp_pc == os.path.getctime(sensory_pc_path):
        # if (i>1000 and get_last_data==True) or offline_point_cloud==True:
        if  offline_point_cloud == True:
            print(Fore.RED, 'Camera is not available. Load data from pool', Fore.RESET)
            break
        wait.step(0.01)
        i+=1
    wait.end()

def crop_side_tray_image(img):
    img_suction = img[200:540, 0:160, 0]
    img_grasp = img[180:570, 780:1032, 0]
    return img_suction, img_grasp

def get_side_bins_images():
    im = cv.imread(texture_image_path)
    img_suction, img_grasp = crop_side_tray_image(im)
    return img_suction, img_grasp


def get_rgb():
    ctime_stamp = os.path.getctime(rgb_path)
    subprocess.run(get_rgb_bash)
    # os.system(get_point_bash)
    wait = wi('Waiting for rgb')
    i = 0
    while os.path.getctime(rgb_path) == ctime_stamp:
        wait.step(0.01)
        i += 1
    wait.end()


def get_real_data():
    point_data = np.load(sensory_pc_path) # (<191000, 3) shape is not constant
    # np.save(empty_bin, point_data)

    point_data=refine_point_cloud(point_data)

    np.save(sensory_pc_path, point_data)

    full_point_clouds = apply_mask(point_data)


    point_data_choice=random_down_sampling(full_point_clouds,config.num_points)

    down_sampled_point_clouds = point_data_choice[:, :3]
    # view_npy_open3d(new_point_data,view_coordinate=True)

    return down_sampled_point_clouds,full_point_clouds


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

