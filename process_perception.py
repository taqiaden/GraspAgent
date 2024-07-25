import os
import subprocess
from colorama import Fore
from lib.report_utils import wait_indicator as wi
import numpy as np
from Configurations import config
from Run import simulation_mode
from dataset.load_test_data import sensory_pc_path
import cv2 as cv

get_point_bash='./bash/get_point.sh'
texture_image_path = config.home_dir + 'texture_image.jpg'
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