import os
import numpy as np

use_xyz= True

home_dir = '/home/shenxiaofei/'
solution_name='Never_stop_learning'
ip_address=r'\\10.5.12.167'

check_points_extension='.pth.tar'

check_points_directory=ip_address+r'/taqiaden_hub/NSL_model_state/'
# check_points_directory=r'/media/shenxiaofei/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/taqiaden_hub/NSL_model_state/'
# check_points_directory=r'/media/shenxiaofei/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/Never_stop_learning/check_points/'

while os.path.split(os.getcwd())[-1]!=solution_name:
    os.chdir('../')

theta_scope=90. # previous scope 60.
theta_scope_rad=(theta_scope/180)*np.pi
theta_cos_scope=np.cos(theta_scope_rad)

phi_scope=360.
phi_scope_rad=(phi_scope/180)*np.pi
beta_scope=180.
beta_scope_rad=(beta_scope/180)*np.pi

distance_scope=0.05
width_scope= 0.05

#dataset and dataloader
num_points= 50000*1 #16384

hide_smbclient_log=True

# robot constants
width_scale=550

shift_length=0.1
