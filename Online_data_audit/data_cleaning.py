import math
import os

import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt

from Configurations import config
from label_unpack import LabelObj
from lib.dataset_utils import online_data
from lib.models_utils import initialize_model_state
from lib.report_utils import progress_indicator
from lib.report_utils import progress_indicator as pi
from models.scope_net import scope_net_vanilla
from distfit import distfit

online_data = online_data()

def track_samples_scope_score():
    indexes=online_data.get_indexes()
    model = initialize_model_state(scope_net_vanilla(in_size=6), suction_scope_model_state_path)
    model.eval()

    counter=0
    total_counter=0
    for index in indexes:
        label=online_data.label.load(index)
        label_obj = LabelObj(label=label)
        if label_obj.is_gripper or label_obj.failure: continue
        normal=label_obj.normal
        approach=normal
        approach[2]=-approach[2]
        transition=label_obj.target_point


        rotation=np.empty((3,3))

        input = np.concatenate([transition, approach])
        input=torch.from_numpy(input).to('cuda')[None,...].float()

        # print(input.shape)
        score=model(input)
        total_counter+=1
        if score<0.5:
            counter+=1
            print(score.item())

    print(counter)
    print(total_counter)

def clean_old_data_redundancy():
    indexes=online_data.get_indexes()
    print(f'total samples = {len(indexes)}')

    pi = progress_indicator('progress ', max_limit=len(indexes))
    print('sort files ')
    indexes.sort()
    print()
    last_pc=None
    counters=[0,0]
    for i in range(len(indexes)):
        current_index=indexes[i]
        current_pc = online_data.point_clouds.load(current_index)

        if last_pc is not None:
            if last_pc.shape==current_pc.shape:
                counters[0]+=1
                dif=np.abs(last_pc-current_pc).sum()
                if dif==0.0:
                    counters[1]+=1
                    print(f'remove file with index {indexes[i-1]}')
                    online_data.point_clouds.remove_file(indexes[i-1])
                    online_data.label.remove_file(indexes[i-1])


        pi.step(i)

        last_pc=current_pc
    pi.end()
    print(f'similarities in shapes ={counters[0]}')
    print(f'total similarities ={counters[1]}')

def convert_all_point_clouds_to_depth():
    file_indexes = online_data.get_indexes()

    progress_indicator=pi(f'total samples size = {len(file_indexes)}, progress:  ',len(file_indexes))
    counter=0
    for i,target_file_index in enumerate(file_indexes):
        '''get data'''
        # depth=online_data.load_depth(target_file_index)
        pc = online_data.point_clouds.load_as_numpy(target_file_index)
        label_obj = LabelObj()
        depth = label_obj.get_depth(point_clouds=pc)

        online_data.depth.save_as_numpy(depth,target_file_index)

        '''update counter'''
        counter+=1
        progress_indicator.step(counter)

def rename_files():
    path=online_data.point_clouds.dir
    for filename in os.listdir(path):
        idx=online_data.get_index(filename)
        os.rename(os.path.join(path,filename),os.path.join(path,idx+online_data.point_clouds.sufix))

def check_collision_in_data():
    indexes=online_data.get_indexes()
    # indexes=sample_positive_buffer(size=None, dict_name=gripper_grasp_tracker,disregard_collision_samples=True)

    print(f'total samples = {len(indexes)}')

    pi = progress_indicator('progress ', max_limit=len(indexes))

    counters=[0,0]
    for i in range(len(indexes)):
        current_index=indexes[i]
        label = online_data.label.load_as_numpy(current_index)
        label_obj = LabelObj(label=label)
        if label_obj.failure or label_obj.is_suction: continue
        depth = online_data.depth.load_as_numpy(current_index)
        collision_state=label_obj.check_collision(depth=depth,visualize=True)

        if collision_state>0:
            counters[0]+=1
        else:
            counters[1]+=1
        pi.step(i)

    pi.end()
    print(f'instances with collision={counters[0]}')
    print(f'instances without collision={counters[1]}')
# check_collision_in_data()
def find_best_distribtuion_fit(data):
    dist = distfit()
    dist.fit_transform(data)
    print(dist.summary)

def view_log_norm_dist(data):
    shape, location, scale = scipy.stats.lognorm.fit(data)
    mu, sigma = np.log(scale), shape
    print('------------------------------------------')
    print('shape: ',shape)
    print('location: ',location)
    print('scale: ',scale)
    print('mu: ',mu)
    x=np.linspace(min(data),max(data),100)
    plt.plot(x,scipy.stats.lognorm.pdf(x,shape,location,scale),'r-',lw=2)
    plt.show()

def get_skewed_normal_distribution(data):
    print('--------------------------------')
    print('mean: ',np.mean(data))
    print('median: ',np.median(data))
    print('std: ',np.std(data))

    print('mode: ',scipy.stats.mode(data,keepdims=True)[0][0])
    print('skewness: ',scipy.stats.skew(data))
    print('kurtosis: ',scipy.stats.kurtosis(data))

'''remove selective samples'''
indexes=online_data.get_indexes()
print(f'total samples = {len(indexes)}')

pi = progress_indicator('progress ', max_limit=len(indexes))
print()
counter=0
acu_width=0
acu_dist=0
dist_list=[]
width_list=[]
for i in range(len(indexes)):
    current_index=indexes[i]
    label = online_data.label.load_as_numpy(current_index)
    label_obj = LabelObj(label=label)
    if label_obj.success and label_obj.is_gripper:
        counter+=1
        acu_width+=(label[21]-0.7)**2.
        acu_dist+=(label[22]-0.32)**2.

        width_list.append(np.copy(label[21])/ (config.width_scale*config.width_scope))
        dist_list.append(np.copy(label[22])/config.distance_scope)

        # dist mean =0.32 std =0.3
        # width mean =0.7 std =18.8
        # print(f'std dist= {math.sqrt(acu_dist/counter)}')
        # print(f'std width= {math.sqrt(acu_width/counter)}')

        pi.step(i)

get_skewed_normal_distribution(dist_list)
get_skewed_normal_distribution(width_list)

# generated_skewed_data=scipy.stats.skewnorm.rvs(0.32,loc=0.319,scale=0.159,size=10000)
# plt.hist(generated_skewed_data,bins=15,edgecolor='black')
# plt.show()
x=np.asarray(dist_list)
x[x<=0]=0.01
x=np.log(np.asarray(x))
log_mean=x.mean()
log_std=x.std()
print('dist mean',log_mean)
print('dist std',log_std)
generated_skewed_data=np.random.lognormal(mean=log_mean,sigma=log_std,size=10000)
generated_skewed_data = torch.distributions.LogNormal(loc=log_mean, scale=log_std)
generated_skewed_data = generated_skewed_data.sample((10000,)).numpy()
generated_skewed_data=generated_skewed_data[generated_skewed_data<1.0]
plt.hist(generated_skewed_data,bins=150,edgecolor='black',color='gray')
plt.xlim(0,1.)

plt.show()


x=1-np.asarray(width_list)
x[x<=0]=0.01
x=np.log(np.asarray(x))
log_mean=x.mean()
log_std=x.std()
print('width mean',log_mean)
print('width std',log_std)
generated_skewed_data=np.random.lognormal(mean=log_mean,sigma=log_std,size=10000)
generated_skewed_data=1-generated_skewed_data[generated_skewed_data<1.0]

plt.hist(generated_skewed_data,bins=150,edgecolor='black',color='gray')
plt.xlim(0,1.)
plt.show()

# generated_skewed_data=scipy.stats.skewnorm.rvs(-0.642,loc=0.697,scale=0.138,size=10000)
# plt.hist(generated_skewed_data,bins=15,edgecolor='black')
# plt.show()

plt.hist(dist_list,bins=15,edgecolor='black',color='gray')
plt.xlim(0,1.)

plt.show()

plt.hist(width_list,bins=15,edgecolor='black',color='gray')
plt.xlim(0,1.)

plt.show()

view_log_norm_dist(dist_list)
# view_log_norm_dist(width_list)

find_best_distribtuion_fit(np.asarray(dist_list))
find_best_distribtuion_fit(np.asarray(width_list))









pi.end()






