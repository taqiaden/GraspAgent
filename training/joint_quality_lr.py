import torch
from colorama import Fore
from scipy import spatial
from torch import nn
from torch.utils import data
from Configurations.config import theta_cos_scope, workers
from Configurations.dynamic_config import save_key, get_float
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker, suction_grasp_tracker
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.joint_quality_dl import JointQualityDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import   custom_print
from lib.dataset_utils import online_data
from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from lib.math_utils import angle_between_vectors_cross
from lib.optimizer import exponential_decay_lr_
from models.joint_grasp_sampler import GraspSampler
from models.joint_quality_networks import JointQualityNet
from pose_object import  pose_7_to_transformation
from registration import camera
import torch.nn.functional as F
from lib.report_utils import  progress_indicator
from filelock import FileLock
from joint_grasp_sampler_tr import module_key as grasp_sampler_key
from visualiztion import view_npy_open3d, view_suction_zone
from analytical_suction_sampler import estimate_suction_direction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

lock = FileLock("file.lock")

module_key='joint_quality_networks'
training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

print=custom_print
max_lr=0.01
min_lr=1*1e-6

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)


def train(batch_size=1,n_samples=None,epochs=1,learning_rate=5e-5):
    '''load  models'''
    quality_net=ModelWrapper(model=JointQualityNet,module_key=module_key)
    quality_net.ini_model(train=True)

    grasp_sampler = ModelWrapper(model=GraspSampler,module_key=grasp_sampler_key)
    grasp_sampler.ini_model(train=False)

    '''optimizers'''
    quality_net.ini_adam_optimizer(learning_rate=learning_rate)

    '''dataloader'''
    file_ids=sample_positive_buffer(size=n_samples, dict_name=suction_grasp_tracker)
    print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
    dataset = JointQualityDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.

        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))

        for i, batch in enumerate(data_loader, 0):

            depth, normals,scores,pixel_index= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            label_normals = normals.cuda().float()
            scores = scores.cuda().float()
            b = depth.shape[0]

            '''generate normals'''
            '''generate grasps'''
            with torch.no_grad():
                generated_grasps, generated_normals= grasp_sampler.model(depth.clone())

            '''process label'''
            label_generated_grasps = generated_grasps.clone()
            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                generated_normals[j, :, pix_A, pix_B] = label_normals[j]

            '''evaluate suction'''
            suction_zone_radius=0.012
            density_factor=30000*suction_zone_radius

            for j in range(b):
                '''get parameters'''
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                normals = estimate_suction_direction(pc, view=False)
                depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()
                target_point = pixel_to_point(np.array([pix_A, pix_B]), depth_value, camera)
                target_point=transform_to_camera_frame(target_point[None, :], reverse=True)[0]
                spatial_mask = estimate_object_mask(pc)

                '''mask suction region'''
                dist_=np.linalg.norm(target_point[np.newaxis] - pc,axis=-1)
                target_point_index=np.argmin(dist_)
                assert np.linalg.norm(pc[target_point_index]-target_point)<0.001,f'{target_point}, {pc[target_point_index]}'
                dist_mask = dist_<suction_zone_radius

                '''region normals'''
                region_normals=normals[dist_mask]
                average_region_normal=np.mean(region_normals,axis=0)
                normals_deviation=np.std(region_normals,axis=0)
                target_normal=normals[target_point_index]
                print(f'Normals average = {average_region_normal}')
                print(f'target normal vector = {target_normal}')
                print(f'Normals std = {normals_deviation}')

                print(f'points density = {dist_mask.sum()/density_factor}')
                angle_radians, angle_degrees= angle_between_vectors_cross(average_region_normal, target_normal)
                angle_difference=angle_degrees
                print(f'Angle difference between normals = {angle_difference}')


                '''view suction region'''
                colors = np.zeros_like(pc)
                colors[spatial_mask, 0] += 1.
                colors[dist_mask, 1] += 1.
                view_suction_zone(target_point, pc, colors)
                print('----------------------------------------------------------')


            # exit()



            pi.step(i)
        pi.end()

        quality_net.export_model()
        quality_net.export_optimizer()

        '''update performance indicator'''
        # performance_indicator= 1 - max(collision_times, out_of_scope_times) / (size)
        # save_key("performance_indicator", performance_indicator, section=module_key)

if __name__ == "__main__":
    for i in range(10000):
        '''get adaptive lr'''
        # performance_indicator = get_float("performance_indicator", section=module_key,default='0')
        # adaptive_lr = exponential_decay_lr_(performance_indicator, max_lr, min_lr)
        # print(Fore.CYAN, f'performance_indicator = {performance_indicator}, Learning rate = {adaptive_lr}', Fore.RESET)
        train(batch_size=1,n_samples=100,epochs=1,learning_rate=5e-4)