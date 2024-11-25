
import torch
from colorama import Fore
from torch import nn
from torch.utils import data
from Configurations.config import  workers
from Online_data_audit.data_tracker import sample_positive_buffer, suction_grasp_tracker, sample_random_buffer
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.joint_quality_dl import JointQualityDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import   custom_print
from lib.dataset_utils import online_data
from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from models.joint_grasp_sampler import GraspSampler
from models.joint_quality_networks import JointQualityNet
from records.training_satatistics import TrainingTracker
from registration import camera
from lib.report_utils import  progress_indicator
from filelock import FileLock
from training.joint_grasp_sampler_tr import module_key as grasp_sampler_key
from training.joint_quality_lr import normals_check, deflection_check, seal_check, curvature_check
from visualiztion import view_npy_open3d, view_suction_zone, view_score2
from analytical_suction_sampler import estimate_suction_direction
import numpy as np

lock = FileLock("file.lock")
instances_per_sample=1

module_key='joint_quality_networks'
training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

print=custom_print
max_lr=0.01
min_lr=1*1e-6

suction_zone_radius = 0.012
curvature_radius = 0.0025
curvature_deviation_threshold = 0.0025
angle_threshold_degree = 5.0
seal_ring_deviation = 0.002
suction_area_deflection = 0.005

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

def view_scores(pc,scores,threshold=0.5):
    view_scores = scores.detach().cpu().numpy()
    view_scores[view_scores < threshold] *= 0.0
    view_score2(pc, view_scores)



def view_suction_area(pc,dist_mask,target_point,direction,spatial_mask):
    colors = np.zeros_like(pc)
    colors[spatial_mask, 0] += 1.
    colors[dist_mask, 1] += 1.
    view_suction_zone(target_point,direction, pc, colors)

def train(batch_size=1,n_samples=None,epochs=1,learning_rate=5e-5):
    '''load  models'''
    quality_net=ModelWrapper(model=JointQualityNet,module_key=module_key)
    quality_net.ini_model(train=True)

    grasp_sampler =GANWrapper(module_key=grasp_sampler_key,generator=GraspSampler)
    grasp_sampler.ini_generator(train=False)


    '''dataloader'''
    file_ids=sample_random_buffer(size=n_samples, dict_name=suction_grasp_tracker)
    print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
    dataset = JointQualityDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    statistics = TrainingTracker(name=module_key, iterations_per_epoch=len(data_loader), samples_size=len(dataset))

    for epoch in range(epochs):

        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))

        for i, batch in enumerate(data_loader, 0):

            depth, normals,scores,pixel_index= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            b = depth.shape[0]


            '''generate grasps'''
            with torch.no_grad():
                generated_grasps, generated_normals= grasp_sampler.generator(depth.clone())

                gripper_score,suction_score,shift_score=quality_net.model(depth.clone(), generated_grasps.clone(),generated_normals.clone())

            '''evaluate suction'''
            for j in range(b):
                '''get parameters'''
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)

                pc = transform_to_camera_frame(pc, reverse=True)
                normals = estimate_suction_direction(pc, view=False)
                spatial_mask = estimate_object_mask(pc)

                # exit()
                '''view suction scores'''
                view_scores(pc, suction_score[j, 0][mask],threshold=0.0)
                for k in range(instances_per_sample):
                    '''pick random pixels'''
                    while True:
                        if k==0:
                            pix_A = pixel_index[j, 0]
                            pix_B = pixel_index[j, 1]
                            break
                        pix_A = np.random.randint(40, 400)
                        pix_B = np.random.randint(40, 632)
                        max_score = torch.max(suction_score).item()
                        range_ = (max_score - torch.min(suction_score)).item()

                        selection_probability = 1 - (max_score - suction_score[j, 0, pix_A, pix_B]).item() / range_
                        selection_probability = selection_probability ** 2

                        if mask[pix_A, pix_B] == 1 and  np.random.random() < selection_probability: break

                    depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()
                    target_point = pixel_to_point(np.array([pix_A, pix_B]), depth_value, camera)
                    target_point=transform_to_camera_frame(target_point[None, :], reverse=True)[0]

                    '''mask suction region'''
                    dist_=np.linalg.norm(target_point[np.newaxis] - pc,axis=-1)
                    target_point_index=np.argmin(dist_)
                    assert np.linalg.norm(pc[target_point_index]-target_point)<0.001,f'{target_point}, {pc[target_point_index]}'
                    dist_mask = dist_<suction_zone_radius

                    target_normal = normals[target_point_index]
                    '''circle to points'''
                    points_at_seal_region=pc[dist_mask]

                    '''suction criteria'''
                    first_criteria=normals_check(normals, dist_mask, target_normal)
                    second_criteria=curvature_check(points_at_seal_region)
                    third_criteria=deflection_check(target_normal,points_at_seal_region)
                    fourth_criteria=seal_check(target_point,points_at_seal_region)

                    '''suction seal loss'''
                    prediction_=suction_score[j, 0, pix_A, pix_B]

                    if first_criteria and second_criteria and third_criteria and fourth_criteria:
                        label = torch.ones_like(prediction_)
                    else:
                        label = torch.zeros_like(prediction_)

                    # '''view suction region'''
                    # print('----------------------------------------')
                    # print(f'label= {label}, prediction= {prediction_}')
                    # if scores[j]!=label.item():
                    #
                    #     view_suction_area(pc,dist_mask,target_point,target_normal,spatial_mask)

                    statistics.update_confession_matrix(label,prediction_)



            pi.step(i)
        pi.end()


    statistics.print()


if __name__ == "__main__":
    for i in range(10000):
        train(batch_size=1,n_samples=300,epochs=1)