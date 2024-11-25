
import torch
from colorama import Fore
from torch import nn
from torch.utils import data
from Configurations.config import  workers
from Online_data_audit.data_tracker import sample_positive_buffer, suction_grasp_tracker
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.joint_quality_dl import JointQualityDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import   custom_print
from lib.dataset_utils import online_data
from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from lib.loss.D_loss import binary_l1
from lib.math_utils import angle_between_vectors_cross, rotation_matrix_from_vectors
from lib.pc_utils import circle_to_points, compute_curvature, numpy_to_o3d
from models.joint_grasp_sampler import GraspSampler
from models.joint_quality_networks import JointQualityNet
from records.training_satatistics import TrainingTracker
from registration import camera
from lib.report_utils import  progress_indicator
from filelock import FileLock
from training.joint_grasp_sampler_tr import module_key as grasp_sampler_key
from visualiztion import view_npy_open3d, view_suction_zone, view_score2
from analytical_suction_sampler import estimate_suction_direction
import numpy as np

lock = FileLock("file.lock")
instances_per_sample=2

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

def normals_check(normals,dist_mask,target_normal):
    '''region normals'''
    region_normals = normals[dist_mask]
    average_region_normal = np.mean(region_normals, axis=0)

    angle_radians, angle_degrees = angle_between_vectors_cross(average_region_normal, target_normal)
    # if angle_degrees < angle_threshold_degree:
    #     print(f'Angle difference between normals = {angle_degrees}')
    #
    # else:
    #     print(Fore.RED, f'Angle difference between normals = {angle_degrees}', Fore.RESET)

    return angle_degrees < angle_threshold_degree

def curvature_check(points_at_seal_region):
    curvature = compute_curvature(points_at_seal_region, radius=curvature_radius)
    curvature = np.array(curvature)
    curvature_std = curvature.std()
    # if curvature_std < curvature_deviation_threshold:
    #     print(f'curvature deviation= {curvature_std}')
    # else:
    #     print(Fore.RED, f'curvature deviation= {curvature_std}', Fore.RESET)

    return curvature_std < curvature_deviation_threshold


def deflection_check(target_normal,points_at_seal_region):
    R = rotation_matrix_from_vectors(target_normal, np.array([0, 0, 1]))
    transformed_points_at_seal_region = np.matmul(R, points_at_seal_region.T).T
    seal_deflection = np.max(transformed_points_at_seal_region[:, 2]) - np.min(transformed_points_at_seal_region[:, 2])
    # if seal_deflection < suction_area_deflection:
    #     print(f'seal deflection = {seal_deflection}')
    # else:
    #     print(Fore.RED, f'seal deflection = {seal_deflection}', Fore.RESET)

    return seal_deflection < suction_area_deflection

def seal_check(target_point,points_at_seal_region):
    seal_test_points = circle_to_points(radius=suction_zone_radius, number_of_points=100, x=target_point[0],
                                        y=target_point[1], z=target_point[2])

    xy_dist = np.linalg.norm(seal_test_points[:, np.newaxis, 0:2] - points_at_seal_region[np.newaxis, :, 0:2], axis=-1)
    min_xy_dist = np.min(xy_dist, axis=1)
    seal_deviation = np.max(min_xy_dist)
    # if seal_deviation < seal_ring_deviation:
    #     print(f'maximum seal deviation = {seal_deviation}')
    # else:
    #     print(Fore.RED, f'maximum seal deviation = {seal_deviation}', Fore.RESET)

    return seal_deviation < seal_ring_deviation

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

    '''optimizers'''
    quality_net.ini_adam_optimizer(learning_rate=learning_rate)

    '''dataloader'''
    file_ids=sample_positive_buffer(size=n_samples, dict_name=suction_grasp_tracker)
    print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
    dataset = JointQualityDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    statistics = TrainingTracker(name=module_key, iterations_per_epoch=len(data_loader), samples_size=len(dataset))

    for epoch in range(epochs):

        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))

        for i, batch in enumerate(data_loader, 0):

            depth, normals,scores,pixel_index= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            label_normals = normals.cuda().float()
            scores = scores.cuda().float()
            b = depth.shape[0]

            quality_net.model.zero_grad()
            quality_net.optimizer.zero_grad()

            '''generate grasps'''
            with torch.no_grad():
                generated_grasps, generated_normals= grasp_sampler.generator(depth.clone())
            '''process label'''
            # for j in range(b):
            #     pix_A = pixel_index[j, 0]
            #     pix_B = pixel_index[j, 1]
            #     generated_normals[j, :, pix_A, pix_B] = label_normals[j]

            gripper_score,suction_score,shift_score=quality_net.model(depth.clone(), generated_grasps.clone(),generated_normals.clone())

            '''evaluate suction'''
            suction_loss=torch.tensor(0.0,device=gripper_score.device)

            for j in range(b):
                '''get parameters'''
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)

                pc = transform_to_camera_frame(pc, reverse=True)
                normals = estimate_suction_direction(pc, view=False)
                spatial_mask = estimate_object_mask(pc)

                # exit()
                '''view suction scores'''
                # view_scores(pc, suction_score[j, 0][mask],threshold=0.5)
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
                        selection_probability = max(selection_probability ** 5,0,0.05)

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

                    '''view suction region'''
                    # print(f'label= {label}, prediction= {prediction_}')
                    # view_suction_area(pc,dist_mask,target_point,target_normal,spatial_mask)

                    statistics.update_confession_matrix(label,prediction_)

                    suction_loss+=binary_l1(prediction_, label)
                    suction_loss+=binary_l1(suction_score[j, 0][mask],torch.zeros_like(suction_score[j, 0][mask])).mean()*0.1

            loss=suction_loss
            if loss.item()>0:
                loss.backward()
                quality_net.optimizer.step()

            statistics.running_loss += loss.item()

            pi.step(i)
        pi.end()



        quality_net.export_model()
        quality_net.export_optimizer()

    statistics.print()

    '''update performance indicator'''
    # performance_indicator= 1 - max(collision_times, out_of_scope_times) / (size)
    # save_key("performance_indicator", performance_indicator, section=module_key)

if __name__ == "__main__":
    for i in range(10000):
        '''get adaptive lr'''
        # performance_indicator = get_float("performance_indicator", section=module_key,default='0')
        # adaptive_lr = exponential_decay_lr_(performance_indicator, max_lr, min_lr)
        # print(Fore.CYAN, f'performance_indicator = {performance_indicator}, Learning rate = {adaptive_lr}', Fore.RESET)
        train(batch_size=1,n_samples=300,epochs=1,learning_rate=1e-5)