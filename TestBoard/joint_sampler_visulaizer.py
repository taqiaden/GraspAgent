import torch
from colorama import Fore
from filelock import FileLock
from torch import nn
from torch.utils import data

from Configurations.config import workers
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker
from analytical_suction_sampler import estimate_suction_direction
from check_points.check_point_conventions import GANWrapper
from dataloaders.joint_grasp_sampler_dl import GraspSamplerDataset
from lib.IO_utils import custom_print
from lib.dataset_utils import online_data
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from models.joint_grasp_sampler import GraspSampler, Critic
from registration import camera
from visualiztion import view_npy_open3d

lock = FileLock("file.lock")

module_key='grasp_sampler'
training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

print=custom_print
max_lr=0.01
min_lr=1*1e-6

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

def train(batch_size=1,n_samples=None,epochs=1):
    '''load  models'''
    gan=GANWrapper(module_key,GraspSampler,Critic)
    gan.ini_models(train=True)

    '''dataloader'''
    file_ids=sample_positive_buffer(size=n_samples, dict_name=gripper_grasp_tracker)
    print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
    dataset = GraspSamplerDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    for epoch in range(epochs):


        for i, batch in enumerate(data_loader, 0):

            depth, pose_7, pixel_index = batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pose_7 = pose_7.cuda().float().squeeze(1)  # [b,7]
            b = depth.shape[0]

            '''generate grasps'''
            with torch.no_grad():
                generated_grasps,normals = gan.generator(depth.clone())

            '''process label'''
            label_generated_grasps = generated_grasps.clone()
            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                label_generated_grasps[j, :, pix_A, pix_B] = pose_7[j]

                '''generate labels'''
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)

                analytical_normals = estimate_suction_direction(pc, view=False)  # inference time on local computer = 1.3 s
                '''mask prediction'''
                masked_prediction = normals.permute(0, 2, 3, 1)[j][mask]
                '''view output'''
                view_npy_open3d(pc, normals=analytical_normals)
                predicted_normals = masked_prediction.detach().cpu().numpy()
                view_npy_open3d(pc, normals=predicted_normals)




if __name__ == "__main__":
    for i in range(10000):
        train(batch_size=1,n_samples=100,epochs=1)