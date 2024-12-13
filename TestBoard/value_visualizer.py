import random
import torch
from colorama import Fore
from torch import nn
from torch.utils import data
from Configurations.config import workers
from Online_data_audit.data_tracker import  gripper_grasp_tracker, sample_all_positive_and_negatives, suction_grasp_tracker
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.value_dl import ValueDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import   custom_print
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from models.action_net import ActionNet
from models.value_net import ValueNet
from registration import camera
from lib.report_utils import  progress_indicator
from training.action_lr import module_key as action_module_key
from visualiztion import view_score2

instances_per_sample=1

module_key='value_net'
training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth
normal_weight = 10

print=custom_print

mes_loss=nn.MSELoss()

def view_scores(pc,scores,threshold=0.5):
    view_scores = scores.detach().cpu().numpy()
    view_scores[view_scores < threshold] *= 0.0
    view_score2(pc, view_scores)

class TrainValueNet:
    def __init__(self):
        self.size=None

        '''model wrapper'''
        self.value_net=self.value_model_wrapper()
        self.action_net=self.action_model_wrapper()
        self.data_loader=self.prepare_data_loader()

    def prepare_data_loader(self):
        gripper_positive_labels,gripper_negative_labels=sample_all_positive_and_negatives(gripper_grasp_tracker, shuffle=True, disregard_collision_samples=True)
        suction_positive_labels,suction_negative_labels=sample_all_positive_and_negatives(suction_grasp_tracker, shuffle=True, disregard_collision_samples=False)
        sampling_size=min(len(gripper_positive_labels),len(gripper_negative_labels),len(suction_positive_labels),len(suction_negative_labels))
        assert sampling_size>1

        file_ids = (gripper_positive_labels[0:sampling_size]+gripper_negative_labels[0:sampling_size]
                    +suction_positive_labels[0:sampling_size]+suction_negative_labels[0:sampling_size])
        random.shuffle(file_ids)

        print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
        dataset = ValueDataset(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers,
                                                       shuffle=True)
        self.size=len(dataset)
        return data_loader

    def value_model_wrapper(self):
        '''load  models'''
        value_net = ModelWrapper(model=ValueNet,module_key=module_key)
        value_net.ini_model(train=True)
        return value_net

    def action_model_wrapper(self):
        '''load  models'''
        action_net = GANWrapper(action_module_key, ActionNet)
        action_net.ini_generator(train=False)
        return action_net

    def begin(self):
        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        for i, batch in enumerate(self.data_loader, 0):
            depth,pose_7,pixel_index,score,normal,is_gripper,random_rgb= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pose_7 = pose_7.cuda().float()
            score = score.cuda().float()
            normal = normal.cuda().float()
            is_gripper = is_gripper.cuda().float()
            random_rgb = random_rgb.cuda().float().permute(0,3,1,2)

            b = depth.shape[0]

            '''zero grad'''
            self.value_net.model.zero_grad()

            '''generated grasps'''
            with torch.no_grad():
                gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier \
                    ,depth_features= self.action_net.generator(depth.clone())
                '''process label'''
                for j in range(b):
                    pix_A = pixel_index[j, 0]
                    pix_B = pixel_index[j, 1]
                    gripper_pose[j, :, pix_A, pix_B] = pose_7[j]
                    suction_direction[j, :, pix_A, pix_B] = normal[j]

                griper_grasp_score,suction_grasp_score=self.value_net.model(random_rgb,depth_features,gripper_pose,suction_direction)

            '''accumulate loss'''
            for j in range(b):
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                spatial_mask = estimate_object_mask(pc, custom_margin=0.01)
                gripper_mask=spatial_mask & (griper_collision_classifier[j,0,mask]>0.5).cpu().numpy()
                scores=griper_grasp_score[j,0,mask]
                scores[~gripper_mask]*=0.
                print('Gripper-------------------')
                view_scores(pc, scores)
                print('Suction-------------------')
                suction_mask=spatial_mask & (suction_quality_classifier[j,0,mask]>0.5).cpu().numpy()
                scores=suction_grasp_score[j,0,mask]
                scores[~suction_mask]*=0.
                view_scores(pc, scores)

            pi.step(i)
        pi.end()

if __name__ == "__main__":
    train_value_net = TrainValueNet()
    train_value_net.begin()
    for i in range(10000):
        try:
            cuda_memory_report()
            train_value_net=TrainValueNet()
            train_value_net.begin()
        except Exception as error_message:
            print(error_message)