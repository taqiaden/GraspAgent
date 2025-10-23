import copy
import os
import torch.nn.functional as F
import numpy as np
from colorama import Fore
from filelock import FileLock
from torch import nn
import torch.nn.functional as F
from Configurations.config import workers
from Online_data_audit.data_tracker import  gripper_grasp_tracker, DataTracker
from analytical_suction_sampler import estimate_suction_direction
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.Grasp_GAN_dl import GraspGANDataset2
from lib.IO_utils import custom_print, load_pickle, save_pickle
from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection, cache_dir
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.loss.balanced_bce_loss import BalancedBCELoss
from lib.models_utils import reshape_for_layer_norm
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from models.Grasp_GAN import gripper_sampling_module_key, G, D, N_gripper_sampling_module_key
from models.Grasp_handover_policy_net import GraspHandoverPolicyNet, grasp_handover_policy_module_key
from models.action_net import ActionNet, action_module_key
# from models.scope_net import scope_net_vanilla, gripper_scope_module_key
from records.training_satatistics import TrainingTracker, MovingRate
from registration import camera
from training.learning_objectives.gripper_collision import  evaluate_grasps3
from training.suction_scope_training import weight_decay
from visualiztion import view_features, plt_features, dense_grasps_visualization, view_npy_open3d
from lib.math_utils import seeds

detach_backbone = False
lock = FileLock("file.lock")
max_samples_per_image=1

max_n = 100
batch_size = 8
# max_batch_size = 2
# G_batch_size=4

key=  gripper_sampling_module_key

w2=0

training_buffer = online_data2()
training_buffer.main_modality = training_buffer.depth

bce_loss = nn.BCELoss()

balanced_bce_loss = BalancedBCELoss()
print = custom_print

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
cache_name = 'normals'
discrepancy_distance = 1.0

collision_expo=1.0
firmness_expo=1.0
generator_expo=1.0

import torch

def get_normal_direction(pc, file_index):
    file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
    if os.path.exists(file_path):
        normals = load_pickle(file_path)
        if  normals.shape[0]!=pc.shape[0]:
            normals = estimate_suction_direction(pc,
                                                 view=False)  # inference time on local computer = 1.3 s        if file_index is not None:
            file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
            save_pickle(file_path, normals)
    else:
        normals = estimate_suction_direction(pc,
                                            view=False)  # inference time on local computer = 1.3 s        if file_index is not None:
        file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
        save_pickle(file_path, normals)

    normals = torch.from_numpy(normals).to('cuda')
    return normals


class TrainGraspGAN:
    def __init__(self, n_samples=None, epochs=1, learning_rate=5e-5):
        self.n_samples = n_samples
        self.size = n_samples
        self.epochs = epochs
        self.learning_rate = learning_rate

        '''model wrapper'''
        self.gan = self.prepare_model_wrapper()
        # self.ref_generator = self.initialize_ref_generator()
        self.data_loader = None

        '''Moving rates'''
        self.moving_collision_rate = None
        self.moving_firmness = None
        self.moving_out_of_scope = None
        self.relative_sampling_timing = None
        self.superior_A_model_moving_rate = None
        self.moving_anneling_factor = None

        '''initialize statistics records'''
        self.gripper_sampler_statistics = None
        self.critic_statistics = None
        self.data_tracker = None


        self.sampling_centroid = None
        self.diversity_momentum = 1.0

        self.sample_from_latent=True

        self.action_net = None
        # self.policy_net=None
        # self.load_action_model()
        # self.load_grasp_policy()

        self.freeze_approach = False
        self.freeze_beta = False
        self.freeze_distance = False
        self.freeze_width = False


    def load_action_model(self):
        try:
            '''load  models'''
            actions_net = ModelWrapper(model=ActionNet(), module_key=action_module_key)
            actions_net.ini_model(train=False)
            self.action_net = actions_net.model
        except Exception as e:
            print(str(e))


    def initialize(self, n_samples=None):
        self.n_samples = n_samples
        self.prepare_data_loader()

        '''Moving rates'''
        self.moving_collision_rate = MovingRate(key + '_collision', decay_rate=0.01, initial_val=1.)
        self.moving_firmness = MovingRate(key + '_firmness', decay_rate=0.01, initial_val=0.)
        self.moving_out_of_scope = MovingRate(key + '_out_of_scope', decay_rate=0.01, initial_val=1.)
        self.relative_sampling_timing = MovingRate(key + '_relative_sampling_timing', decay_rate=0.01,
                                                   initial_val=1.)
        self.superior_A_model_moving_rate=MovingRate(key + '_superior_A_model', decay_rate=0.01,
                                                   initial_val=0.)
        self.moving_anneling_factor = MovingRate(key + '_anneling_factor', decay_rate=0.1,
                                                 initial_val=0.)


        '''initialize statistics records'''


        self.gripper_sampler_statistics = TrainingTracker(name=key + '_gripper_sampler',
                                                          iterations_per_epoch=len(self.data_loader),
                                                          track_label_balance=False)

        self.critic_statistics = TrainingTracker(name=key + '_critic',
                                                 iterations_per_epoch=len(self.data_loader), track_label_balance=False)

        self.data_tracker = DataTracker(name=gripper_grasp_tracker)

        # gripper_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=gripper_scope_module_key)
        # gripper_scope.ini_model(train=False)
        # self.gripper_arm_reachability_net = gripper_scope.model

    def prepare_data_loader(self):
        file_ids = training_buffer.get_indexes()
        # file_ids = sample_positive_buffer(size=self.n_samples, dict_name=gripper_grasp_tracker,
        #                                   disregard_collision_samples=True,sample_with_probability=False)
        print(Fore.CYAN, f'Buffer size = {len(file_ids)}', Fore.RESET)
        dataset = GraspGANDataset2(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers,
                                                  shuffle=True)
        self.size = len(dataset)
        self.data_loader = data_loader

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(key, G, D)
        gan.ini_models(train=False)
        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate*10,momentum=0.,weight_decay_=0.)
        # gan.generator_sgd_optimizer(learning_rate=self.learning_rate,momentum=0.0)
        gan.critic_adam_optimizer(learning_rate=self.learning_rate,beta1=0.5,weight_decay_=0.0)
        gan.generator_adam_optimizer(learning_rate=self.learning_rate, beta1=0.9)

        return gan

    def simulate_elevation_variations(self, original_depth,objects_mask, max_elevation=0.15, exponent=2.0):
        '''Elevation-based Augmentation'''
        shift_entities_mask = objects_mask & (original_depth > 0.0001) if np.random.random()>0.5 else (original_depth > 0.0001)
        new_depth = original_depth.clone().detach()
        new_depth[shift_entities_mask] -= max_elevation * (np.random.rand() ** exponent) * camera.scale

        return new_depth

    def analytical_bin_mask(self, pc, file_ids):
        try:
            bin_mask,floor_elevation = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=False,
                                            file_index=file_ids[0], cache_name='bin_planes2')
            return bin_mask, floor_elevation
        except Exception as error_message:
            print('bin mask generation error:',file_ids[0])
            print(error_message)
            return None, None



    def ref_reg_loss(self,gripper_pose_ref2,objects_mask):
        width_scope_loss = (torch.clamp(gripper_pose_ref2[:, 6:7][objects_mask] - 1, min=0.) ** 2).mean()
        dist_scope_loss = (torch.clamp(-gripper_pose_ref2[:, 5:6][objects_mask] + 0.01, min=0.) ** 2).mean()
        width_scope_loss += (torch.clamp(-gripper_pose_ref2[:, 6:7][objects_mask], min=0.) ** 2).mean()
        dist_scope_loss += (torch.clamp(gripper_pose_ref2[:, 5:6][objects_mask] - 1, min=0.) ** 2).mean()


        # beta_entropy = self.soft_entropy_loss(gripper_pose_ref2[:, 3:4][objects_mask], bins=36, sigma=0.1, min_val=-1,
        #                                       max_val=1) ** 2
        # beta_entropy += self.soft_entropy_loss(gripper_pose_ref2[:, 4:5][objects_mask], bins=36, sigma=0.1, min_val=-1,
        #                                        max_val=1) ** 2

        beta_std=0.5-((gripper_pose_ref2[:, 3:4][objects_mask]+1)/2).std()
        beta_std+=0.5-((gripper_pose_ref2[:, 4:5][objects_mask]+1)/2).std()

        # beta_entropy = torch.tensor(0., device=gripper_pose_ref2.device) if torch.isnan(beta_entropy) else beta_entropy

        loss =  width_scope_loss * 100 + dist_scope_loss * 100  + (beta_std**2)

        return loss


    def pixel_to_point_index(self,depth,mask,gripper_pixel_index):
        tmp = torch.zeros_like(depth)[0, 0]
        tmp[gripper_pixel_index[0], gripper_pixel_index[1]] = 1.0
        tmp = tmp[mask]

        point_index = (tmp == 1).nonzero(as_tuple=False)
        return point_index.item()

    def pick_trainable_grasp_parameters(self):
        if np.random.random()>0.5:
            v_ = np.random.random()
            self.freeze_approach = False if v_ < 0.35 else True
            self.freeze_beta = False if 0.75 > v_ > 0.35 else True
            self.freeze_width = False if v_ > 0.75 else True
        else:
            self.freeze_approach = False
            self.freeze_beta = False
            self.freeze_width = False

        print(Fore.CYAN,
              f'freeze_approach={self.freeze_approach}, freeze_beta={self.freeze_beta}, freeze_width={self.freeze_width}',
              Fore.RESET)

    def get_scope_loss(self,pose):
        dist_scope_loss = (torch.clamp(pose[-2]- 0.99, min=0.) ** 2)
        width_scope_loss = (torch.clamp(pose[-1] - 0.99, min=0.) ** 2)
        dist_scope_loss += (torch.clamp(-pose[-2] + 0.01, min=0.) ** 2)
        width_scope_loss += (torch.clamp(-pose[-1] + 0.01, min=0.) ** 2)

        return width_scope_loss+dist_scope_loss

    def begin(self):

        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        gripper_pose = None
        for i, batch in enumerate(self.data_loader, 0):
            depth, file_ids ,pose_7,valid_gripper_pose,gripper_pixel_index= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pose_7 = pose_7.cuda().float()[0]
            gripper_pixel_index=gripper_pixel_index[0]
            valid_gripper_pose=valid_gripper_pose[0]

            pose_7[-2]=pose_7[-2]*(np.random.rand()**2)
            # pose_7[-1]=pose_7[-1]+(1-pose_7[-1])*np.random.rand()

            pi.step(i)

            pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)

            '''background detection head'''
            bin_mask,floor_elevation = self.analytical_bin_mask(pc, file_ids)
            if bin_mask is None: continue
            objects_mask_numpy = bin_mask <= 0.5
            objects_mask = torch.from_numpy(objects_mask_numpy).cuda()
            objects_mask_pixel_form=torch.ones_like(depth)
            objects_mask_pixel_form[0,0][mask]=objects_mask_pixel_form[0,0][mask]*objects_mask
            objects_mask_pixel_form=objects_mask_pixel_form>0.5

            # view_features(reshape_for_layer_norm(objects_mask_pixel_form, camera=camera, reverse=False))

            '''Elevation-based augmentation'''
            if np.random.rand()>1.:
                depth=self.simulate_elevation_variations(depth,objects_mask_pixel_form,exponent=5.0)
                pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                altered_objects_elevation=True
            else:
                altered_objects_elevation=False

            n_s=max_samples_per_image if valid_gripper_pose else 1
            for k in range(n_s):
                with torch.no_grad():
                    gripper_pose = self.gan.generator(
                        depth.clone(),  detach_backbone=True) # [1,7,h,w]

                if k == 0:
                    assert torch.any(torch.isnan(gripper_pose)) == False, f'{gripper_pose}'
                    # gripper_pose[:,-2,...]*=0.
                    # print(f'i={i}, k={k}, pc shape={pc.shape}')
                    dense_grasps_visualization(pc, gripper_pose[0].permute(1, 2, 0)[mask],
                                               view_mask=objects_mask,
                                               sampling_p=None, view_all=False,exclude_collision=True)
                    break

    def view_result(self, values):
        with torch.no_grad():

            self.gripper_sampler_statistics.print()
            self.critic_statistics.print()

            # values = gripper_pose.permute(1, 0, 2, 3).flatten(1).detach()
            try:
                print(f'gripper_pose sample = {values[np.random.randint(0, values.shape[0])].cpu()}')
            except Exception as e:
                print('result view error',str(e))
            values[:, 3:5] = F.normalize(values[:, 3:5], dim=1)
            print(f'gripper_pose std = {torch.std(values, dim=0).cpu()}')
            print(f'gripper_pose mean = {torch.mean(values, dim=0).cpu()}')
            print(f'gripper_pose max = {torch.max(values, dim=0)[0].cpu()}')
            print(f'gripper_pose min = {torch.min(values, dim=0)[0].cpu()}')

            self.moving_collision_rate.view()
            self.moving_firmness.view()
            self.moving_out_of_scope.view()
            self.relative_sampling_timing.view()
            self.moving_anneling_factor.view()
            self.superior_A_model_moving_rate.view()

    def save_statistics(self):
        self.moving_collision_rate.save()
        self.moving_firmness.save()
        self.moving_out_of_scope.save()
        self.relative_sampling_timing.save()
        self.moving_anneling_factor.save()
        self.superior_A_model_moving_rate.save()

        self.critic_statistics.save()
        self.gripper_sampler_statistics.save()

        self.data_tracker.save()

    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()

    def clear(self):
        self.gripper_sampler_statistics.clear()
        self.critic_statistics.clear()

def train_N_grasp_GAN(n=1):
    lr = 5e-4
    Train_grasp_GAN = TrainGraspGAN(n_samples=None, learning_rate=lr)
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():
        cuda_memory_report()
        Train_grasp_GAN.initialize(n_samples=None)
        Train_grasp_GAN.begin()


    # del Train_grasp_GAN

if __name__ == "__main__":
    seeds(8)
    train_N_grasp_GAN(n=10000)
