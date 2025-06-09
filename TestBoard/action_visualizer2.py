import numpy as np
import torch
from colorama import Fore
from filelock import FileLock
from Configurations.config import workers
from check_points.check_point_conventions import  ModelWrapper
from dataloaders.action_dl import  ActionDataset2
from lib.IO_utils import custom_print
from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.image_utils import view_image
from lib.math_utils import seeds
from lib.report_utils import progress_indicator
from models.action_net import ActionNet,  action_module_key
from registration import camera
from visualiztion import dense_grasps_visualization
from visualiztion import view_score2

detach_backbone=False

lock = FileLock("file.lock")

training_buffer = online_data2()
training_buffer.main_modality=training_buffer.depth

print=custom_print

def view_scores(pc,scores,threshold=0.5):
    view_scores = scores.detach().cpu().numpy()
    view_scores[view_scores < threshold] *= 0.0
    view_score2(pc, view_scores)

class TrainActionNet:
    def __init__(self,n_samples=None,epochs=1,learning_rate=5e-5):
        self.n_samples=n_samples
        self.size=n_samples
        self.epochs=epochs
        self.learning_rate=learning_rate

        '''model wrapper'''
        self.actions_net=self.prepare_model_wrapper()
        self.data_loader=None

    def initialize(self,n_samples=None):
        self.n_samples=n_samples
        self.prepare_data_loader()

    def prepare_data_loader(self):
        file_ids=training_buffer.get_indexes()
        print(Fore.CYAN, f'Buffer size = {len(file_ids)}',Fore.RESET)
        dataset = ActionDataset2(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers,
                                                       shuffle=True)
        self.size=len(dataset)
        self.data_loader= data_loader

    def prepare_model_wrapper(self):
        '''load  models'''
        actions_net = ModelWrapper(model=ActionNet(), module_key=action_module_key)
        actions_net.ini_model(train=False)
        return actions_net

    def simulate_elevation_variations(self,original_depth,seed,max_elevation=0.2,exponent=2.0):
        with torch.no_grad():
            _, _, _, _, _, background_class_3= self.actions_net.model(
                original_depth.clone())

            '''Elevation-based Augmentation'''
            objects_mask = background_class_3 <= 0.5
            shift_entities_mask = objects_mask & (original_depth > 0.0001)
            new_depth = original_depth.clone().detach()
            new_depth[shift_entities_mask] -= max_elevation * (np.random.rand()**exponent) * camera.scale

            return new_depth

    def begin(self):

        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        gripper_pose=None
        for i, batch in enumerate(self.data_loader, 0):

            depth,file_ids= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pi.step(i)

            seed=np.random.randint(0,5000)
            '''Elevation-based augmentation'''
            depth=self.simulate_elevation_variations(depth,seed,exponent=5.0)

            '''get parameters'''
            pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)

            latent_vector=torch.randn(size=(depth.numel(),8),device=depth.device)

            '''generated grasps'''
            gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier,background_class= self.actions_net.model(
                depth.clone(),detach_backbone=detach_backbone)

            # print(self.gan.generator.gripper_sampler.dist_biased_tanh.b.data)
            # # print(self.gan.generator.gripper_sampler.dist_biased_tanh.k.data)
            #
            # print(self.gan.generator.gripper_sampler.width_biased_tanh.b.data)
            # # print(self.gan.generator.gripper_sampler.width_biased_tanh.k.data)
            #
            # exit()
            # bin_mask = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=True,
            #                                 file_index=file_ids[0], cache_name='bin_planes2')

            suction_head_predictions=suction_quality_classifier[0, 0][mask]
            shift_head_predictions = shift_affordance_classifier[0, 0][mask]
            background_class_predictions = background_class.permute(0, 2, 3, 1)[0, :, :, 0][mask]
            objects_mask = background_class_predictions.detach().cpu().numpy() <= .5
            gripper_poses=gripper_pose[0].permute(1,2,0)[mask]#.cpu().numpy()
            collision_with_objects_predictions=griper_collision_classifier[0, 0][mask]
            collision_with_bin_predictions=griper_collision_classifier[0, 1][mask]
            gripper_sampling_mask=(collision_with_objects_predictions<.7) & (collision_with_bin_predictions<.7)

            # print(pc[objects_mask].min())
            # print(pc[~objects_mask].min())
            # print(pc[objects_mask].max())
            # print(pc[~objects_mask].max())
            # exit()
            # view_image(depth[0,0].cpu().numpy().astype(np.float64))

            # view_image(background_class[0, 0].detach().cpu().numpy().astype(np.float64))
            sampling_p=1.-torch.sqrt(collision_with_objects_predictions*collision_with_bin_predictions)

            dense_grasps_visualization(pc, gripper_poses, view_mask=gripper_sampling_mask&torch.from_numpy(objects_mask).cuda(),sampling_p=sampling_p,view_all=False)

            # gripper_poses[:,3:5]*=-1
            # dense_grasps_visualization(pc, gripper_poses, view_mask=gripper_sampling_mask&torch.from_numpy(objects_mask).cuda(),sampling_p=sampling_p,view_all=False)

            suction_head_predictions[~torch.from_numpy(objects_mask).cuda()]*=0.

            # view_scores(pc, background_class_predictions, threshold=0.5)
            # view_scores(pc, suction_head_predictions, threshold=0.5)
            # view_scores(pc, shift_head_predictions, threshold=0.5)


if __name__ == "__main__":
    seeds(1)

    with torch.no_grad():
        lr = 1e-6
        train_action_net = TrainActionNet(n_samples=None, learning_rate=lr)
        train_action_net.initialize(n_samples=100)
        train_action_net.begin()

        cuda_memory_report()
        train_action_net.initialize(n_samples=None)
        train_action_net.begin()


