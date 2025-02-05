import time

import torch
from Configurations.config import workers
from Online_data_audit.data_tracker2 import DataTracker2
from check_points.check_point_conventions import ModelWrapper
from dataloaders.policy_dl import GraspQualityDataset
from lib.dataset_utils import online_data2
from lib.report_utils import progress_indicator
from models.policy_net import policy_module_key, PolicyNet
from records.training_satatistics import TrainingTracker
from training.policy_control_board import PPOMemory
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["TORCH_USE_CUDA_DSA"]="1"
# os.environ["PYTORCH_USE_CUDA_DSA"]="1"

buffer_file='buffer.pkl'
action_data_tracker_path=r'online_data_dict'

online_data2=online_data2()

bce_loss= torch.nn.BCELoss()

class TrainPolicyNet:
    def __init__(self,learning_rate=5e-5):

        self.learning_rate=learning_rate
        self.model_wrapper=ModelWrapper(model=PolicyNet(), module_key=policy_module_key)

        self.quality_dataloader=None

        '''initialize statistics records'''
        self.gripper_quality_net_statistics = TrainingTracker(name=policy_module_key + '_gripper_quality',
                                                              track_label_balance=True)
        self.suction_quality_net_statistics = TrainingTracker(name=policy_module_key + '_suction_quality',
                                                              track_label_balance=True)

        self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()
        self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
        self.last_tracker_size=len(self.data_tracker)

    def initialize_model(self):
        self.model_wrapper.ini_model(train=True)
        self.model_wrapper.ini_adam_optimizer(learning_rate=self.learning_rate)

    @property
    def training_trigger(self):
        return len(self.data_tracker)>self.last_tracker_size

    def synchronize_buffer(self):
        self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()
        self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)

    def experience_sampling(self,replay_size):
        replay_ids=self.data_tracker.selective_grasp_sampling(size=replay_size,sampling_rates=(self.buffer.g_p_sampling_rate.val,self.buffer.g_n_sampling_rate.val,
                                                                                    self.buffer.s_p_sampling_rate.val,self.buffer.s_n_sampling_rate.val))
        return replay_ids


    def init_quality_data_loader(self,file_ids,batch_size):
        dataset = GraspQualityDataset(data_pool=online_data2, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                                       shuffle=True)
        return  data_loader

    def step_quality_training(self,size=2,batch_size=1):
        file_ids=self.experience_sampling(size)
        data_loader=self.init_quality_data_loader(file_ids,batch_size)
        pi = progress_indicator('Begin new training round: ', max_limit=len(data_loader))
        assert size==len(file_ids)

        for i, batch in enumerate(data_loader, 0):


            rgb, mask, pose_7, gripper_pixel_index, \
                suction_pixel_index, gripper_score, \
                suction_score, normal, used_gripper, used_suction = batch

            rgb = rgb.cuda().float().permute(0, 3, 1, 2)
            mask = mask.cuda().float()
            pose_7 = pose_7.cuda().float()
            gripper_score = gripper_score.cuda().float()
            suction_score = suction_score.cuda().float()
            normal = normal.cuda().float()

            b = rgb.shape[0]
            w = rgb.shape[2]
            h = rgb.shape[3]

            '''zero grad'''
            self.model_wrapper.model.zero_grad()

            '''process pose'''
            pose_7_stack = torch.zeros((b, 7, w, h), device=rgb.device)
            normal_stack = torch.zeros((b, 3, w, h), device=rgb.device)

            for j in range(b):
                g_pix_A = gripper_pixel_index[j, 0]
                g_pix_B = gripper_pixel_index[j, 1]
                s_pix_A = suction_pixel_index[j, 0]
                s_pix_B = suction_pixel_index[j, 1]
                pose_7_stack[j, :, g_pix_A, g_pix_B] = pose_7[j]
                normal_stack[j, :, s_pix_A, s_pix_B] = normal[j]

            griper_grasp_score, suction_grasp_score, \
                shift_affordance_classifier, q_value, action_probs = \
                self.model_wrapper.model(rgb, pose_7_stack, normal_stack, mask)

            '''accumulate loss'''
            loss = torch.tensor(0., device=rgb.device)*griper_grasp_score.mean()
            for j in range(b):
                if used_gripper[j]:
                    label = gripper_score[j]
                    if label==-1:continue
                    g_pix_A = gripper_pixel_index[j, 0]
                    g_pix_B = gripper_pixel_index[j, 1]
                    prediction = griper_grasp_score[j, 0, g_pix_A, g_pix_B]
                    l=bce_loss(prediction, label)

                    self.gripper_quality_net_statistics.loss=l.item()
                    self.gripper_quality_net_statistics.update_confession_matrix(label,prediction)
                    loss += l

                if used_suction[j]:
                    label = suction_score[j]
                    if label==-1:continue
                    s_pix_A = suction_pixel_index[j, 0]
                    s_pix_B = suction_pixel_index[j, 1]
                    prediction = suction_grasp_score[j, 0, s_pix_A, s_pix_B]
                    l=bce_loss(prediction, label)

                    self.suction_quality_net_statistics.loss=l.item()
                    self.suction_quality_net_statistics.update_confession_matrix(label,prediction)

                    loss += l

            loss.backward()
            self.model_wrapper.optimizer.step()

            pi.step(i)
        pi.end()


    def view_result(self):
        with torch.no_grad():
            self.gripper_quality_net_statistics.print()
            self.suction_quality_net_statistics.print()

    def save_statistics(self):
        self.gripper_quality_net_statistics.save()
        self.suction_quality_net_statistics.save()

    def export_check_points(self):
        self.model_wrapper.export_model()
        self.model_wrapper.export_optimizer()

    def clear(self):
        self.gripper_quality_net_statistics.clear()
        self.suction_quality_net_statistics.clear()

if __name__ == "__main__":
    lr = 1e-5
    train_action_net = TrainPolicyNet(  learning_rate=lr)

    while True:
        train_action_net.initialize_model()
        train_action_net.synchronize_buffer()
        train_action_net.step_quality_training()
        train_action_net.export_check_points()
        train_action_net.save_statistics()
        train_action_net.view_result()
        # if train_action_net.training_trigger:
        #     train_action_net.initialize_model()
        #     train_action_net.synchronize_buffer()
        #     train_action_net.step_quality_training()
        # else:
        #     time.sleep(3)

