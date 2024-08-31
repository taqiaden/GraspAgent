import datetime
import os
import torch
from colorama import Fore
from torch import nn
from torch.utils import data
from Configurations.config import theta_cos_scope
from Configurations.dynamic_config import save_key, get_value, add_to_value, get_float
from dataloaders.Grasp_GAN_dl import Grasp_GAN_dataset, load_training_buffer
from lib.IO_utils import   custom_print
from lib.collision_unit import grasp_collision_detection
from lib.dataset_utils import  training_data, online_data
from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from lib.models_utils import export_model_state, initialize_model
from lib.optimizer import export_optm, load_opt, exponential_decay_lr_
from models.Grasp_GAN import gripper_sampler_net, gripper_sampler_path, gripper_critic_path, critic_net
from pose_object import  pose_7_to_pose_good_grasp
from registration import camera, standardize_depth
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

gripper_critic_optimizer_path = r'gripper_critic_optimizer.pth.tar'
gripper_sampler_optimizer_path = r'gripper_sampler_optimizer.pth.tar'

training_data=training_data()
online_data=online_data()

print=custom_print
max_lr=0.01
min_lr=1*1e-6
weight_decay = 0.000001

loss_power=2.0

m1=1.0
m2=1.0
m3=1.0

maximum_gpus=1

activate_full_power_at_midnight=True

result_queue=mp.Queue()

def ddp_setup(rank,world_size):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="12355"
    init_process_group(backend="nccl",rank=rank,world_size=world_size)

class train_config:
    def __init__(self,batch_size,epochs,workers,world_size):
        self.batch_size=batch_size
        self.epochs=epochs
        self.workers=workers
        self.world_size=world_size


def evaluate_grasps(batch_size,pixel_index,depth,generated_grasps,pose_7):
    '''Evaluate generated grasps'''
    collision_state_list = []
    firmness_state_list = []
    out_of_scope_list = []
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index[j].cpu().numpy(), depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

        target_pose=generated_grasps[j, :, pix_A, pix_B]
        pose_good_grasp = pose_7_to_pose_good_grasp(target_pose, target_point)
        if j == 0: print(f'Example _pose = {generated_grasps[j, :, pix_A, pix_B]}')

        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        '''check collision'''
        collision_intensity = grasp_collision_detection(pose_good_grasp, pc, visualize=False )
        collision_state_=collision_intensity > 0
        collision_state_list.append(collision_state_)

        '''check parameters are within scope'''
        ref_approach = torch.tensor([0.0, 0.0, 1.0],device=target_pose.device)  # vertical direction

        approach_cos = F.cosine_similarity(target_pose[0:3], ref_approach, dim=0)
        in_scope = 1.0 > generated_grasps[j, -2, pix_A, pix_B] > 0.0 and 1.0 > generated_grasps[
            j, -1, pix_A, pix_B] > 0.0 and approach_cos > theta_cos_scope
        out_of_scope_list.append(not in_scope)

        '''check firmness'''
        label_dist = pose_7[j, -2]
        generated_dist = generated_grasps[j, -2, pix_A, pix_B]
        firmness_=1 if generated_dist.item() > label_dist.item() and not collision_state_ and in_scope else 0
        firmness_state_list.append(firmness_)

    return collision_state_list,firmness_state_list,out_of_scope_list

def train_critic(Critic,Generator,C_optimizer,generated_grasps,batch_size,pixel_index,label_generated_grasps,depth,
                 collision_state_list,out_of_scope_list,firmness_state_list):
    '''zero grad'''
    Critic.zero_grad()
    Generator.zero_grad()

    '''concatenation'''
    generated_grasps_cat = torch.cat([generated_grasps, label_generated_grasps], dim=0)
    depth_cat = depth.repeat(2, 1, 1, 1)

    '''get predictions'''
    critic_score = Critic(depth_cat, generated_grasps_cat)

    '''accumulate loss'''
    curriculum_loss = 0.
    collision_loss = 0.
    firmness_loss = 0.
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        prediction_ = critic_score[j, 0, pix_A, pix_B]
        label_ = critic_score[j + batch_size, 0, pix_A, pix_B]
        print(Fore.YELLOW,f'prediction score = {prediction_.item()}, label score = {label_.item()}',Fore.RESET)

        collision_state_ = collision_state_list[j]
        out_of_scope = out_of_scope_list[j]
        bad_state_grasp = collision_state_ or out_of_scope
        firmness_state = firmness_state_list[j]
        curriculum_loss += (torch.clamp(label_ - prediction_ - m1, 0))**loss_power
        collision_loss += (torch.clamp(prediction_ - label_ + m2, 0) * bad_state_grasp)**loss_power
        firmness_loss += (torch.clamp((prediction_ - label_) * (1 - 2 * firmness_state), 0) * (1 - bad_state_grasp))**loss_power
        print(f'cur_l={curriculum_loss}, col_l={collision_loss}, fir_l={firmness_loss}')

    C_loss = ((curriculum_loss + collision_loss + firmness_loss) / batch_size)

    '''optimizer step'''
    C_loss.backward()
    C_optimizer.step()
    C_optimizer.zero_grad()

    return C_loss.item()

def train_generator(Critic,Generator,G_optimizer,depth,label_generated_grasps,batch_size,pixel_index,collision_state_list,out_of_scope_list):
    '''zero grad'''
    Critic.zero_grad()
    Generator.zero_grad()

    '''critic score of reference label '''
    with torch.no_grad():
        label_critic_score = Critic(depth, label_generated_grasps)

    '''generated grasps'''
    generated_grasps = Generator(depth)

    '''Critic score of generated grasps'''
    generated_critic_score = Critic(depth, generated_grasps)

    '''accumulate loss'''
    G_loss = 0.0
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        collision_state_ = collision_state_list[j]
        out_of_scope = out_of_scope_list[j]
        bad_state_grasp = collision_state_ or out_of_scope

        label_score = label_critic_score[j, 0, pix_A, pix_B]
        prediction_score = generated_critic_score[j, 0, pix_A, pix_B]

        G_loss += (torch.clamp(label_score - prediction_score - m3 * (1 - bad_state_grasp), min=0.))**loss_power
    G_loss = (G_loss / batch_size)

    '''optimizer step'''
    G_loss.backward()
    G_optimizer.step()
    G_optimizer.zero_grad()

    return G_loss.item()

def view_metrics(generated_grasps,collision_state_list,out_of_scope_list,firmness_state_list):
    values = generated_grasps.permute(1, 0, 2, 3).flatten(1)
    std = torch.std(values, dim=-1)
    max_ = torch.max(values, dim=-1)[0]
    min_ = torch.min(values, dim=-1)[0]
    print(f'std = {std}')
    print(f'max = {max_}')
    print(f'min = {min_}')
    print(f'Collision times = {sum(collision_state_list)}')
    print(f'Out of scope times = {sum(out_of_scope_list)}')
    print(f'good firmness times = {sum(firmness_state_list)}')

def prepare_models_and_optimizers(adaptive_lr):

    print(Fore.CYAN,'Import check points',Fore.RESET)

    '''models'''
    Critic = initialize_model(critic_net, gripper_critic_path)
    Critic.train(True)
    # Critic.share_memory()
    Generator = initialize_model(gripper_sampler_net, gripper_sampler_path)
    Generator.train(True)
    # Generator.share_memory()

    '''optimizers'''
    # C_optimizer = torch.optim.Adam(Critic.parameters(), lr=adaptive_lr, betas=(0.9, 0.999), eps=1e-8,weight_decay=weight_decay)
    C_optimizer = torch.optim.SGD(Critic.parameters(), lr=adaptive_lr, weight_decay=weight_decay)
    C_optimizer = load_opt(C_optimizer, gripper_critic_optimizer_path)
    # G_optimizer = torch.optim.Adam(Generator.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    G_optimizer = torch.optim.SGD(Generator.parameters(), lr=adaptive_lr, weight_decay=weight_decay)
    # G_optimizer = torch.optim.RMSprop(Generator.parameters(),alpha=0.9, lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    G_optimizer = load_opt(G_optimizer, gripper_sampler_optimizer_path)

    return Critic,Generator,C_optimizer,G_optimizer
from lib.report_utils import  progress_indicator

class TrainerDDP:
    def __init__(self,gpu_id: int,Critic: nn.Module,Generator: nn.Module,C_optimizer,G_optimizer,training_congiurations):
        '''set devices and model wrappers'''
        self.world_size=training_congiurations.world_size
        self.batch_size=training_congiurations.batch_size
        self.epochs=training_congiurations.epochs
        self.workers=training_congiurations.workers
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()

        self.device=f'cuda:{gpu_id}'

        self.Critic=Critic
        self.Generator=Generator
        self.Critic.to(self.device)
        self.Generator.to(self.device)

        self.Critic=DDP(Critic,device_ids=[gpu_id],find_unused_parameters=True)
        self.Generator=DDP(Generator,device_ids=[gpu_id],find_unused_parameters=True)



        self.gpu_id=gpu_id

        '''dataloader'''
        dataset = Grasp_GAN_dataset(data_pool=training_data)
        self.data_laoder=self._prepare_dataloader(dataset)

        '''optimizers'''
        self.C_optimizer=C_optimizer
        self.G_optimizer=G_optimizer

    def _prepare_dataloader(self,dataset):
        dloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=False,
                                              sampler=DistributedSampler(dataset))
        return dloader

    def export_check_points(self):
        if self.gpu_id==0:
            '''export models'''
            export_model_state(self.Critic, gripper_critic_path)
            export_model_state(self.Generator, gripper_sampler_path)
            '''export optimizers'''
            export_optm(self.C_optimizer, gripper_critic_optimizer_path)
            export_optm(self.G_optimizer, gripper_sampler_optimizer_path)

            print(Fore.CYAN, 'Check points exported successfully',Fore.RESET)


    def train(self):

        def train_one_epoch():
            G_running_loss = 0.
            C_running_loss = 0.
            collision_times = 0.
            out_of_scope_times = 0.
            good_firmness_times = 0.

            for i, batch in enumerate(self.data_laoder, 0):

                depth, pose_7, pixel_index = batch
                depth = depth.to(self.device).float()  # [b,1,480.712]
                pose_7 = pose_7.to(self.device).float().squeeze(1)  # [b,7]
                b = depth.shape[0]


                '''generate grasps'''
                with torch.no_grad():
                    generated_grasps = self.Generator(depth)

                '''Evaluate generated grasps'''
                collision_state_list, firmness_state_list, out_of_scope_list = evaluate_grasps(b, pixel_index, depth,
                                                                                               generated_grasps, pose_7)
                collision_times += sum(collision_state_list)
                out_of_scope_times += sum(out_of_scope_list)
                good_firmness_times += sum(firmness_state_list)

                '''Quality metrics'''
                view_metrics(generated_grasps,collision_state_list,out_of_scope_list,firmness_state_list)

                '''label sub batch'''
                label_generated_grasps = generated_grasps.clone()
                for j in range(b):
                    pix_A = pixel_index[j, 0]
                    pix_B = pixel_index[j, 1]
                    label_generated_grasps[j, :, pix_A, pix_B] = pose_7[j]

                '''train critic'''
                C_loss = train_critic(self.Critic, self.Generator, self.C_optimizer, generated_grasps, b, pixel_index,
                                      label_generated_grasps, depth,
                                      collision_state_list, out_of_scope_list, firmness_state_list)
                C_running_loss += C_loss
                print(Fore.GREEN, 'C_loss=', C_loss, Fore.RESET)

                '''train generator'''
                G_loss = train_generator(self.Critic, self.Generator, self.G_optimizer, depth, label_generated_grasps, b, pixel_index,
                                         collision_state_list, out_of_scope_list)
                G_running_loss += G_loss
                print(Fore.GREEN, 'G_loss=', G_loss, Fore.RESET)

                pi.step(i)
                print()
            return C_running_loss, G_running_loss, collision_times, out_of_scope_times, good_firmness_times

        for epoch in range(self.epochs):
            pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(self.data_laoder))

            C_running_loss, G_running_loss, collision_times, out_of_scope_times, good_firmness_times = train_one_epoch()

            pi.end()

            '''send information to main process'''
            add_to_value("C_running_loss", C_running_loss, section='Grasp_GAN', lock_other_process=True)
            add_to_value("G_running_loss", G_running_loss, section='Grasp_GAN', lock_other_process=True)
            add_to_value("collision_times", collision_times, section='Grasp_GAN', lock_other_process=True)
            add_to_value("out_of_scope_times", out_of_scope_times, section='Grasp_GAN', lock_other_process=True)
            add_to_value("good_firmness_times", good_firmness_times, section='Grasp_GAN', lock_other_process=True)

            '''export models and optimizers'''
            self.export_check_points()

def train_Grasp_GAN(n_samples=None):
    training_data.clear()
    performance_indicator = float(get_value("performance_indicator", section='Grasp_GAN'))
    while True:
        '''zero records'''
        save_key("C_running_loss", 0., section='Grasp_GAN')
        save_key("G_running_loss", 0., section='Grasp_GAN')
        save_key("collision_times", 0., section='Grasp_GAN')
        save_key("out_of_scope_times", 0., section='Grasp_GAN')
        save_key("good_firmness_times", 0., section='Grasp_GAN')

        '''get adaptive lr'''
        adaptive_lr = exponential_decay_lr_(performance_indicator,max_lr,min_lr)
        print(Fore.CYAN, f'performance_indicator = {performance_indicator}, Learning rate = {adaptive_lr}', Fore.RESET)

        '''load check points'''
        Critic, Generator, C_optimizer, G_optimizer = prepare_models_and_optimizers(adaptive_lr)

        '''configure world_size'''
        now = datetime.datetime.now()
        if activate_full_power_at_midnight and 7>now.hour>3:
            world_size = torch.cuda.device_count()
        else:
            world_size = torch.cuda.device_count() if maximum_gpus is None else maximum_gpus

        '''train configurations'''
        BATCH_SIZE=2
        print(Fore.YELLOW, f'Batch size per node = {BATCH_SIZE} ', Fore.RESET)
        t_config=train_config(BATCH_SIZE,1,0,world_size)

        '''prepare buffer'''
        if len(training_data) == 0:
            load_training_buffer(size=n_samples)
        buffer_size=len(training_data)
        print(Fore.YELLOW,f'Buffer size = {buffer_size} ', Fore.RESET)

        '''Setup communication between process'''

        '''Begin multi processing'''
        mp.spawn(main_ddp,args=(t_config,Critic,Generator,C_optimizer,G_optimizer),nprocs=world_size)

        '''Final report'''
        print(Fore.BLUE)
        C_running_loss_all=get_float("C_running_loss", section='Grasp_GAN')
        G_running_loss_all=get_float("G_running_loss",  section='Grasp_GAN')
        total_collision=get_float("collision_times",  section='Grasp_GAN')
        total_out_of_scope=get_float("out_of_scope_times",  section='Grasp_GAN')
        total_firm_grasp=get_float("good_firmness_times",  section='Grasp_GAN')
        print(f'Collision ratio = {total_collision/buffer_size}')
        print(f'out of scope ratio = {total_out_of_scope/buffer_size}')
        print(f'firm grasp ratio = {total_firm_grasp/buffer_size}')
        print(f'Average Critic loss = {C_running_loss_all/buffer_size}')
        print(f'Average Generator loss = {G_running_loss_all/buffer_size}')
        print(Fore.RESET)

        '''update performance indicator'''
        performance_indicator=1-max(total_collision,total_out_of_scope)/buffer_size
        save_key("performance_indicator", performance_indicator, section='Grasp_GAN')

        '''clear buffer'''
        training_data.clear()


def main_ddp(rank: int, t_config,Critic,Generator,C_optimizer,G_optimizer):
    '''setup parallel training'''
    ddp_setup(rank,t_config.world_size)

    '''training'''
    trainer=TrainerDDP(gpu_id=rank,Critic=Critic,Generator=Generator,C_optimizer=C_optimizer,G_optimizer=G_optimizer,training_congiurations=t_config)
    trainer.train()

    '''destroy process'''
    destroy_process_group()


if __name__ == "__main__":
    train_Grasp_GAN(900)