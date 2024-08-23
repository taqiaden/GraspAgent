import torch
from colorama import Fore
from torch.utils import data

from Configurations.config import theta_cos_scope
from dataloaders.Grasp_GAN_dl import Grasp_GAN_dataset, load_training_buffer
from lib.IO_utils import   custom_print
from lib.bbox import   decode_gripper_pose
from lib.collision_unit import grasp_collision_detection
from lib.dataset_utils import  training_data, online_data
from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from lib.models_utils import export_model_state, initialize_model
from lib.optimizer import export_optm, load_opt
from lib.report_utils import  progress_indicator
from models.Grasp_GAN import gripper_sampler_net, gripper_sampler_path, gripper_critic_path, critic_net
from pose_object import output_processing, approach_vec_to_theta_phi
from registration import camera
import torch.nn.functional as F

gripper_critic_optimizer_path = r'gripper_critic_optimizer.pth.tar'
gripper_sampler_optimizer_path = r'gripper_sampler_optimizer.pth.tar'

training_data=training_data()
online_data=online_data()

print=custom_print
BATCH_SIZE=2
learning_rate=5*1e-5
EPOCHS = 3
weight_decay = 0.000001
workers=2

m1=m2=m3=1.0
ref_approach = torch.tensor([0.0, 0.0, 1.0]).to('cuda') # vertical direction

def get_contrastive_loss(positive,negative,margin: float =0.5):
    loss=torch.clamp(margin-positive+negative,min=0.0)
    return loss.mean()

def train():
    '''dataloader'''
    dataset = Grasp_GAN_dataset(data_pool=training_data)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''models'''
    Critic=initialize_model(critic_net,gripper_critic_path)
    Critic.train(True)
    Generator=initialize_model(gripper_sampler_net,gripper_sampler_path)
    Generator.train(True)


    '''optimizers'''
    C_optimizer = torch.optim.Adam(Critic.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    C_optimizer = load_opt(C_optimizer, gripper_critic_optimizer_path)
    G_optimizer = torch.optim.SGD(Generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    G_optimizer = load_opt(G_optimizer, gripper_sampler_optimizer_path)



    def train_one_epoch():
        G_running_loss = 0.
        C_running_loss = 0.
        collision_times = 0.
        out_of_scope_times = 0.

        for i, batch in enumerate(dloader, 0):
            depth, pose_7, pixel_index = batch
            depth = depth.cuda().float()
            pose_7 = pose_7.cuda().float()  # [b,1,7]
            b = depth.shape[0]

            '''generate grasps'''
            with torch.no_grad():
                generated_grasps=Generator(depth)

            '''Evaluate generated grasps'''
            collision_state_list=[]
            firmness_state_list=[]
            out_of_scope_list=[]
            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

                '''get pose parameters'''
                target_point = pixel_to_point(pixel_index[j].cpu().numpy(), depth_value, camera)
                target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

                target_pose = generated_grasps[j,:,pix_A,pix_B]
                approach = target_pose[ 0:3]
                theta, phi_sin, phi_cos = approach_vec_to_theta_phi(approach[None,:])
                target_pose[ 0:1] = theta
                target_pose[ 1:2] = phi_sin
                target_pose[2:3] = phi_cos
                pose_5 = output_processing(target_pose[None, :, None]).squeeze(-1)
                if j==0:print(f'Example _pose = {pose_5}')
                pose_good_grasp = decode_gripper_pose(pose_5, target_point)

                '''get point clouds'''
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)

                '''check collision'''
                collision_intensity = grasp_collision_detection(pose_good_grasp, pc, visualize=False,
                                                                add_floor=False)
                collision_state_list.append(collision_intensity>0)

                '''check firmness'''
                label_dist=pose_7[j,0,-2]
                generated_dist=generated_grasps[j,-2,pix_A,pix_B]
                firmness_state_list.append(generated_dist>label_dist)

                '''check parameters are within scope'''
                approach_cos = F.cosine_similarity(approach, ref_approach,dim=0)
                in_scope=1.0>generated_grasps[j,-2,pix_A,pix_B]>0.0 and 1.0>generated_grasps[j,-1,pix_A,pix_B]>0.0 and approach_cos>theta_cos_scope
                out_of_scope_list.append(not in_scope)

            collision_times+=sum(collision_state_list)
            out_of_scope_times+=sum(out_of_scope_list)

            '''Quality metrics'''
            values=generated_grasps.permute(1,0,2,3).flatten(1)
            std=torch.std(values,dim=-1)
            max_=torch.max(values,dim=-1)[0]
            min_=torch.min(values,dim=-1)[0]
            print(f'std = {std}')
            print(f'max = {max_}')
            print(f'min = {min_}')

            '''zero grad'''
            Critic.zero_grad()
            Generator.zero_grad()

            '''label sub batch'''
            label_generated_grasps=generated_grasps.clone()
            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                label_generated_grasps[j,:,pix_A,pix_B]=pose_7[j]


            '''concatenation'''
            generated_grasps_cat=torch.cat([generated_grasps,label_generated_grasps],dim=0)
            depth_cat=depth.repeat(2,1,1,1)


            '''get predictions'''
            critic_score=Critic(depth_cat,generated_grasps_cat)

            '''accumulate loss'''
            curriculum_loss=0.
            collision_loss=0.
            firmness_loss=0.
            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                prediction_ = critic_score[j, 0, pix_A, pix_B]
                label_=critic_score[j+b, 0, pix_A, pix_B]
                collision_state_=collision_state_list[j]
                out_of_scope=out_of_scope_list[j]
                bad_state_grasp=collision_state_*out_of_scope
                firmness_state=firmness_state_list[j]
                curriculum_loss+=torch.clamp(label_-prediction_-m1,0)
                collision_loss+=(torch.clamp(prediction_-label_+m2,0) * bad_state_grasp)
                firmness_loss+=(torch.clamp((prediction_-label_)*(1-2*firmness_state),0) * (1-bad_state_grasp))

            C_loss = (curriculum_loss+collision_loss+firmness_loss)/b
            print('C_loss=',C_loss.item())
            print(f'Collision times = {sum(collision_state_list)}')

            '''optimizer step'''
            C_loss.backward()
            C_optimizer.step()

            C_running_loss += C_loss.item()

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
            G_loss=0.0
            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                collision_state_=collision_state_list[j]
                out_of_scope = out_of_scope_list[j]
                bad_state_grasp = collision_state_ * out_of_scope

                label_score=label_critic_score[j,0,pix_A, pix_B]
                prediction_score=generated_critic_score[j,0,pix_A, pix_B]


                G_loss += torch.clamp(label_score - prediction_score - m3*(1-bad_state_grasp), min=0.)
            G_loss=(G_loss/b)
            print('G_loss=',G_loss.item())

            '''optimizer step'''
            G_loss.backward()
            G_optimizer.step()
            G_running_loss += G_loss.item()

            pi.step(i)
            print()
        return C_running_loss,G_running_loss,collision_times,out_of_scope_times

    for epoch in range(EPOCHS):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1),max_limit=len(dloader))

        # with torch.autograd.set_detect_anomaly(True):
        C_running_loss,G_running_loss,collision_times,out_of_scope_times= train_one_epoch()

        export_optm(C_optimizer, gripper_critic_optimizer_path)
        export_optm(G_optimizer, gripper_sampler_optimizer_path)

        pi.end()

        print('   Critic running loss = ', C_running_loss, ', loss per iteration = ', C_running_loss / len(dataset))
        print('   Generator running loss = ', G_running_loss, ', loss per iteration = ', G_running_loss / len(dataset))
        print('   collision times = ', collision_times, ', collision ratio = ', collision_times / len(dataset))
        print('   Out of scope times = ', out_of_scope_times, ', Out of scope ratio = ', out_of_scope_times / len(dataset))

    return Critic,Generator



def train_Grasp_GAN(n_samples=None):
    # training_data.clear()
    while True:
        # try:
        if len(training_data) == 0:
            load_training_buffer(size=n_samples)
        Critic,Generator = train()
        print(Fore.GREEN + 'Training round finished' + Fore.RESET)
        export_model_state(Critic, gripper_critic_path)
        export_model_state(Generator, gripper_sampler_path)
        training_data.clear()
        # except Exception as e:
        #     print(Fore.RED, str(e), Fore.RESET)

if __name__ == "__main__":
    train_Grasp_GAN(1000)