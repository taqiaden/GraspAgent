import torch
from colorama import Fore
from torch import nn
from torch.utils import data
from Configurations.config import theta_cos_scope, workers
from Configurations.dynamic_config import save_key, get_float
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker
from analytical_suction_sampler import estimate_suction_direction
from check_points.check_point_conventions import GANWrapper
from dataloaders.joint_grasp_sampler_dl import GraspSamplerDataset
from lib.IO_utils import   custom_print
from lib.collision_unit import grasp_collision_detection
from lib.dataset_utils import online_data
from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from lib.optimizer import exponential_decay_lr_
from models.joint_grasp_sampler import GraspSampler, Critic
from pose_object import  pose_7_to_transformation
from registration import camera
import torch.nn.functional as F
from lib.report_utils import  progress_indicator
from filelock import FileLock

lock = FileLock("file.lock")

module_key='grasp_sampler'
training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

print=custom_print
max_lr=0.01
min_lr=1*1e-6

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

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
        T_d, width, distance=pose_7_to_transformation(target_pose, target_point)
        if j == 0: print(f'Example _pose = {generated_grasps[j, :, pix_A, pix_B]}')

        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        '''check collision'''
        collision_intensity = grasp_collision_detection(T_d,width, pc, visualize=False )
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

def train_critic(gan,generated_grasps,batch_size,pixel_index,label_generated_grasps,depth,
                 collision_state_list,out_of_scope_list,firmness_state_list):
    '''zero grad'''
    gan.critic.zero_grad()
    gan.generator.zero_grad()

    '''concatenation'''
    generated_grasps_cat = torch.cat([generated_grasps, label_generated_grasps], dim=0)
    depth_cat = depth.repeat(2, 1, 1, 1)

    '''get predictions'''
    critic_score = gan.critic(depth_cat, generated_grasps_cat)

    '''accumulate loss'''
    # curriculum_loss = 0.
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
        # curriculum_loss += (torch.clamp(label_ - prediction_ - m1, 0))**loss_power
        collision_loss += (torch.clamp(prediction_ - label_ + 1, 0) * bad_state_grasp)
        generated_dist = generated_grasps[j, -2, pix_A, pix_B]
        activate_firmness_loss=1 if generated_dist<0.2 else 0.0
        firmness_loss += (torch.clamp((prediction_ - label_) * (1 - 2 * firmness_state), 0) * (1 - bad_state_grasp))*activate_firmness_loss

        print(f'col_l={collision_loss}, fir_l={firmness_loss}')

    C_loss = (( collision_loss + firmness_loss) / batch_size)

    print(Fore.GREEN, 'C_loss=', C_loss.item(), Fore.RESET)

    '''optimizer step'''
    C_loss.backward()
    gan.critic_optimizer.step()
    gan.critic_optimizer.zero_grad()

    return C_loss.item()
def gripper_sampler_loss(pixel_index,j,collision_state_list,out_of_scope_list,label_critic_score,generated_critic_score):
    pix_A = pixel_index[j, 0]
    pix_B = pixel_index[j, 1]
    collision_state_ = collision_state_list[j]
    out_of_scope = out_of_scope_list[j]
    bad_state_grasp = collision_state_ or out_of_scope

    label_score = label_critic_score[j, 0, pix_A, pix_B]
    prediction_score = generated_critic_score[j, 0, pix_A, pix_B]

    return (torch.clamp(label_score - prediction_score - 1 * (1 - bad_state_grasp), min=0.))

def suction_sampler_loss(depth,j,normals):
    '''generate labels'''
    pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
    pc = transform_to_camera_frame(pc, reverse=True)
    labels = estimate_suction_direction(pc, view=False)  # inference time on local computer = 1.3 s
    labels = torch.from_numpy(labels).to('cuda')
    '''mask prediction'''
    masked_prediction = normals[j][mask]
    '''view output'''
    # view_npy_open3d(pc,normals=normals)
    # normals=masked_prediction.detach().cpu().numpy()
    # view_npy_open3d(pc,normals=normals)
    return ((1 - cos(masked_prediction, labels.squeeze())) ** 2).mean()
def train_generator(gan,depth,label_generated_grasps,batch_size,pixel_index,collision_state_list,out_of_scope_list):
    '''zero grad'''
    gan.critic.zero_grad()
    gan.generator.zero_grad()

    '''critic score of reference label '''
    with torch.no_grad():
        label_critic_score = gan.critic(depth.clone(), label_generated_grasps)

    '''generated grasps'''
    generated_grasps,normals = gan.generator(depth.clone())

    '''Critic score of generated grasps'''
    generated_critic_score = gan.critic(depth.clone(), generated_grasps)

    '''accumulate loss'''
    G_loss = 0.0
    for j in range(batch_size):
        gripper_loss=gripper_sampler_loss(pixel_index,j,collision_state_list,out_of_scope_list,label_critic_score,generated_critic_score)
        suction_loss=suction_sampler_loss(depth, j, normals.permute(0, 2, 3, 1))
        balance_weight=1 if suction_loss<=gripper_loss else gripper_loss/suction_loss
        G_loss+=gripper_loss
        G_loss+=suction_loss*balance_weight
        print(Fore.GREEN, f'Gripper sampler loss = {gripper_loss.item()}, Suction sampler loss = {suction_loss.item()}, balance weight = {balance_weight}', Fore.RESET)
    G_loss = (G_loss / batch_size)/2


    '''optimizer step'''
    G_loss.backward()
    gan.generator_optimizer.step()
    gan.generator_optimizer.zero_grad()

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

def train(batch_size=1,n_samples=None,epochs=1,learning_rate=5e-5):
    '''load  models'''
    gan=GANWrapper(module_key,GraspSampler,Critic)
    gan.ini_models(train=True)

    '''optimizers'''
    gan.critic_sgd_optimizer(learning_rate=learning_rate)
    gan.generator_adam_optimizer(learning_rate=learning_rate)

    '''dataloader'''
    file_ids=sample_positive_buffer(size=n_samples, dict_name=gripper_grasp_tracker)
    print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
    dataset = GraspSamplerDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    for epoch in range(epochs):
        g_running_loss = 0.
        c_running_loss = 0.
        collision_times = 0.
        out_of_scope_times = 0.
        good_firmness_times = 0.

        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))

        for i, batch in enumerate(data_loader, 0):

            depth, pose_7, pixel_index = batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pose_7 = pose_7.cuda().float().squeeze(1)  # [b,7]
            b = depth.shape[0]

            '''generate grasps'''
            with torch.no_grad():
                generated_grasps,_ = gan.generator(depth.clone())

            '''process label'''
            label_generated_grasps = generated_grasps.clone()
            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                label_generated_grasps[j, :, pix_A, pix_B] = pose_7[j]

            '''Evaluate generated grasps'''
            collision_state_list, firmness_state_list, out_of_scope_list = evaluate_grasps(b, pixel_index, depth,
                                                                                           generated_grasps, pose_7)
            collision_times += sum(collision_state_list)
            out_of_scope_times += sum(out_of_scope_list)
            good_firmness_times += sum(firmness_state_list)

            '''Quality metrics'''
            view_metrics(generated_grasps, collision_state_list, out_of_scope_list, firmness_state_list)

            '''train critic'''
            c_running_loss += train_critic(gan, generated_grasps, b, pixel_index,
                                  label_generated_grasps, depth,
                                  collision_state_list, out_of_scope_list, firmness_state_list)

            '''train generator'''
            g_running_loss += train_generator(gan, depth, label_generated_grasps, b,
                                     pixel_index,
                                     collision_state_list, out_of_scope_list)

            '''print batch summary'''
            view_metrics(generated_grasps, collision_state_list, out_of_scope_list, firmness_state_list)

            pi.step(i)
        pi.end()

        gan.export_models()
        gan.export_optimizers()

        '''Final report'''
        print(Fore.BLUE)
        size=len(file_ids)
        print(f'Collision ratio = {collision_times / size}')
        print(f'out of scope ratio = {out_of_scope_times / size}')
        print(f'firm grasp ratio = {good_firmness_times / size}')
        print(f'Average Critic loss = {c_running_loss / size}')
        print(f'Average Generator loss = {g_running_loss / size}')
        print(Fore.RESET)

        '''update performance indicator'''
        performance_indicator= 1 - max(collision_times, out_of_scope_times) / (size)
        save_key("performance_indicator", performance_indicator, section=module_key)

if __name__ == "__main__":
    for i in range(10000):
        '''get adaptive lr'''
        performance_indicator = get_float("performance_indicator", section=module_key,default='0')
        adaptive_lr = exponential_decay_lr_(performance_indicator, max_lr, min_lr)
        print(Fore.CYAN, f'performance_indicator = {performance_indicator}, Learning rate = {adaptive_lr}', Fore.RESET)
        train(batch_size=1,n_samples=100,epochs=1,learning_rate=adaptive_lr)