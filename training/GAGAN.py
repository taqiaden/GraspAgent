import math
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from colorama import Fore
from torch.utils import data
from Configurations import config
from dataloaders.GAGAN_dataloader import gripper_dataset, load_training_data_from_online_pool
from lib.IO_utils import   custom_print
from lib.bbox import   decode_gripper_pose
from lib.collision_unit import  grasp_collision_detection
from lib.dataset_utils import  training_data, online_data
from lib.grasp_utils import    get_homogenous_matrix
from lib.mesh_utils import construct_gripper_mesh
from lib.models_utils import export_model_state, initialize_model
from lib.optimizer import export_optm, load_opt
from lib.report_utils import  progress_indicator
from masks import get_spatial_mask
from models.GAGAN import dense_gripper_generator_path, contrastive_discriminator_path, gripper_generator, \
    contrastive_discriminator
from models.gripper_D import gripper_discriminator, dense_gripper_discriminator_path
from pose_object import output_processing, approach_vec_to_theta_phi
from records.records_managment import save_record
from visualiztion import vis_scene

contrastive_discriminator_optimizer_path = r'contrastive_discriminator_optimizer.pth.tar'
dense_gripper_generator_GAN_optimizer_path = r'dense_gripper_generator_GAN_optimizer.pth.tar'

time_seed=math.floor(datetime.now().timestamp())

training_data=training_data()
online_data=online_data()

weight_decay = 0.0000001
print=custom_print
BATCH_SIZE=1
learning_rate=5*1e-6

EPOCHS=1
online_samples_per_round=500

move_trained_data=False

def verify(pc_data,poses_7,idx,view=False,evaluation_metric=None):

    poses_7=poses_7[:,None,:].clone()
    collision_intensity_list=[]
    # visualize verfication
    for i in range(pc_data.shape[0]):

        if evaluation_metric is not None:
            # if abs(evaluation_metric[i]-predictions[i])<0.5:continue
            if evaluation_metric[i] != 1:
                print('failed to grasp')
                # continue
            else:
                print('successful grasp')
        pc_ = pc_data[i].cpu().numpy()  # + mean
        # continue
        # print(f'center point={center_point}')
        # print(f'Pose value={dense_pose[i, :, idx[i]]}')

        center_point = pc_[idx[i]]
        target_pose=poses_7[i,:, :]
        approach=target_pose[:, 0:3]

        theta,phi_sin,phi_cos=approach_vec_to_theta_phi(approach)
        target_pose[:,0:1]=theta
        target_pose[:,1:2]=phi_sin
        target_pose[:,2:3]=phi_cos
        pose_5=output_processing(target_pose[:,:,None]).squeeze(-1)
        # print(pose_5.shape)
        pose_good_grasp = decode_gripper_pose(pose_5, center_point[0:3])
        # view_local_grasp(pose_good_grasp, pc_)
        collision_intensity = grasp_collision_detection(pose_good_grasp, pc_[:, 0:3], visualize=view,add_floor=False)
        collision_intensity_list.append(collision_intensity)

    collision_intensity=torch.FloatTensor(collision_intensity_list)
    return collision_intensity

def get_mesh(pc_data,poses_7,idx):
    # print(pc_data.shape)
    # print(poses_7.shape)
    # print(idx.shape)
    #
    # exit()
    poses_7=poses_7[:,None,:].clone()
    # visualize verfication
    i=0
    pc_ = pc_data[i].cpu().numpy()  # + mean

    center_point = pc_[idx]
    target_pose=poses_7[i,:, :]
    approach=target_pose[:, 0:3]
    theta,phi_sin,phi_cos=approach_vec_to_theta_phi(approach)
    target_pose[:,0:1]=theta
    target_pose[:,1:2]=phi_sin
    target_pose[:,2:3]=phi_cos
    pose_5=output_processing(target_pose[:,:,None]).squeeze(-1)

    pose_good_grasp = decode_gripper_pose(pose_5, center_point[0:3])
    # view_local_grasp(pose_good_grasp, pc_)
    T = get_homogenous_matrix(pose_good_grasp)
    width = pose_good_grasp[0, 0]

    mesh = construct_gripper_mesh(width, T)

    return mesh

bce_loss=nn.BCELoss()
bce_loss_vec=nn.BCELoss(reduction=False)

l1_loss=nn.L1Loss()
mse_loss=nn.MSELoss()
mse_loss_vec=nn.MSELoss(reduction='none')
l1_loss_vec=nn.L1Loss(reduction='none')
l1_smooth_loss=nn.SmoothL1Loss(beta=1.0)

def get_contrastive_loss(positive,negative,margin: float =0.5):
    dist=positive-negative
    loss=torch.clamp(margin-dist,min=0.0)
    # loss=torch.pow(loss,2)
    return loss.mean()

counter = 0

collision_accumulator=None
collision_ref_accumulator=None

def gradient_penalty(pc,critic,generated_pose_7,index,real,fake):
    batch_size,p=real.shape
    epsilon=torch.rand((batch_size,1)).repeat(1,p).to('cuda')
    interpolated_poses=real*epsilon+fake*(1-epsilon)
    interpolated_poses=interpolated_poses.detach().clone()
    interpolated_poses.requires_grad=True
    dense_pose_7 = generated_pose_7.detach().clone()
    # dense_pose_7.requires_grad=True

    for ii, j in enumerate(index):
        dense_pose_7[ii, :, j] = interpolated_poses[ii]
    critic_scores = critic(pc, dense_pose_7)
    mixed_score = torch.stack([critic_scores[i, :, j] for i, j in enumerate(index)])

    gradient=torch.autograd.grad(inputs=interpolated_poses,
                                 outputs=mixed_score,
                                 grad_outputs=torch.ones_like(mixed_score),
                                 create_graph=True,
                                 retain_graph=True)[0]
    gradient=gradient.view(gradient.shape[0],-1)
    gradient_norm=gradient.norm(2,dim=1)

    gradient_penalty=torch.mean((gradient_norm-1)**2)


    return gradient_penalty
critic_optimizer=None
generator_optimizer=None
generator_net=None
critic_net=None
def train(EPOCHS_,batch_size,directory):
    global critic_optimizer
    global generator_optimizer
    global critic_net
    global generator_net

    counter_x=[0,0,0]

    regular_dis = initialize_model(gripper_discriminator,dense_gripper_discriminator_path)

    regular_dis.eval()
    # representation_net_ = representation_net()
    # representation_net_ = initialize_model_state(representation_net_, representation_net_model_path)
    # representation_net_.train(True)
    if critic_net is None:
        contrastive_dis=initialize_model(contrastive_discriminator,contrastive_discriminator_path)
        contrastive_dis.train(False)
    else:
        contrastive_dis=critic_net


    # collision_net=collision_estimator_net_
    # collision_net.train()
    if generator_net is None:
        generator = initialize_model(gripper_generator,dense_gripper_generator_path)
        generator.train(False)
    else:
        generator=generator_net
    # cloned_generator=copy.deepcopy(generator)

    # generator_f=generator.direct_output
    dataset = gripper_dataset(num_points=config.num_points, path=directory)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)


    #contrastive_dis_optimizer = decayed_optimizer(contrastive_dis,lr_=learning_rate,decay_rate=0.99999,use_RMSprop=False,use_sgd=False)
    if critic_optimizer is None:
        contrastive_dis_optimizer = torch.optim.Adam(contrastive_dis.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        # contrastive_dis_optimizer=torch.optim.SGD(contrastive_dis.parameters(), lr=learning_rate, weight_decay=weight_decay)

        contrastive_dis_optimizer = load_opt(contrastive_dis_optimizer, contrastive_discriminator_optimizer_path)
    else:
        contrastive_dis_optimizer=critic_optimizer

    if generator_optimizer is None:
        # gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        gen_optimizer=torch.optim.SGD(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)

        gen_optimizer = load_opt(gen_optimizer, dense_gripper_generator_GAN_optimizer_path)
    else:
        gen_optimizer=generator_optimizer
    # bins = [2, 2, 2, 1, 1, 2, 2]
    def export_model():
        print(Fore.GREEN, 'Export check point', Fore.RESET)
        export_optm(gen_optimizer, dense_gripper_generator_GAN_optimizer_path)
        export_model_state(generator, dense_gripper_generator_path)
        # if np.random.rand()>0.5: export_optm(gen_optimizer, dense_gripper_generator_GAN_optimizer_path)
        export_optm(contrastive_dis_optimizer, contrastive_discriminator_optimizer_path)
        export_model_state(contrastive_dis, contrastive_discriminator_path)
        global critic_optimizer
        global generator_optimizer
        global critic_net
        global generator_net
        critic_net=contrastive_dis
        generator_net=generator
        critic_optimizer=contrastive_dis_optimizer
        generator_optimizer=gen_optimizer
    def train_one_epoch():
        running_loss = 0.
        running_loss2 = 0.
        running_loss3 = 0.

        for i, batch in enumerate(dloader, 0):
            # continue
            pc,pose_7_positive, score,index= batch
            pc = pc.float().cuda(non_blocking=True)
            pose_7_positive=pose_7_positive.cuda(non_blocking=True).squeeze(1)
            score = score.cuda(non_blocking=True)[:,None]
            index = index.cuda(non_blocking=True)

            def dense_grasps_visualization2(pc,generated_pose_7):
                with torch.no_grad():
                    regular_dis = initialize_model(gripper_discriminator,dense_gripper_discriminator_path)
                    regular_dis.eval()
                    score,spatial_score=regular_dis(pc,generated_pose_7)
                    score_mask=score<0.3
                    score_mask=score_mask.squeeze()
                target_mask=get_spatial_mask(pc)
                total_mask=target_mask.squeeze() & score_mask
                # Method 1
                pose_good_grasp_list=[]

                # print(score_mask.shape)
                # print(generated_pose_7.shape)
                masked_generated_pose=generated_pose_7[:,:,total_mask]
                # print(masked_generated_pose.shape)
                masked_pc=pc[0,total_mask,0:3]
                masked_pc = masked_pc.cpu().numpy()

                for i in range(masked_pc.shape[0]):
                    if np.random.random()>0.05: continue
                    # random_index=np.random.randint(0,config.num_points)
                    # if target_mask[0,random_index,0]==False:continue

                    tmp_pose = masked_generated_pose[0:1, :, i]

                    poses_7 = tmp_pose[:, None, :].clone()


                    # visualize verfication
                    # masked_pc = masked_pc.cpu().numpy()  # + mean

                    center_point = masked_pc[i]
                    target_pose = poses_7[0, :, :]
                    approach = target_pose[:, 0:3]

                    theta, phi_sin, phi_cos = approach_vec_to_theta_phi(approach)
                    target_pose[:, 0:1] = theta
                    target_pose[:, 1:2] = phi_sin
                    target_pose[:, 2:3] = phi_cos
                    pose_5 = output_processing(target_pose[:, :, None]).squeeze(-1)

                    pose_good_grasp = decode_gripper_pose(pose_5, center_point[0:3])

                    # extreme_z = center_point[-1] - poses_7[0,0, -2] * config.distance_scope
                    # if  extreme_z < config.z_limits[0]: continue

                    pose_good_grasp_list.append(pose_good_grasp)
                if len(pose_good_grasp_list)==0: return
                pose_good_grasp=np.concatenate(pose_good_grasp_list,axis=0)


                vis_scene(pose_good_grasp[:, :], npy=pc[0,:,0:3].cpu().numpy())

                return
            def dense_grasps_visualization(pc,generated_pose_7):
                with torch.no_grad():
                    regular_dis = initialize_model(gripper_discriminator,dense_gripper_discriminator_path)
                    regular_dis.eval()
                    score,spatial_score=regular_dis(pc,generated_pose_7)
                target_mask=get_spatial_mask(pc)
                # Method 1
                pose_good_grasp_list=[]
                for i in range(5000):
                    random_index=np.random.randint(0,config.num_points)
                    if target_mask[0,random_index,0]==False:continue

                    if score[0,0,random_index]<0.4:continue

                    tmp_pose = generated_pose_7[0:1, :, random_index]

                    poses_7 = tmp_pose[:, None, :].clone()


                    # visualize verfication
                    pc_ = pc[0, :, 0:3].cpu().numpy()  # + mean

                    center_point = pc_[random_index]
                    target_pose = poses_7[0, :, :]
                    approach = target_pose[:, 0:3]

                    theta, phi_sin, phi_cos = approach_vec_to_theta_phi(approach)
                    target_pose[:, 0:1] = theta
                    target_pose[:, 1:2] = phi_sin
                    target_pose[:, 2:3] = phi_cos
                    pose_5 = output_processing(target_pose[:, :, None]).squeeze(-1)

                    pose_good_grasp = decode_gripper_pose(pose_5, center_point[0:3])

                    # extreme_z = center_point[-1] - poses_7[0,0, -2] * config.distance_scope
                    # if  extreme_z < config.z_limits[0]: continue

                    pose_good_grasp_list.append(pose_good_grasp)
                if len(pose_good_grasp_list)==0: return
                pose_good_grasp=np.concatenate(pose_good_grasp_list,axis=0)
                # print(pose_good_grasp.shape)
                # print(target_mask.squeeze().shape)
                # print(pc.shape)
                # masked_pc=pc[0,target_mask.squeeze(),0:3]
                # print(masked_pc.shape)
                # print(pc[0, :, 0:3].shape)

                vis_scene(pose_good_grasp[:, :], npy=pc[0, :, 0:3].cpu().numpy())

                return


                # Method 2

                # return 0
                mesh_list=[]
                for i in range(100):
                    random_index=np.random.randint(0,config.num_points)
                    if target_mask[0,random_index,0]==False:continue
                    tmp_pose = generated_pose_7[0:1, :, random_index]
                    mesh=get_mesh(pc[:, :, 0:3], tmp_pose.clone(), random_index)
                    mesh_list.append(mesh)

                scene = trimesh.Scene()
                scene.add_geometry([trimesh.PointCloud(pc[0, :, 0:3].cpu().numpy()), mesh_list])
                scene.show()

            def get_labels(pc,index):
                with torch.no_grad():
                    generated_pose_7= generator(pc[:,:,0:3])
                    sub_generated_pose=torch.stack([generated_pose_7[i, :, j] for i, j in enumerate(index)])

                collision_intensity = verify(pc[:, :, 0:3], sub_generated_pose, index, view=False)
                weight = 1.0 if collision_intensity > 0. else 0.0
                return generated_pose_7,sub_generated_pose,weight
            def dis_one_pass(pc,pose_7_positive,index,batch_index):
                contrastive_dis.zero_grad()

                b=pc.shape[0]
                generated_pose_7,sub_generated_pose,weight = get_labels(pc,index)


                if  (batch_index + 1 == len(dloader)):
                    generated_pose_std = torch.std(generated_pose_7, dim=-1)
                    generated_pose_max, _ = torch.max(generated_pose_7, dim=-1)
                    generated_pose_min, _ = torch.min(generated_pose_7, dim=-1)

                    generated_pose_ave = torch.mean(generated_pose_7, dim=-1)
                    generated_pose_cv = generated_pose_std / generated_pose_ave
                    print(Fore.YELLOW, f'generated pose std = {generated_pose_std}')
                    print(Fore.YELLOW, f'generated pose ave = {generated_pose_ave}')
                    print(Fore.YELLOW, f'generated pose max = {generated_pose_max}')
                    print(Fore.YELLOW, f'generated pose min = {generated_pose_min}')

                    print(Fore.YELLOW, f'generated pose cv = {generated_pose_cv}', Fore.RESET)
                    save_record(generated_pose_std.cpu().squeeze().numpy().tolist(), 'generated_pose_std.txt')
                    save_record(generated_pose_ave.cpu().squeeze().numpy().tolist(), 'generated_pose_ave.txt')
                    save_record(generated_pose_max.cpu().squeeze().numpy().tolist(), 'generated_pose_max.txt')
                    save_record(generated_pose_min.cpu().squeeze().numpy().tolist(), 'generated_pose_min.txt')
                    save_record(generated_pose_cv.cpu().squeeze().numpy().tolist(), 'generated_pose_cv.txt')

                pose_7_cat=torch.cat([pose_7_positive,sub_generated_pose],dim=0)
                index_cat=torch.cat([index,index],dim=0)
                dense_pose_7_cat=generated_pose_7.repeat(2,1,1)
                for ii, j in enumerate(index_cat):
                    dense_pose_7_cat[ii, :, j] = pose_7_cat[ii].clone()
                pc_cat=pc[:,:,0:3].repeat(2,1,1)


                contrastive_output=contrastive_dis(pc_cat,dense_pose_7_cat)

                masked_contrastive_scores = torch.stack([contrastive_output[i, :, j] for i, j in enumerate(index_cat)])

                positive_scores=masked_contrastive_scores[0:b].squeeze()
                generator_score=masked_contrastive_scores[b:2*b].squeeze()


                is_bad_pose=1.0
                if weight==0.0 :
                    ref_approach = torch.tensor([0.0, 0.0, 1.0], device=sub_generated_pose.device)[None, :]
                    new_vector = sub_generated_pose[:, 0:3]
                    cos_new = F.cosine_similarity(new_vector, ref_approach, dim=1)
                    if sub_generated_pose[0,-2]>0 and sub_generated_pose[0,-1]>0.0 and cos_new>0.5 and sub_generated_pose[0,-2]<=1.0 and sub_generated_pose[0,-1]<=1.0:
                        is_bad_pose=0.0

                good_approach=1 if sub_generated_pose[0,-2]>=pose_7_cat[0,-2] else 0
                if good_approach and is_bad_pose==0:
                    counter_x[2]+=1
                if is_bad_pose == 1.0:
                    counter_x[0]+=1
                else:
                    counter_x[1] += 1

                first_penalty=is_bad_pose*get_contrastive_loss(positive= positive_scores.clone(),negative= generator_score.clone(),margin=1.0) #*activate_fake_loss

                if good_approach:
                    second_penalty =  (1 - is_bad_pose) * get_contrastive_loss(
                        positive=generator_score.clone(), negative=positive_scores.clone(), margin=0.0)
                else:
                    second_penalty=(1-is_bad_pose)*get_contrastive_loss(positive= positive_scores.clone(),negative= generator_score.clone(),margin=0.0)

                third_penalty=torch.clamp(positive_scores.clone()-generator_score.clone()-1.0, min=0.0)

                loss=first_penalty+second_penalty+third_penalty

                if (batch_index + 1 == len(dloader)):
                    print('first_penalty = ',first_penalty.item(),'    second_penalty = ',second_penalty.item(),'    third_penalty = ',third_penalty.item())
                loss.backward()

                contrastive_dis_optimizer.step()
                contrastive_dis_optimizer.zero_grad()

                return loss.item(),third_penalty.item(),is_bad_pose

            def gen_one_pass(pc,pose_7_positive, index,batch_index,weight=None):
                generator.zero_grad()

                with torch.no_grad():
                    # r=np.random.rand()>0.5
                    # r=False
                    dense_pose_7 = torch.rand(size=(1, 7, config.num_points)).to('cuda')
                    for ii, j in enumerate(index):
                        dense_pose_7[ii, :, j] = pose_7_positive[ii].clone()
                    contrastive_output_label = contrastive_dis(pc[:,:,0:3], dense_pose_7)

                    contrastive_scores_label = torch.stack([contrastive_output_label[i, :, j] for i, j in enumerate(index)])

                generated_pose_7 = generator(pc[:,:,0:3].clone(),dense_pose_7.clone())

                with torch.no_grad():
                    quality_score, grasp_ability_score = regular_dis(pc[:,:,0:3].clone(), generated_pose_7)
                    target_score = quality_score[0, 0, index].item()
                    save_record(target_score, 'quality_score.txt')

                masked_pose=torch.stack([generated_pose_7[i, :, j] for i, j in enumerate(index)])

                if weight is None:
                    collision_intensity = verify(pc[:, :, 0:3], masked_pose, index, view=False)
                    weight = 1.0 if collision_intensity > 0. else 0.0

                ref_approach = torch.tensor([0.0, 0.0, 1.0], device=masked_pose.device)[None, :]
                new_vector = masked_pose[:, 0:3]
                cos_new = F.cosine_similarity(new_vector, ref_approach, dim=1)

                if masked_pose[0, -2] < 0.0 or masked_pose[0, -1] < 0.0 or cos_new<0.5:
                    weight = 1.0

                contrastive_output_pred = contrastive_dis(pc[:,:,0:3], generated_pose_7)

                masked_contrastive_score = torch.stack([contrastive_output_pred[i, :, j] for i, j in enumerate(index)])

                loss=torch.clamp(contrastive_scores_label-masked_contrastive_score+.0,min=0.)*weight
                loss+=torch.clamp(contrastive_scores_label-masked_contrastive_score-1.0,min=0.)*(1-weight)

                if (batch_index + 1 == len(dloader)):
                    print(f'Collision times = {counter_x[0]}, collision free times = {counter_x[1]}, good distance times = {counter_x[2]}')
                    print(f'generated pose = {masked_pose}')
                    print(f'label pose = {pose_7_positive}')
                    print('   ------ Generator loss = ', loss.item() )
                    print('-------------------------------------------------------------------')

                loss.backward()

                gen_optimizer.step()
                gen_optimizer.zero_grad()

                return  loss.item()

            loss1,loss2,weight=dis_one_pass(pc.clone(), pose_7_positive.clone(),  index,i)
            running_loss += loss1
            running_loss2 += loss2


            loss_gen=gen_one_pass(pc.clone(),pose_7_positive.clone(), index,i,weight)
            running_loss3 += loss_gen

            pi.step(i)

        return running_loss,running_loss2,running_loss3
    # try:
    for epoch in range(EPOCHS_):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1),max_limit=len(dloader))

        # with torch.autograd.set_detect_anomaly(True):
        running_loss,running_loss2,running_loss3= train_one_epoch()
        export_model()

        pi.end()
        n_batches=len(dloader)
        save_record(running_loss, 'discriminator_loss.txt')
        save_record(running_loss2, 'curriculum_loss.txt')
        save_record(running_loss3, 'generator_loss.txt')

        save_record(counter_x[0], 'collision_times.txt')
        save_record(counter_x[1], 'collision__free_times.txt')
        save_record(counter_x[2], 'good_approach_times.txt')


        print('   Total running loss 1= ',running_loss,', average total loss = ',running_loss/n_batches)
        print('   Total running loss 2= ',running_loss2,', average total loss = ',running_loss2/n_batches)
        print('   Total running loss 3= ',running_loss3,', average total loss = ',running_loss3/n_batches)


    global  counter
    global collision_accumulator
    global collision_ref_accumulator


    print(Fore.GREEN, 'Better generator is found', Fore.RESET)
    export_model()


    counter=0.
    collision_accumulator=0
    collision_ref_accumulator=0

def train_generator(n_samples=None):
    global online_samples_per_round
    if n_samples is not None:
        online_samples_per_round=n_samples
    training_data.remove_all_labeled_data()
    if len(training_data) == 0:
        load_training_data_from_online_pool(number_of_online_samples=online_samples_per_round,
                                        move_online_files=move_trained_data)
    train(EPOCHS, batch_size=BATCH_SIZE, directory=training_data.dir)
    training_data.remove_all_labeled_data()

if __name__ == "__main__":
    pass