import math
from datetime import datetime

import numpy as np
import torch
import trimesh
from colorama import Fore
from torch import nn
from torch.utils import data
from Configurations import config
from dataloaders.opening_GAN_dataloader import opening_dataset, load_training_data_from_online_pool
from lib.IO_utils import  custom_print
from lib.bbox import   decode_gripper_pose
from lib.collision_unit import  grasp_collision_detection
from lib.dataset_utils import training_data
from lib.grasp_utils import  update_pose_,  get_homogenous_matrix
from lib.mesh_utils import construct_gripper_mesh
from lib.models_utils import export_model_state, initialize_model
from lib.optimizer import export_optm, load_opt
from lib.report_utils import  progress_indicator
from models.GAGAN import dense_gripper_generator_path, \
     gripper_generator
from models.gripper_D import gripper_discriminator, dense_gripper_discriminator_path
from models.opening_GAN import opening_generator_path, opening_critic_path, opening_critic, opening_generator
from pose_object import output_processing, approach_vec_to_theta_phi
from records.records_managment import save_record
from visualiztion import vis_scene

opening_critic_optimizer_path = r'opening_critic_optimizer.pth.tar'
opening_generator_optimizer_path = r'opening_generator_optimizer.pth.tar'

time_seed=math.floor(datetime.now().timestamp())
# rehearsal_data=rehearsal_data()
# training_data=training_data()
training_data = training_data()

weight_decay = 0.0000001
print=custom_print
BATCH_SIZE=1
# accum_iter=3 # the frequency of updating the gradient
learning_rate=5*1e-5

EPOCHS=1
augmentation_factor =1
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

critic_optimizer=None
generator_optimizer=None
generator_net=None
critic_net=None
def train(EPOCHS_,batch_size,directory,activate_generator_training=True):
    global critic_optimizer
    global generator_optimizer
    global critic_net
    global generator_net

    counter_x=[0,0,0]


    gripper_pose_model=initialize_model(gripper_generator,dense_gripper_generator_path)
    gripper_pose_model.eval()
    if critic_net is None:
        critic_net=initialize_model(opening_critic,opening_critic_path)
        critic_net.train(False)


    if generator_net is None:
        generator_net = initialize_model(opening_generator,opening_generator_path)
        generator_net.train(False)


    dataset = opening_dataset(num_points=config.num_points, path=directory)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)


    #contrastive_dis_optimizer = decayed_optimizer(contrastive_dis,lr_=learning_rate,decay_rate=0.99999,use_RMSprop=False,use_sgd=False)
    if critic_optimizer is None:
        critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        # contrastive_dis_optimizer=torch.optim.SGD(contrastive_dis.parameters(), lr=learning_rate, weight_decay=weight_decay)

        critic_optimizer = load_opt(critic_optimizer, opening_critic_optimizer_path)


    if generator_optimizer is None:
        # gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        generator_optimizer=torch.optim.SGD(generator_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        generator_optimizer = load_opt(generator_optimizer, opening_generator_optimizer_path)

    # bins = [2, 2, 2, 1, 1, 2, 2]
    def export_model():
        global critic_optimizer
        global generator_optimizer
        global critic_net
        global generator_net
        print(Fore.GREEN, 'Export check point', Fore.RESET)
        export_optm(generator_optimizer, opening_generator_optimizer_path)
        export_model_state(generator_net, opening_generator_path)
        export_optm(critic_optimizer, opening_critic_optimizer_path)
        export_model_state(critic_net, opening_critic_path)

    def train_one_epoch():
        print(Fore.CYAN, f'Launch one epoch of gripper opening training  ', Fore.RESET)
        running_loss = 0.
        running_loss2 = 0.
        running_loss3 = 0.

        for i, batch in enumerate(dloader, 0):
            # continue
            pc,pose_7_positive, score,index= batch
            pc = pc.float().cuda(non_blocking=True)
            pose_7_positive=pose_7_positive.cuda(non_blocking=True).squeeze(1) # ()
            pose_6_positive=pose_7_positive[:,0:-1].float()  # (1,6)
            positive_opening=pose_7_positive[:,-1:]#(1,1)

            score = score.cuda(non_blocking=True)[:,None]
            index = index.cuda(non_blocking=True)

            def get_spatial_mask(pc):
                x = pc[:,:, 0:1]
                y = pc[:,:, 1:2]
                z = pc[:,:, 2:3]
                x_mask = (x > 0.280 + 0.00) & (x < 0.582 - 0.00)
                y_mask = (y > -0.21 + 0.00) & (y < 0.21 - 0.00)
                z_mask = (z > config.z_limits[0]) & (z < config.z_limits[1])
                # print(z)
                spatial_mask = x_mask & y_mask & z_mask
                return spatial_mask
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


            def get_labels(pc,index,dense_grasp_pose_6):
                with torch.no_grad():
                    generated_opening= generator_net(pc[:,:,0:3],dense_grasp_pose_6)

                sub_generated_opening=torch.stack([generated_opening[i, :, j] for i, j in enumerate(index)])
                sub_grasp_pose=torch.stack([dense_grasp_pose_6[i, :, j] for i, j in enumerate(index)])
                sub_parameters=torch.cat([sub_grasp_pose,sub_generated_opening],dim=-1) # (1,1,n)

                dense_parameters=torch.cat([dense_grasp_pose_6,generated_opening],dim=1) # (1,7,n)

                width=sub_generated_opening.squeeze().item()
                print(f'Generated opening = {width}, label opening = {positive_opening.item()}')
                if width>1.0 or width <0.0: weight=1.0
                else:
                    collision_intensity = verify(pc[:, :, 0:3], sub_parameters, index, view=False)
                    weight = 1.0 if collision_intensity > 0. else 0.0

                return dense_parameters,sub_parameters,generated_opening,weight

            def get_dense_grasp_pose_6(pose_6_positive):
                with torch.no_grad():
                    generated_pose_7_softmax,generated_pose_7_tanh,generated_pose_7_direct= gripper_pose_model(pc[:,:,0:3])

                    generated_pose_6=generated_pose_7_direct[:,0:-1,:] # (


                    generated_pose_6[0,:,index]=pose_6_positive.transpose(0,1)
                return generated_pose_6
            def dis_one_pass(pc,pose_7_positive,score,index,batch_index):
                critic_net.zero_grad()
                dense_pose_6=get_dense_grasp_pose_6(pose_6_positive) # (1,6,n)

                b=pc.shape[0]
                generated_pose_7,sub_generated_pose,generated_opening,weight = get_labels(pc,index,dense_pose_6)

                dense_pose_6[0,:,index]=pose_6_positive.transpose(0,1)
                dense_pose_6_cat=dense_pose_6.repeat(2,1,1)

                dense_opening_cat=generated_opening.repeat(2,1,1)
                dense_opening_cat[0,...]=positive_opening.transpose(0,1)

                index_cat=torch.cat([index,index],dim=0)

                pc_cat=pc[:,:,0:3].repeat(2,1,1)


                critic_scores=critic_net(pc_cat,dense_pose_6_cat,dense_opening_cat)
                # masked_p = torch.stack([dense_pose_6_cat[i, :, j] for i, j in enumerate(index_cat)])
                # masked_o = torch.stack([dense_opening_cat[i, :, j] for i, j in enumerate(index_cat)])
                # print(masked_p)
                # print(masked_o)


                masked_critic_scores = torch.stack([critic_scores[i, :, j] for i, j in enumerate(index_cat)])

                positive_scores=masked_critic_scores[0:b].squeeze()
                generator_score=masked_critic_scores[b:2*b].squeeze()
                print(f'positive_scores = {positive_scores.item()}, generator_score = {generator_score.item()}')
                # print(pose_6_positive)
                # exit()

                if weight == 1.0:
                    counter_x[0]+=1
                else:
                    counter_x[1] += 1

                loss=weight*get_contrastive_loss(positive= positive_scores.clone(),negative= generator_score.clone(),margin=1.0) #*activate_fake_loss



                # if (batch_index + 1 == len(dloader)):
                print('critic loss = ',loss.item())

                loss.backward()

                critic_optimizer.step()
                critic_optimizer.zero_grad()

                return loss.item(),0,weight,dense_pose_6,positive_scores.detach()

            def gen_one_pass(pc,representation,pose_7_positive, index,batch_index,dense_pose_6,positive_scores):
                generator_net.zero_grad()

                generated_opening = generator_net(pc[:, :, 0:3], dense_pose_6)

                critic_scores=critic_net(pc[:,:,0:3],dense_pose_6,generated_opening)
                masked_score=torch.stack([critic_scores[i, :, j] for i, j in enumerate(index)])

                # loss=masked_score.detach().clone()+1-masked_score
                loss=torch.clamp(positive_scores-masked_score,min=0.0)
                print(f'Generator loss = {loss.item()}')

                loss.backward()

                generator_optimizer.step()
                generator_optimizer.zero_grad()

                return  loss.item()

            loss1,loss2,weight,dense_pose_6,positive_scores=dis_one_pass(pc.clone(), pose_7_positive.clone(), score.clone(),  index,i)
            running_loss += loss1
            running_loss2 += loss2

            if activate_generator_training and weight==1.0 :
                loss_gen=gen_one_pass(pc.clone(),None,pose_7_positive.clone(), index,i,dense_pose_6,positive_scores)
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

    # except Exception as e:
    #         print(str(e))
    global  counter
    # global  entropy_accumulator
    # global  entropy_ref_accumulator
    global collision_accumulator
    global collision_ref_accumulator


    print(Fore.GREEN, 'Better generator is found', Fore.RESET)
    export_model()


    counter=0.

    collision_accumulator=0
    collision_ref_accumulator=0


def train_opening_GAN(n_samples=None,activate_generator_training=True):
    #seeds(time_seed)
    global online_samples_per_round
    if n_samples is not None:
        online_samples_per_round=n_samples
    # training_data.remove_all_labeled_data()
    if len(training_data) == 0:
        load_training_data_from_online_pool(number_of_online_samples=online_samples_per_round)
    train(EPOCHS, batch_size=BATCH_SIZE, directory=training_data.dir,activate_generator_training=activate_generator_training)
    training_data.remove_all_labeled_data()

if __name__ == "__main__":
    pass