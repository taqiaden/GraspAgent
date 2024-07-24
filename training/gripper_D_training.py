import math
import random
from datetime import datetime
import numpy as np
import torch
from colorama import Fore
from torch.utils import data
from Configurations import config
from dataloaders.gripper_d_dataloader import gripper_dataset
from lib.loss.D_loss import custom_loss
from lib.optimizer import load_opt, export_optm
from models.GAGAN import gripper_generator, dense_gripper_generator_path
from models.gripper_D import dense_gripper_discriminator_path, gripper_discriminator
from pose_object import vectors_to_ratio_metrics,  output_processing, approach_vec_to_theta_phi
from suction_D_training import affordance_net, affordance_net_model_path
from dataset.load_test_data import estimate_suction_direction, random_sampling_augmentation
from lib.IO_utils import unbalance_check, update_balance_counter, move_single_labeled_sample, custom_print
from lib.bbox import   decode_gripper_pose
from lib.collision_unit import grasp_collision_detection
from lib.dataset_utils import rehearsal_data, training_data, online_data
from lib.grasp_utils import  update_pose_

from lib.models_utils import export_model_state, initialize_model_state, initialize_model
from lib.report_utils import  progress_indicator
from visualiztion import  vis_scene

dense_gripper_discriminator_optimizer_path = r'dense_gripper_discriminator_optimizer.pth.tar'

time_seed=math.floor(datetime.now().timestamp())
rehearsal_data=rehearsal_data()
training_data=training_data()
online_data=online_data()
selection_prob_k=1.0
print=custom_print
BATCH_SIZE=2
learning_rate=5*1e-6

EPOCHS = 1
weight_decay = 0.000001

augmentation_factor =1


move_trained_data=False
skip_unbalanced_samples = False

activate_balance_data=True

arbitrary_label=None

def get_collision_state(center_point,new_label,point_data,visualize=False):
    label = new_label
    label[:3] = center_point
    distance = label[22]
    width = label[21] / config.width_scale
    transformation = label[5:21].copy().reshape(-1, 4)
    transformation[0:3, 3] = label[:3] + transformation[0:3, 0] * distance
    label[5:21]=transformation.reshape(-1)
    transformation = label[5:21].copy().reshape(-1, 4)
    pose_good_grasp = update_pose_(transformation, width=width, distance=distance)
    collision_intensity = grasp_collision_detection(pose_good_grasp, point_data, visualize=visualize)
    # print(Fore.YELLOW,collision_intensity,Fore.RESET)
    label[3] = 0 if collision_intensity>0 else 1
    label[4] = 1
    label[23] = 0
    return label, collision_intensity


def load_training_data_from_online_pool(number_of_online_samples,generation_model=None,move_online_files=False):
    # counters to balance positive and negative
    # [n_grasp_positive,n_grasp_negative,n_suction_positive,n_suction_negative]
    # second, copy from online data to training data with down sampled point cloud
    generator = initialize_model(gripper_generator,dense_gripper_generator_path)
    generator.eval()
    discriminator=initialize_model(gripper_discriminator,dense_gripper_discriminator_path)

    discriminator.eval()

    suction_model = affordance_net()
    suction_model = initialize_model_state(suction_model, affordance_net_model_path)
    suction_model.eval()

    balance_indicator=0
    unbalance_allowance=0
    online_pc_filenames = online_data.get_pc_names()
    # assert len(online_pc_filenames) >= number_of_online_samples
    random.shuffle(online_pc_filenames)
    selection_p=get_selection_probabilty(online_data,online_pc_filenames)
    s_counter=0
    max_aug_from_suction=0.2*number_of_online_samples

    from lib.report_utils import progress_indicator as pi
    progress_indicator=pi(f'Get {number_of_online_samples} training data from online pool and apply maximum augmentation of {augmentation_factor} per sample : ',number_of_online_samples)

    balance_counter2 = np.array([0, 0, 0, 0])
    n = augmentation_factor
    counter=0
    for i,file_name in enumerate(online_pc_filenames):
        if np.random.rand() > selection_p[i]: continue
        sample_index=online_data.get_index(file_name)

        try:
            label=online_data.load_label_by_index(sample_index)
            # print(label.shape)
            # exit()
            # if label[4]==1 and load_grasp==False:continue
            # if label[23]==1 and load_suction==False: continue
            # if only_success and label[3]==0: continue
            if label[23] == 1:# and s_counter>max_aug_from_suction:
                continue
            if label[23] == 1 and label[3]==0:continue

            if label[3]==1 and balance_indicator>unbalance_allowance and label[4]==1:
                continue
            if ((label[4]==1 and label[3]==0) or (label[23] == 1)) and balance_indicator<-unbalance_allowance:
                continue

            point_data=online_data.load_pc(file_name)
        except Exception as e:
            print(Fore.RED, str(e),Fore.RESET)
            continue
        center_point = np.copy(label[:3])
        # print(balance_indicator)
        # assert -unbalance_allowance <= balance_indicator <= unbalance_allowance , f'{balance_indicator}'

        if activate_balance_data:
            balance_data= unbalance_check(label, balance_counter2)==0
            if skip_unbalanced_samples and not balance_data:
                continue

            n=augmentation_factor if balance_data else 1
        # for j in range(n):
        j=0
        down_sampled_pc, index = random_sampling_augmentation(center_point, point_data, config.num_points)
        if index is None:
            break
        ####################################################################################################

        normals = estimate_suction_direction(down_sampled_pc, view=False)
        down_sampled_pc = np.concatenate([down_sampled_pc, normals], axis=-1)

        label[:3] = down_sampled_pc[index, 0:3]

        global arbitrary_label


        ####################
        # get bad label from generator
        if label[3]==1:
            with torch.no_grad():
                pc_torch=torch.from_numpy(down_sampled_pc).to('cuda')[None,:,0:3].float()
                _,_,generated_poses=generator(pc_torch)

                # scores,_=discriminator(pc_torch,generated_poses)

            # max_score_index=torch.argmax(scores.squeeze()).item()

            # make the generated label
            # new_label = np.copy(label)
            # center_point = down_sampled_pc[max_score_index,0:3]
            # print(generated_poses.shape)
            generated_poses_5 = vectors_to_ratio_metrics(generated_poses)
            pose=generated_poses_5[:,:,index]
            pose_good_grasp = decode_gripper_pose(pose, center_point)
            # Transformation = get_homogenous_matrix(pose_good_grasp)
            # print(pose_good_grasp)
            # rotation_matrix=



            # pose_good_grasp[0, 0] = generated_width_ratio*config.width_scope
            collision_intensity = grasp_collision_detection(pose_good_grasp, point_data, visualize=False)
            if collision_intensity<=0. and label[3]==1:

                with torch.no_grad():
                    pc_torch = torch.from_numpy(down_sampled_pc).to('cuda')[None, :, :].float()
                    # _, _, dense_pose = generator(pc_torch)
                    suction_pred_ ,suction_pred_2 = suction_model(pc_torch)
                    target_score = suction_pred_[0, 0, index].item()

                    if target_score > 0.5:
                        print(Fore.RED, '   >', target_score, Fore.RESET)
                        continue
                    else:
                        print(Fore.GREEN, '   >', target_score, Fore.RESET)
                # print(Fore.GREEN, '-S', Fore.RESET)
            # elif collision_intensity>0. and label[3]==0:
            #     print(Fore.GREEN, '-F', Fore.RESET)
            else:
                print(Fore.RED, '-', Fore.RESET)
                continue

        ###########################################
        training_data.save_labeled_data(down_sampled_pc,label,sample_index + f'_aug{j}')

        if label[23]==1:
            arbitrary_label=None

        if label[3]==1:
            balance_indicator+=1
        else:
            # assert label[3]==0 , f'{label[3]}'
            balance_indicator-=1
        # if index is None:
        #     continue

        balance_counter2 = update_balance_counter(balance_counter2, is_grasp=label[4] == 1,score=label[3],n=n)

        if move_online_files:
            # move copied labeled_data from online data to rehearsal data
            move_single_labeled_sample(online_data, rehearsal_data, file_name)

        counter+=1
        progress_indicator.step(counter)

        if counter >= number_of_online_samples: break

    # view_data_summary(balance_counter=balance_counter2)
    return balance_counter2

def verify(pc_data,poses_7,idx,evaluation_metric=None,predictions=None):
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
        vis_scene(pose_good_grasp[:, :].reshape(1, 14), npy=pc_[:,0:3])

def train(EPOCHS_,batch_size,directory):
    dataset = gripper_dataset(num_points=config.num_points, path=directory)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    model=initialize_model(gripper_discriminator,dense_gripper_discriminator_path)

    model.train(False)

    # d=0.5
    # model_list=[model.back_bone,model.dis,model.dis3,model.spatial_dist,model.grasp_ability_]
    # lr_list=[learning_rate*d,learning_rate,learning_rate,learning_rate,learning_rate]
    # optimizer = decayed_optimizer(model_list,lr_list=lr_list,decay_rate=d ,use_RMSprop=False)
    # optimizer = decayed_optimizer(model,lr_=learning_rate,decay_rate=0.99999 ,use_RMSprop=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    # optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    optimizer = load_opt(optimizer, dense_gripper_discriminator_optimizer_path)


    def train_one_epoch():
        running_loss = 0.
        running_loss2 = 0.

        for i, batch in enumerate(dloader, 0):
            pc,pose, score,index= batch
            pc = pc.float().cuda(non_blocking=True)
            pose_7=pose.cuda(non_blocking=True).squeeze(1)
            score = score.cuda(non_blocking=True)[:,None]
            index = index.cuda(non_blocking=True)
            # depth_image = depth_image.cuda(non_blocking=True).float()
            # spatial_information = spatial_information.cuda(non_blocking=True).float()

            dense_pose_7 = torch.rand(size=(pc.shape[0], 7, config.num_points)).to('cuda')
            # print(pose)

            for ii, j in enumerate(index):
                dense_pose_7[ii, :, j] = pose_7[ii].clone()

            model.zero_grad()

            quality_score,grasp_ability_score=model(pc,dense_pose_7,use_collision_module=False,use_quality_module=True)
            # with torch.no_grad():
            #     r=representation_net_.back_bone(pc[:,:,0:3])
            # loss=mse_loss(quality_score,r.clone().detach())
            # loss2=mse_loss(quality_score2,r.clone().detach())

            masked_predictions = torch.stack([quality_score[i, :, j] for i, j in enumerate(index)])
            masked_predictions2 = torch.stack([grasp_ability_score[i, :, j] for i, j in enumerate(index)])

            # masked_predictions = torch.stack([predictions[ii, :, x,y] for ii,(x,y) in enumerate(zip(x_index,y_index))])
            # print(masked_predictions)
            # print(score)
            # exit()\apply
            if np.random.random()>0.95:
                print(f'Label = {score}')
                print(f'prediction = {masked_predictions}')
                print('---------------------')

            # verify(pc, pose, index, evaluation_metric=score,predictions=masked_predictions)

            loss=custom_loss(masked_predictions,score.clone())
            loss2=custom_loss(masked_predictions2,score.clone())


            running_loss+=loss.item()
            running_loss2+=loss2.item()
            loss=loss+loss2
            loss.backward()
            optimizer.step()



            pi.step(i)

        return running_loss,running_loss2

    for epoch in range(EPOCHS_):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1),max_limit=len(dloader))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        # with torch.autograd.set_detect_anomaly(True):
        running_loss,running_loss2= train_one_epoch()
        export_optm(optimizer, dense_gripper_discriminator_optimizer_path)

        pi.end()
        n_batches=len(dloader)
        print('   Total running loss = ',running_loss,', average total loss = ',running_loss/n_batches)
        print('   Total running loss 1= ',running_loss2,', average total loss = ',running_loss2/n_batches)

    return model

def train_gripper(n_samples=None):
    if n_samples is not None:
        online_samples_per_round=n_samples
    #seeds(time_seed)

    training_data.remove_all_labeled_data()
    if len(training_data) == 0:
        load_training_data_from_online_pool(number_of_online_samples=online_samples_per_round,
                                                               generation_model=None, move_online_files=move_trained_data)
    new_model = train(EPOCHS, batch_size=BATCH_SIZE,directory=training_data.dir)
    print(Fore.GREEN + 'A better model state is found for gripper head' + Fore.RESET)
    export_model_state(new_model, dense_gripper_discriminator_path)
    training_data.remove_all_labeled_data()
if __name__ == "__main__":
    pass