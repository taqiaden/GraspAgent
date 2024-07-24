import copy
import math
import random
from datetime import datetime
import numpy as np
import torch
from colorama import Fore
from torch.utils import data
from dataloaders.suction_d_dataloader import suction_dataset
from lib.loss.D_loss import custom_loss
from lib.optimizer import export_optm, load_opt
from dataset.load_test_data import estimate_suction_direction, random_sampling_augmentation
from lib.IO_utils import  unbalance_check, update_balance_counter, move_single_labeled_sample, custom_print
from lib.models_utils import initialize_model
from models.GAGAN import gripper_generator, dense_gripper_generator_path
from Configurations import config
from lib.dataset_utils import rehearsal_data,training_data,online_data
from lib.report_utils import   progress_indicator
from lib.models_utils import  export_model_state, initialize_model_state

from models.gripper_D import gripper_discriminator, dense_gripper_discriminator_path
from models.suction_D import affordance_net, affordance_net_model_path

time_seed=math.floor(datetime.now().timestamp())
rehearsal_data=rehearsal_data()
training_data=training_data()
online_data=online_data()
selection_prob_k=1.0

print=custom_print
BATCH_SIZE=2
lr=5*1e-6
weight_decay = 0.000001

EPOCHS = 1
augmentation_factor =1
online_samples_per_round=500
move_trained_data=False
skip_unbalanced_samples = True

activate_balance_data=True

only_success=False

suction_discriminator_optimizer_path = r'suction_discriminator_optimizer.pth.tar'


def load_training_data_from_online_pool(number_of_online_samples,generation_model=None,move_online_files=False,load_grasp=True,load_suction=True):
    regular_dis = initialize_model(gripper_discriminator,dense_gripper_discriminator_path)

    regular_dis.eval()
    generator = initialize_model(gripper_generator,dense_gripper_generator_path)
    generator.eval()
    # counters to balance positive and negative
    # [n_grasp_positive,n_grasp_negative,n_suction_positive,n_suction_negative]
    # second, copy from online data to training data with down sampled point cloud
    balance_indicator=0.0
    unbalance_allowance=0
    online_pc_filenames = online_data.get_pc_names()
    # assert len(online_pc_filenames) >= number_of_online_samples
    random.shuffle(online_pc_filenames)
    selection_p=get_selection_probabilty(online_data,online_pc_filenames)

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
            if label[4]==1 and load_grasp==False:continue
            if label[23]==1 and load_suction==False: continue
            if only_success and label[3]==0: continue
            if label[3]==1 and balance_indicator>unbalance_allowance:
                continue
            elif label[3]==0 and balance_indicator<-1*unbalance_allowance:
                continue

            point_data=online_data.load_pc(file_name)
        except Exception as e:
            print(Fore.RED, str(e),Fore.RESET)
            continue
        center_point = np.copy(label[:3])
        if activate_balance_data:
            balance_data= unbalance_check(label, balance_counter2)==0
            if skip_unbalanced_samples and not balance_data:continue

            n=augmentation_factor if balance_data else 1

        # for j in range(n):
        j=0
        # print(j)
        # print(center_point)

        down_sampled_pc, index = random_sampling_augmentation(center_point, point_data, config.num_points)
        if index is None:
            break

        # data-centric adversial measure
        if label[3] == 1:
            with torch.no_grad():
                pc_torch = torch.from_numpy(down_sampled_pc).to('cuda')[None, :, 0:3].float()
                _, _, dense_pose = generator(pc_torch)
                quality_score, grasp_ability_score = regular_dis(pc_torch, dense_pose)
                target_score = quality_score[0, 0, index].item()

                if target_score > 0.5:
                    print(Fore.RED, '   >', target_score, Fore.RESET)
                    continue
                else:
                    print(Fore.GREEN, '   >', target_score, Fore.RESET)



        normals = estimate_suction_direction(down_sampled_pc, view=False)
        down_sampled_pc=np.concatenate([down_sampled_pc,normals],axis=-1)

        label[:3] = down_sampled_pc[index,0:3]
        training_data.save_labeled_data(down_sampled_pc,label,sample_index + f'_aug{j}')
        if label[3]==1:
            balance_indicator+=1
        else:
            balance_indicator-=1


        if index is None:
            continue

        balance_counter2 = update_balance_counter(balance_counter2, is_grasp=label[4] == 1,score=label[3],n=n)
        # try:
        #     if generation_model:
        #         sample_index=get_sample_index(file_name)
        #         with torch.no_grad():
        #             for i in range(n):
        #                 pc = online_data.load_pc_with_mean(file_name)
        #                 generation_model.eval()
        #                 pc = pc.cuda(non_blocking=True)
        #                 if label[4] == 1: # for grasp ----------
        #                     generated_label = get_grasp_label(pc, generation_model,get_low_score=label[3]==0)
        #                 else: # for suction -----------------
        #                     generated_label = get_suction_label(pc, generation_model,get_low_score=label[3]==0)
        #                 pc_ = pc.detach().cpu().numpy().squeeze()
        #                 # save the generated label data
        #                 training_data.save_labeled_data(pc_,generated_label,sample_index + f'_gen{i}')
        #
        #         # The score in the generated label is not 1, but we select the highest score generated by the model.
        #         # for simplicity the balance counter will be updated considering the score is equal to 1 means success grasp/suction
        #         balance_counter2 = update_balance_counter(balance_counter2, is_grasp=label[4] == 1,score=label[3],n=n)
        # except Exception as e:
        #     print(Fore.RED,'Unable to generate labeled data. exception message: ',str(e),Fore.RESET)
        if move_online_files:
            # move copied labeled_data from online data to rehearsal data
            move_single_labeled_sample(online_data, rehearsal_data, file_name)

        counter+=1
        progress_indicator.step(counter)

        if counter >= number_of_online_samples: break

    # view_data_summary(balance_counter=balance_counter2)
    return balance_counter2


def train(EPOCHS_,model,batch_size,directory):
    dataset = suction_dataset(num_points=config.num_points, path=directory)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    model.train(False)

    # parameters=[]
    # parameters+=decay_lr(min_lr=lr*0.7,max_lr=lr,model=model)
    # optimizer = torch.optim.SGD(parameters, lr=lr,  weight_decay=WEIGHT_DECAY)
    # d=0.5
    # model_list=[model.back_bone,model.decoder,model.back_bone2,model.get_feature1,model.get_feature2,model.get_feature3,model.fc,model.decoder2]
    # lr_list=[lr*d,lr*1.0,lr*d*d*d,lr*d,lr*d*d,lr*d*d,lr*d,lr]
    # optimizer = decayed_optimizer(model_list,lr_list=lr_list,decay_rate=d,use_RMSprop=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    # optimizer=torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = decayed_optimizer(model,lr_=lr,decay_rate=0.999999,use_RMSprop=False)

    # optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)

    optimizer=load_opt(optimizer,suction_discriminator_optimizer_path)

    def train_one_epoch():
        running_loss = 0.
        running_loss2 = 0.

        for i, batch in enumerate(dloader, 0):
            pc,normal_label,score,index= batch
            pc = pc.float().cuda(non_blocking=True)
            normal_label=normal_label.cuda(non_blocking=True)
            score = score.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)

            # train suction pose network
            # train_suction_pose_(suction_cls_label,pc_n,idx,normal_label,suction_score_label,mean_point,view=False)

            for ii, j in enumerate(index):
                # print(pc[i,j,3:6])
                # print(normal_label[i])
                pc[ii,j,3:6]=normal_label[ii]

            suction_pred_ ,suction_pred_2= model(pc)
            suction_score_pred = torch.stack([suction_pred_[i, 0, j] for i, j in enumerate(index)])
            suction_score_pred2 = torch.stack([suction_pred_2[i, 0, j] for i, j in enumerate(index)])

            loss = custom_loss(suction_score_pred, score.clone())
            loss2 = custom_loss(suction_score_pred2, score.clone())

            # print(loss.item())
            # print(loss2.item())

            # print('-------------------------')
            # for i in range(pc.shape[0]):
            #     pc_instance=pc[i]
            #     id=index[i]
            #     pre=suction_score_pred[i]
            #     truth=score[i]
            #     # if abs(pre-truth)<0.5:continue
            #     print(f'prediction={pre},  truth={truth}')
            #
            #     suction_xyz=pc_instance[id]
            #     normal=np.array([.0,0.,1.0])
            #     pc_instance=pc_instance.cpu().numpy()
            #     suction_xyz, pre_grasp_mat, end_effecter_mat, suction_pose, T, pred_approch_vector \
            #         = get_suction_pose(id, pc_instance, normal)
            #     visualize_suction_pose(suction_xyz, suction_pose, T, end_effecter_mat, npy=pc_instance)

            pi.step(i)
            running_loss += loss.item()
            running_loss2 += loss2.item()
            loss=loss+loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return running_loss,running_loss2


    for epoch in range(EPOCHS_):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1),max_limit=len(dloader))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        with torch.autograd.set_detect_anomaly(True):
            running_loss,running_loss2= train_one_epoch()

        pi.end()
        n_batches=len(dloader)
        print('   Total running loss = ',running_loss,', average total loss = ',running_loss/n_batches)
        print('   Total running loss = ',running_loss2,', average total loss = ',running_loss2/n_batches)

        # if first_loss is None:first_loss=running_loss if running_loss!=0 else 1e-5
        # elif running_loss/first_loss<termination_loss_ratio:break
    export_optm(optimizer,suction_discriminator_optimizer_path)

    return model

def train_suction(n_samples=None):
    #seeds(time_seed)
    global online_samples_per_round
    if n_samples is not None:
        online_samples_per_round=n_samples
    training_data.remove_all_labeled_data()
    if len(training_data) == 0:
        load_training_data_from_online_pool(number_of_online_samples=online_samples_per_round,
                                                               generation_model=None, move_online_files=move_trained_data,
                                                               load_grasp=False, load_suction=True)
    current_model = affordance_net()
    try:
        current_model = initialize_model_state(current_model, affordance_net_model_path)
    except Exception as e:
        print(Fore.RED,f'model is not initialized, error msg: {str(e)}',Fore.RESET)
    new_model = copy.deepcopy(current_model)

    new_model = train(EPOCHS, new_model, batch_size=BATCH_SIZE, directory=training_data.dir)
    print(Fore.GREEN + 'A better model state is found for suction head' + Fore.RESET)
    export_model_state(new_model, affordance_net_model_path)
    training_data.remove_all_labeled_data()

if __name__ == "__main__":
    pass