import copy
import torch
from colorama import Fore
from torch.utils import data
from dataloaders.suction_d_dataloader import suction_dataset, load_training_data_from_online_pool
from lib.loss.D_loss import custom_loss
from lib.optimizer import export_optm, load_opt
from lib.IO_utils import   custom_print
from Configurations import config
from lib.dataset_utils import training_data
from lib.report_utils import   progress_indicator
from lib.models_utils import  export_model_state, initialize_model_state
from models.point_net_base.suction_D import affordance_net, affordance_net_model_path

training_data=training_data()

print=custom_print
BATCH_SIZE=2
lr=5*1e-6
weight_decay = 0.000001
EPOCHS = 1
online_samples_per_round=500

suction_discriminator_optimizer_path = r'suction_discriminator_optimizer.pth.tar'

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
        load_training_data_from_online_pool(number_of_online_samples=online_samples_per_round )
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