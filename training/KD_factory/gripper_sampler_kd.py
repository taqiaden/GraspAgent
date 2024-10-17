import torch
from colorama import Fore
from torch import nn
from dataloaders.gripper_sampler_dl import gripper_sampler_dataset_kd, load_training_buffer
from lib.IO_utils import   custom_print
from lib.dataset_utils import  training_data
from lib.depth_map import depth_to_point_clouds
from lib.models_utils import initialize_model, export_model_state
from lib.optimizer import load_opt, export_optm
from lib.report_utils import progress_indicator
from models.point_net_base.GAGAN import gripper_generator, dense_gripper_generator_path
from registration import camera, transform_to_camera_frame
from models.Grasp_GAN import gripper_sampler_net,gripper_sampler_path


gripper_generator_optimizer_path=r'gripper_generator_optimizer'

training_data=training_data()
training_data.main_modality=training_data.depth
print=custom_print
BATCH_SIZE=1
learning_rate=5*1e-5
EPOCHS = 1
weight_decay = 0.000001
workers=2

l1_loss=nn.L1Loss(reduction='none')
mes_loss=nn.MSELoss()
cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

def knowledge_distillation():
    '''dataloader'''
    dataset = gripper_sampler_dataset_kd(data_pool=training_data)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    model=initialize_model(gripper_sampler_net,gripper_sampler_path)
    model.train(True)

    '''optimizer'''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    optimizer = load_opt(optimizer, gripper_generator_optimizer_path)

    '''distillation source model'''
    source_model = initialize_model(gripper_generator,dense_gripper_generator_path)
    source_model.eval()
    def train_one_epoch():
        running_loss = 0.

        for i, batch in enumerate(dloader, 0):
            depth= batch
            depth=depth.cuda().float()
            '''get predictions'''
            model.zero_grad()
            predictions=model(depth)


            loss=0.
            for j in range(BATCH_SIZE):
                '''generate labels'''
                pc, mask = depth_to_point_clouds(depth[j,0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                pc=torch.from_numpy(pc).to('cuda').float()
                with torch.no_grad():
                    labels=source_model(pc[None,:,:])
                prediction_=predictions[j].permute(1,2,0)[mask]
                labels=labels.squeeze().permute(1,0)
                '''compute loss'''
                approach_loss=(1-cos(prediction_[:,0:3],labels[:,0:3])**2).mean()
                beta_loss=(1-cos(prediction_[:,3:5],labels[:,3:5])**2).mean()
                dist_loss=mes_loss(prediction_[:,5],labels[:,5])
                width_loss=mes_loss(prediction_[:,6],labels[:,6])
                # print(f'approach loss= {approach_loss.item()}, beta loss={beta_loss.item()}, dist loss={dist_loss.item()}, width loss={width_loss.item()}')
                loss+=(approach_loss+beta_loss+dist_loss+width_loss*0.3)

            loss=loss/BATCH_SIZE
            print(loss.item())

            '''optimize'''
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pi.step(i)
            print()

        return running_loss

    for epoch in range(EPOCHS):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1),max_limit=len(dloader))

        running_loss= train_one_epoch()

        export_optm(optimizer, gripper_generator_optimizer_path)

        pi.end()

        print('   Running loss = ',running_loss,', loss per iteration = ',running_loss/len(dloader))

    return model

def train_gripper_sampler_kd(n_samples=None):
    while True:
        try:
            if len(training_data) == 0:
                load_training_buffer(size=n_samples)
            new_model = knowledge_distillation()
            print(Fore.GREEN + 'Training round finished' + Fore.RESET)
            export_model_state(new_model, gripper_sampler_path)
            training_data.clear()
        except Exception as e:
            print(Fore.RED, str(e), Fore.RESET)


if __name__ == "__main__":
    train_gripper_sampler_kd(10)