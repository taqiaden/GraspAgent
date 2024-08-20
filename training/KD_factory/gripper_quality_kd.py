import torch
from colorama import Fore
from torch import nn
from dataloaders.gripper_quality_dl import gripper_quality_dataset_kd, load_training_buffer_kd
from lib.IO_utils import   custom_print
from lib.dataset_utils import  training_data
from lib.depth_map import depth_to_point_clouds
from lib.models_utils import initialize_model, export_model_state
from lib.optimizer import load_opt, export_optm
from lib.report_utils import progress_indicator
from models.gripper_D import dense_gripper_discriminator_path, gripper_discriminator
from models.gripper_quality import gripper_quality_model_state_path, gripper_quality_net, gripper_scope_net, \
    gripper_scope_model_state_path
from models.gripper_sampler import gripper_sampler_net, gripper_generator_model_state_path
from registration import camera, transform_to_camera_frame

gripper_quality_optimizer_path=r'gripper_quality_optimizer'

gripper_scope_optimizer_path=r'gripper_scope_optimizer'

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

def knowledge_distillation():
    '''dataloader'''
    dataset = gripper_quality_dataset_kd(data_pool=training_data)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''models'''
    model=initialize_model(gripper_quality_net,gripper_quality_model_state_path)
    model.train(True)
    scope_model=initialize_model(gripper_scope_net,gripper_scope_model_state_path)
    scope_model.train(True)

    '''grasp generator'''
    generator = initialize_model(gripper_sampler_net, gripper_generator_model_state_path)
    generator.eval()

    '''optimizer'''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    optimizer = load_opt(optimizer, gripper_quality_optimizer_path)
    scope_optimizer = torch.optim.Adam(scope_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    scope_optimizer = load_opt(scope_optimizer, gripper_scope_optimizer_path)

    '''distillation from source model'''
    source_model = initialize_model(gripper_discriminator,dense_gripper_discriminator_path)
    source_model.eval()
    def train_one_epoch():
        running_loss = 0.

        for i, batch in enumerate(dloader, 0):
            depth= batch
            depth=depth.cuda().float()

            '''generate pose'''
            with torch.no_grad():
                generated_grasps=generator(depth)

            '''zero grad'''
            model.zero_grad()
            scope_model.zero_grad()

            '''get predictions'''
            grasp_score=model(depth,generated_grasps)
            scope_score=scope_model(generated_grasps)
            predictions = grasp_score * scope_score

            loss=0.
            for j in range(BATCH_SIZE):
                '''generate labels'''
                pc, mask = depth_to_point_clouds(depth[j,0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                pc=torch.from_numpy(pc).to('cuda').float()
                target_grasps=generated_grasps[j]
                target_grasps=target_grasps[None,:,mask]
                with torch.no_grad():
                    labels=source_model(pc[None,:,:],target_grasps)
                prediction_=predictions[j].permute(1,2,0)[mask].squeeze()
                labels=labels.squeeze()
                '''compute loss'''
                loss+=mes_loss(prediction_,labels)

            loss=loss/BATCH_SIZE
            print(loss.item())

            '''optimizer step'''
            loss.backward()
            optimizer.step()
            scope_optimizer.step()

            running_loss += loss.item()
            pi.step(i)
            print()

        return running_loss

    for epoch in range(EPOCHS):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1),max_limit=len(dloader))

        running_loss= train_one_epoch()

        export_optm(optimizer, gripper_quality_optimizer_path)
        export_optm(scope_optimizer, gripper_scope_optimizer_path)

        pi.end()

        print('   Running loss = ',running_loss,', loss per iteration = ',running_loss/len(dloader))

    return model, scope_model


def train_gripper_sampler_kd(n_samples=None):
    # training_data.clear()
    while True:
        # try:
        if len(training_data) == 0:
            load_training_buffer_kd(size=n_samples)
        new_model,scope_model = knowledge_distillation()
        print(Fore.GREEN + 'Training round finished' + Fore.RESET)
        export_model_state(new_model, gripper_quality_model_state_path)
        export_model_state(scope_model, gripper_scope_model_state_path)
        training_data.clear()
        # except Exception as e:
        #     print(Fore.RED, str(e), Fore.RESET)



if __name__ == "__main__":
    train_gripper_sampler_kd(10)