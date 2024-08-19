import numpy as np
import torch
from colorama import Fore
from torch import nn

from Verfication_tests.suction_verf import view_suction_label
from dataloaders.suction_quality_dl import suction_quality_dataset_kd, load_training_buffer, load_training_buffer_kd, \
    suction_quality_dataset

from lib.IO_utils import   custom_print
from lib.dataset_utils import  training_data
from lib.depth_map import depth_to_point_clouds, pixel_to_point
from lib.loss.D_loss import custom_loss
from lib.models_utils import initialize_model, export_model_state
from lib.optimizer import load_opt, export_optm
from lib.report_utils import progress_indicator
from models.suction_D import affordance_net_model_path, affordance_net
from models.suction_quality import suction_quality_net, suction_quality_model_state_path, suction_scope_net, \
    suction_scope_model_state_path
from models.suction_sampler import suction_sampler_net, suction_sampler_model_state_path
from registration import camera, transform_to_camera_frame
from visualiztion import visualize_suction_pose

suction_quality_optimizer_path=r'suction_quality_optimizer'

suction_scope_optimizer_path=r'suction_scope_optimizer'

training_data=training_data()
training_data.main_modality=training_data.depth
print=custom_print
BATCH_SIZE=1
learning_rate=5*1e-5
EPOCHS = 1
weight_decay = 0.000001
workers=1

l1_loss=nn.L1Loss(reduction='none')
mes_loss=nn.MSELoss()

def knowledge_distillation():
    '''dataloader'''
    dataset = suction_quality_dataset_kd(data_pool=training_data)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    model=initialize_model(suction_quality_net,suction_quality_model_state_path)
    model.train(True)
    scope_model=initialize_model(suction_scope_net,suction_scope_model_state_path)
    scope_model.train(True)

    '''suction generator'''
    generator = initialize_model(suction_sampler_net, suction_sampler_model_state_path)
    generator.eval()

    '''optimizer'''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    optimizer = load_opt(optimizer, suction_quality_optimizer_path)
    scope_optimizer = torch.optim.Adam(scope_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=weight_decay)
    scope_optimizer = load_opt(scope_optimizer, suction_scope_optimizer_path)

    '''distillation source model'''
    source_model = initialize_model(affordance_net,affordance_net_model_path)
    source_model.eval()

    def train_one_epoch():
        running_loss = 0.

        for i, batch in enumerate(dloader, 0):
            depth= batch
            depth=depth.cuda().float()

            '''generate pose'''
            with torch.no_grad():
                generated_normals=generator(depth)

            '''zero grad'''
            model.zero_grad()
            scope_model.zero_grad()

            '''get predictions'''
            suction_score = model(depth, generated_normals)
            scope_score = scope_model(generated_normals)
            predictions = suction_score * scope_score

            loss=0.
            for j in range(BATCH_SIZE):
                '''generate labels'''
                pc, mask = depth_to_point_clouds(depth[j,0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                pc=torch.from_numpy(pc).to('cuda').float()
                normals=generated_normals[j]
                normals=normals[None,:,mask]
                pc_with_normals=torch.cat([pc[None,:,:],normals.transpose(1,2)],dim=-1)
                with torch.no_grad():
                    labels=source_model(pc_with_normals)
                prediction_=predictions[j].permute(1,2,0)[mask].squeeze()
                labels=labels.squeeze()
                '''compute loss'''
                loss+=mes_loss(prediction_,labels)

            loss=loss/BATCH_SIZE
            print(loss.item())

            '''optimizer'''
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pi.step(i)
            print()

        return running_loss

    for epoch in range(EPOCHS):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1),max_limit=len(dloader))

        running_loss= train_one_epoch()

        export_optm(optimizer, suction_quality_optimizer_path)
        export_optm(scope_optimizer, suction_scope_optimizer_path)

        pi.end()

        print('   Running loss = ',running_loss,', loss per iteration = ',running_loss/len(dloader))

    return model,scope_model

def training():
    '''dataloader'''
    dataset = suction_quality_dataset(data_pool=training_data)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    model=initialize_model(suction_quality_net,suction_quality_model_state_path)
    model.train(True)
    scope_model=initialize_model(suction_scope_net,suction_scope_model_state_path)
    scope_model.train(True)

    '''suction generator'''
    generator = initialize_model(suction_sampler_net, suction_sampler_model_state_path)
    generator.eval()

    '''optimizer'''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    optimizer = load_opt(optimizer, suction_quality_optimizer_path)
    scope_optimizer = torch.optim.Adam(scope_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=weight_decay)
    scope_optimizer = load_opt(scope_optimizer, suction_scope_optimizer_path)


    def train_one_epoch():
        running_loss = 0.

        for i, batch in enumerate(dloader, 0):
            depth,normal,score,pixel_index= batch
            depth=depth.cuda().float()
            normal=normal.cuda().float()
            score=score.cuda().float()

            b=depth.shape[0]

            '''generate normals'''
            with torch.no_grad():
                generated_normals=generator(depth)

            '''zero grad'''
            model.zero_grad()
            scope_model.zero_grad()

            '''insert the label pose'''
            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                generated_normals[j,:,pix_A,pix_B]=normal[j]

            '''get predictions'''
            suction_score = model(depth, generated_normals)
            scope_score = scope_model(generated_normals)
            predictions = suction_score * scope_score

            '''accumulate loss'''
            loss=0.
            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                prediction_=predictions[j,:,pix_A,pix_B]
                label_=score[j:j+1]
                loss+=custom_loss(prediction_,label_)


            '''Verification'''
            # view_suction_label(depth, normal, pixel_index, b)

            loss=loss/b
            print(loss.item())

            '''optimizer'''
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pi.step(i)
            print()

        return running_loss

    for epoch in range(EPOCHS):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1),max_limit=len(dloader))

        running_loss= train_one_epoch()

        export_optm(optimizer, suction_quality_optimizer_path)
        export_optm(scope_optimizer, suction_scope_optimizer_path)

        pi.end()

        print('   Running loss = ',running_loss,', loss per iteration = ',running_loss/len(dloader))

    return model,scope_model


def train_suction_sampler_kd(n_samples=None):
    # training_data.clear()
    while True:
        # try:
        if len(training_data) == 0:
            load_training_buffer_kd(size=n_samples)
        new_model,scope_model = knowledge_distillation()
        print(Fore.GREEN + 'Training round finished' + Fore.RESET)
        export_model_state(new_model, suction_quality_model_state_path)
        export_model_state(scope_model, suction_scope_model_state_path)
        training_data.clear()
        # except Exception as e:
        #     print(Fore.RED, str(e), Fore.RESET)

def train_suction_sampler(n_samples=None):
    # training_data.clear()
    while True:
        # try:
        if len(training_data) == 0:
            load_training_buffer(size=n_samples)
        new_model,scope_model = training()
        print(Fore.GREEN + 'Training round finished' + Fore.RESET)
        export_model_state(new_model, suction_quality_model_state_path)
        export_model_state(scope_model, suction_scope_model_state_path)
        training_data.clear()
        # except Exception as e:
        #     print(Fore.RED, str(e), Fore.RESET)

if __name__ == "__main__":
    train_suction_sampler(10)