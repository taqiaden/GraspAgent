import torch
from colorama import Fore
from torch import nn
from Online_data_audit.data_tracker import sample_random_buffer, suction_grasp_tracker, sample_positive_buffer
from dataloaders.suction_sampler_dl import suction_sampler_dataset
from lib.IO_utils import custom_print
from lib.dataset_utils import  online_data
from lib.depth_map import depth_to_point_clouds
from lib.models_utils import initialize_model, export_model_state
from lib.optimizer import load_opt, export_optm
from lib.report_utils import progress_indicator
from models.suction_sampler import suction_sampler_net, suction_sampler_model_state_path
from registration import transform_to_camera_frame, camera
from analytical_suction_sampler import estimate_suction_direction
from visualiztion import view_npy_open3d

suction_sampler_optimizer_path = r'suction_sampler_optimizer'
training_buffer = online_data()

training_buffer.main_modality=training_buffer.depth
print=custom_print
BATCH_SIZE=1
learning_rate=5*1e-6
EPOCHS = 1
weight_decay = 0.000001
workers=2

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)
def train_(file_ids):
    '''dataloader'''
    dataset = suction_sampler_dataset(data_pool=training_buffer,file_ids=file_ids)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    model=initialize_model(suction_sampler_net,suction_sampler_model_state_path)
    model.train(True)

    '''optimizer'''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    optimizer = load_opt(optimizer, suction_sampler_optimizer_path)


    for epoch in range(EPOCHS):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1),max_limit=len(dloader))

        running_loss = 0.

        for i, batch in enumerate(dloader, 0):
            depth = batch
            depth = depth.cuda().float()

            '''get predictions'''
            model.zero_grad()
            predictions = model(depth.clone())
            predictions = predictions.permute(0, 2, 3, 1)  # inference time on local computer = 8.9e-05 s

            loss = 0
            for j in range(BATCH_SIZE):
                '''generate labels'''
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                normals = estimate_suction_direction(pc, view=False)  # inference time on local computer = 1.3 s
                labels = torch.from_numpy(normals).to('cuda')
                '''mask prediction'''
                masked_prediction = predictions[j][mask]
                loss += ((1 - cos(masked_prediction, labels.squeeze())) ** 2).mean()
                '''view output'''
                # view_npy_open3d(pc,normals=normals)
                # normals=masked_prediction.detach().cpu().numpy()
                # view_npy_open3d(pc,normals=normals)

            '''compute loss'''
            loss = loss / BATCH_SIZE
            print(loss.item())

            '''optimize'''
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pi.step(i)
            print()

        export_optm(optimizer, suction_sampler_optimizer_path)

        pi.end()

        print('   Running loss = ',running_loss,', loss per iteration = ',running_loss/len(dloader))

    return model

def train_suction_sampler(n_samples=None):
    while True:
        file_ids = sample_positive_buffer(size=n_samples, dict_name=suction_grasp_tracker)
        new_model = train_(file_ids)
        print(Fore.GREEN + 'Training round finished' + Fore.RESET)
        export_model_state(new_model, suction_sampler_model_state_path)

if __name__ == "__main__":
    train_suction_sampler()


