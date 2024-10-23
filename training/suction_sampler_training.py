import torch
from colorama import Fore
from torch import nn
from dataloaders.suction_sampler_dl import suction_sampler_dataset, load_training_buffer
from lib.IO_utils import custom_print
from lib.dataset_utils import training_data
from lib.depth_map import depth_to_point_clouds
from lib.models_utils import initialize_model, export_model_state
from lib.optimizer import load_opt, export_optm
from lib.report_utils import progress_indicator
from models.suction_sampler import suction_sampler_net, suction_sampler_model_state_path
from registration import transform_to_camera_frame, camera
from analytical_suction_sampler import estimate_suction_direction

suction_sampler_optimizer_path = r'suction_sampler_optimizer'
training_data=training_data()
training_data.main_modality=training_data.depth
print=custom_print
BATCH_SIZE=2
learning_rate=5*1e-6
EPOCHS = 1
weight_decay = 0.000001
workers=2

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)
def train_():
    '''dataloader'''
    dataset = suction_sampler_dataset(data_pool=training_data)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    model=initialize_model(suction_sampler_net,suction_sampler_model_state_path)
    model.train(True)

    '''optimizer'''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    optimizer = load_opt(optimizer, suction_sampler_optimizer_path)

    def train_one_epoch():
        running_loss = 0.

        for i, batch in enumerate(dloader, 0):
            depth= batch
            depth=depth.cuda().float()


            '''get predictions'''
            model.zero_grad()
            predictions=model(depth.clone())
            predictions = predictions.permute(0,2, 3, 1) # inference time on local computer = 8.9e-05 s

            loss=0
            for j in range(BATCH_SIZE):
                '''generate labels'''
                pc, mask = depth_to_point_clouds(depth[j,0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                normals = estimate_suction_direction(pc, view=False) # inference time on local computer = 1.3 s
                labels=torch.from_numpy(normals).to('cuda')
                '''mask prediction'''
                masked_prediction = predictions[j][mask]
                loss += ((1 - cos(masked_prediction, labels.squeeze()))**2).mean()
                '''view output'''
                # view_npy_open3d(pc,normals=normals)
                # normals=masked_prediction.detach().cpu().numpy()
                # view_npy_open3d(pc,normals=normals)

            '''compute loss'''
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

        export_optm(optimizer, suction_sampler_optimizer_path)

        pi.end()

        print('   Running loss = ',running_loss,', loss per iteration = ',running_loss/len(dloader))

    return model

def train_suction_sampler(n_samples=None):
    training_data.clear()
    while True:
        if len(training_data) == 0:
            load_training_buffer(size=n_samples)
        new_model = train_()
        print(Fore.GREEN + 'Training round finished' + Fore.RESET)
        export_model_state(new_model, suction_sampler_model_state_path)
        training_data.clear()

if __name__ == "__main__":
    train_suction_sampler(100)


