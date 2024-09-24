import os
import datetime
import torch
from colorama import Fore
from torch import nn
from Verfication_tests.gripper_verf import view_gripper_label
from dataloaders.gripper_quality_dl import  load_training_buffer, gripper_quality_dataset
from lib.IO_utils import   custom_print
from lib.dataset_utils import  training_data
from lib.loss.D_loss import l1_with_threshold_new
from lib.loss.regular_loss import mse_loss
from lib.models_utils import initialize_model, export_model_state
from lib.optimizer import load_opt, export_optm
from lib.report_utils import progress_indicator
from models.Grasp_GAN import gripper_sampler_path, gripper_sampler_net
from models.gripper_quality import gripper_quality_model_state_path, gripper_quality_net
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from filelock import FileLock

from training.suction_quality_training import train_suction_quality

lock = FileLock("file.lock")

gripper_quality_optimizer_path=r'gripper_quality_optimizer'


training_data=training_data()
training_data.main_modality=training_data.depth
print=custom_print
weight_decay = 0.000001

max_lr=0.01
min_lr=1*1e-6



activate_full_power_at_midnight=True

l1_loss=nn.L1Loss(reduction='none')
mes_loss=nn.MSELoss()

def ddp_setup(rank,world_size):
    os.environ["MASTER_ADDR"]="localhost"
    os.environ["MASTER_PORT"]="12355"
    init_process_group(backend="nccl",rank=rank,world_size=world_size)

class train_config:
    def __init__(self,learning_rate,batch_size,epochs,workers,world_size):
        self.batch_size=batch_size
        self.epochs=epochs
        self.workers=workers
        self.world_size=world_size
        self.learning_rate=learning_rate

def prepare_models():

    print(Fore.CYAN,'Import check points',Fore.RESET)

    '''models'''
    model=initialize_model(gripper_quality_net,gripper_quality_model_state_path)
    model.train(True)


    return model

class TrainerDDP:
    def __init__(self,gpu_id: int,model: nn.Module,generator,training_congiurations):
        '''set devices and model wrappers'''
        self.world_size=training_congiurations.world_size
        self.batch_size=training_congiurations.batch_size
        self.epochs=training_congiurations.epochs
        self.workers=training_congiurations.workers
        self.learning_rate=training_congiurations.learning_rate
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()

        self.device=f'cuda:{gpu_id}'

        self.model=model
        self.model.to(self.device)

        self.model=DDP(model,device_ids=[gpu_id],find_unused_parameters=True)

        self.gpu_id=gpu_id

        '''dataloader'''
        dataset = gripper_quality_dataset(data_pool=training_data)
        self.data_laoder=self._prepare_dataloader(dataset)

        '''optimizers'''
        self.optimizer=self._prepare_optimizer()

        '''gripper generator'''
        self.generator  = generator
        self.generator.eval()
        self.generator.to(self.device)

    def _prepare_dataloader(self,dataset):
        dloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=False,
                                              sampler=DistributedSampler(dataset))
        return dloader

    def export_check_points(self):
        if self.gpu_id==0:
            '''export models'''
            export_model_state(self.model, gripper_quality_model_state_path)
            '''export optimizers'''
            export_optm(self.optimizer, gripper_quality_optimizer_path)

            print(Fore.CYAN, 'Check points exported successfully',Fore.RESET)

    def _prepare_optimizer(self):
        '''optimizers'''
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,  weight_decay=weight_decay)
        with lock:
            optimizer = load_opt(optimizer, gripper_quality_optimizer_path)
        return optimizer

    def train(self):
        def train_one_epoch():
            running_loss = 0.

            for i, batch in enumerate(self.data_laoder, 0):
                depth, pose_7, score, pixel_index = batch
                depth = depth.to(self.device).float()
                pose_7 = pose_7.to(self.device).float()  # [b,1,7]
                score = score.to(self.device).float()  # [b]
                b = depth.shape[0]

                '''generate pose'''
                with torch.no_grad():
                    generated_grasps = self.generator(depth.clone())

                '''zero grad'''
                self.model.zero_grad()
                self.optimizer.zero_grad()

                '''insert the label pose'''
                for j in range(b):
                    pix_A = pixel_index[j, 0]
                    pix_B = pixel_index[j, 1]
                    generated_grasps[j, :, pix_A, pix_B] = pose_7[j]

                '''get predictions'''
                predictions = self.model(depth.clone(), generated_grasps.clone())

                '''accumulate loss'''
                loss = 0.
                for j in range(b):
                    pix_A = pixel_index[j, 0]
                    pix_B = pixel_index[j, 1]
                    prediction_ = predictions[j, :, pix_A, pix_B]
                    label_ = score[j:j + 1]
                    print(Fore.YELLOW,f'prediction = {prediction_.item()}, label = {label_.item()}',Fore.RESET)
                    # loss += l1_with_threshold_new(prediction_, label_,with_smooth=False)#**2.0
                    if label_>5.0:
                        loss+=l1_loss(prediction_, label_)
                    else:
                        l=l1_loss(prediction_, label_)
                        if l>1.: l=l**2
                        loss += l
                loss = loss / b
                print(loss.item())

                '''Verification'''
                #view_gripper_label(depth, pose_7, pixel_index, b)

                '''optimizer step'''
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pi.step(i)
                print()

            return running_loss

        for epoch in range(self.epochs):
            pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(self.data_laoder))

            running_loss= train_one_epoch()
            print(Fore.BLUE,f' Running loss = {running_loss}',Fore.RESET)

            pi.end()

            '''export models and optimizers'''
            self.export_check_points()



def main_ddp(rank: int, t_config,model,generator):
    '''setup parallel training'''
    ddp_setup(rank,t_config.world_size)

    '''training'''
    trainer=TrainerDDP(gpu_id=rank,model=model,generator=generator,training_congiurations=t_config)
    trainer.train()

    '''destroy process'''
    destroy_process_group()

def train_gripper_quality(n_samples=None,BATCH_SIZE = 1,epochs=1,maximum_gpus=None,learning_rate=5*1e-3):
    training_data.clear()


    '''load check points'''
    model = prepare_models()

    '''configure world_size'''
    now = datetime.datetime.now()
    if activate_full_power_at_midnight and 7 > now.hour > 2:
        world_size = torch.cuda.device_count()
    else:
        world_size = torch.cuda.device_count() if maximum_gpus is None else maximum_gpus

    '''train configurations'''
    print(Fore.YELLOW, f'Batch size per node = {BATCH_SIZE} ', Fore.RESET)
    t_config = train_config(learning_rate,BATCH_SIZE, epochs, 0, world_size)

    '''prepare buffer'''
    if len(training_data) == 0:
        load_training_buffer(size=n_samples)
    buffer_size = len(training_data)
    print(Fore.YELLOW, f'Buffer size = {buffer_size} ', Fore.RESET)

    '''load sampler'''
    generator = initialize_model(gripper_sampler_net, gripper_sampler_path)

    '''Begin multi processing'''
    mp.spawn(main_ddp, args=(t_config, model,generator), nprocs=world_size)

    '''clear buffer'''
    training_data.clear()

if __name__ == "__main__":
    while True:
        train_gripper_quality(12,BATCH_SIZE = 1,epochs=1,maximum_gpus=None)
        # train_suction_quality(600)