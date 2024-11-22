import os
import datetime
import torch
from colorama import Fore
from torch import nn
from Configurations.config import  untested_model_stamp
from Online_data_audit.data_tracker import sample_random_buffer, suction_grasp_tracker
from dataloaders.suction_quality_dl import  suction_quality_dataset
from lib.IO_utils import   custom_print
from lib.dataset_utils import online_data
from lib.loss.D_loss import binary_l1, binary_smooth_l1, smooth_l1_loss
from lib.models_utils import initialize_model, export_model_state, delete_check_point, model_exist_check
from lib.optimizer import load_opt, export_optm
from lib.report_utils import progress_indicator
from models.suction_quality import suction_quality_net, suction_quality_model_state_path
from models.suction_sampler import suction_sampler_net, suction_sampler_model_state_path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from filelock import FileLock
from records.training_satatistics import TrainingTracker, MovingMetrics

# from torch.utils.tensorboard import SummaryWriter
# writer=SummaryWriter('runs/quality_training/suction')

module_key='sq'
counter=0
lock = FileLock("file.lock")

suction_quality_optimizer_path=r'suction_quality_optimizer'

training_buffer = online_data()

training_buffer.main_modality=training_buffer.depth

l1_loss=nn.L1Loss()

print=custom_print
weight_decay = 0.000001
max_lr=0.01
min_lr=5*1e-4

activate_full_power_at_midnight=False

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
    model=initialize_model(suction_quality_net,suction_quality_model_state_path)
    model.train(True)
    return model

def accumulate_loss(batch_size,pixel_index,predictions,score,statistics,moving_rates):
    loss = 0.
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        prediction_ = predictions[j, :, pix_A, pix_B]
        label_ = score[j:j + 1]
        # print(Fore.YELLOW, f'prediction = {prediction_.item()}, label = {label_.item()}', Fore.RESET)
        # loss += binary_smooth_l1(prediction_, label_)
        decay_loss=binary_l1(predictions[j],torch.zeros_like(predictions[j])).mean()
        lambda1=max(moving_rates.tnr-moving_rates.tpr,0)

        # print(f'lambda  = {lambda1}')
        main_loss = binary_l1(prediction_, label_).mean()**2
        if label_>0.5:main_loss*=(1+lambda1)
        # main_loss=binary_l1(prediction_, label_).mean()*(1+lambda1) if label_>0.5 else binary_l1(prediction_, label_).mean()**2

        # global counter
        # counter+=1
        # writer.add_scalar('loss',positive_loss.item(),counter)
        loss+=decay_loss+main_loss
        # if label_ > 5.0:
        #     loss += binary_smooth_l1(prediction_, label_)
        # else:
        #     loss += (binary_l1(prediction_, label_) ** 2) * 0.5
        if loss==0:statistics.labels_with_zero_loss+=1

        statistics.update_confession_matrix(label_,prediction_)
        moving_rates.update(label_, prediction_)
        # moving_rates.view()

    return loss

class TrainerDDP:
    def __init__(self,gpu_id: int,model: nn.Module,generator,training_congiurations,file_ids):
        '''set devices and model wrappers'''
        self.world_size=training_congiurations.world_size
        self.batch_size=training_congiurations.batch_size
        self.epochs=training_congiurations.epochs
        self.workers=training_congiurations.workers
        self.learning_rate=training_congiurations.learning_rate
        self.file_ids=file_ids
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()

        self.device=f'cuda:{gpu_id}'

        self.model=model
        self.model.to(self.device)

        self.model=DDP(model,device_ids=[gpu_id],find_unused_parameters=True)

        self.gpu_id=gpu_id

        '''dataloader'''
        self.dataset = suction_quality_dataset(data_pool=training_buffer,file_ids=file_ids)
        self.data_laoder=self._prepare_dataloader(self.dataset)

        '''optimizers'''
        self.optimizer=self._prepare_optimizer()

        '''suction generator'''
        self.generator = generator
        self.generator.to(self.device)

        '''metrics'''
        self.moving_rates = MovingMetrics(module_key)

    def _prepare_dataloader(self,dataset):
        dloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=False,
                                              sampler=DistributedSampler(dataset))
        return dloader

    def export_check_points(self):
        if self.gpu_id==0 :
            '''export models'''
            export_model_state(self.model, suction_quality_model_state_path+untested_model_stamp)
            '''export optimizers'''
            export_optm(self.optimizer, suction_quality_optimizer_path)

            print(Fore.CYAN, 'Check points exported successfully',Fore.RESET)

    def _prepare_optimizer(self):
        '''optimizers'''
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
        #                              betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,  weight_decay=weight_decay)
        with lock:
            optimizer = load_opt(optimizer, suction_quality_optimizer_path)

        return optimizer

    def train(self):

        for epoch in range(self.epochs):
            pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(self.data_laoder))
            statistics = TrainingTracker(name='', iterations_per_epoch=len(self.data_laoder), samples_size=len(self.dataset))
            for i, batch in enumerate(self.data_laoder, 0):
                depth, normal, score, pixel_index = batch
                depth = depth.to(self.device).float()
                normal = normal.to(self.device).float()
                score = score.to(self.device).float()

                b = depth.shape[0]

                '''generate normals'''
                with torch.no_grad():
                    generated_normals = self.generator(depth.clone())

                '''zero grad'''
                self.model.zero_grad()
                self.optimizer.zero_grad()

                '''insert the label pose'''
                for j in range(b):
                    pix_A = pixel_index[j, 0]
                    pix_B = pixel_index[j, 1]
                    generated_normals[j, :, pix_A, pix_B] = normal[j]

                '''get predictions'''
                predictions = self.model(depth.clone(), generated_normals.clone())

                '''compute loss'''
                loss = accumulate_loss(b, pixel_index, predictions, score,statistics,self.moving_rates)


                '''Verification'''
                # view_suction_label(depth, normal, pixel_index, b)

                loss = loss / b
                # print(loss.item())

                '''optimizer'''
                loss.backward()
                self.optimizer.step()

                statistics.running_loss += loss.item()
                pi.step(i)
                # print()

            # statistics.print()
            pi.end()

            '''export models and optimizers'''
            self.export_check_points()

            '''save metrics'''
            self.moving_rates.save()

def main_ddp(rank: int, t_config,model,generator,file_ids):
    '''setup parallel training'''
    ddp_setup(rank,t_config.world_size)

    '''training'''
    trainer=TrainerDDP(gpu_id=rank,model=model,generator=generator,training_congiurations=t_config,file_ids=file_ids)
    trainer.train()

    '''destroy process'''
    destroy_process_group()

def train_suction_quality(n_samples=None,BATCH_SIZE=4,epochs=1,maximum_gpus=None,learning_rate=5*1e-3):

    '''load check points'''
    model = prepare_models()

    '''configure world_size'''
    now = datetime.datetime.now()
    if activate_full_power_at_midnight and 7>now.hour>2:
        world_size = torch.cuda.device_count()
    else:
        world_size = torch.cuda.device_count() if maximum_gpus is None else maximum_gpus

    '''train configurations'''
    # print(Fore.YELLOW, f'Batch size per node = {BATCH_SIZE} ', Fore.RESET)
    t_config=train_config(learning_rate,BATCH_SIZE,epochs,0,world_size)

    '''prepare buffer'''
    file_ids = sample_random_buffer(size=n_samples, dict_name=suction_grasp_tracker,load_test_samples=0)
    buffer_size=len(file_ids)
    print(Fore.YELLOW,f'Buffer size = {buffer_size} ', Fore.RESET)

    '''load sampler'''
    generator = initialize_model(suction_sampler_net, suction_sampler_model_state_path)
    generator.eval()

    '''Begin multi processing'''
    mp.spawn(main_ddp,args=(t_config,model,generator,file_ids),nprocs=world_size)

def testing_loss(model,depth,generated_normals,batch_size,pixel_index,score,statistics):
    predictions = model(depth.clone(), generated_normals.clone())
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        prediction_ = predictions[j, :, pix_A, pix_B]
        label_ = score[j:j + 1]
        statistics.update_cumulative_discrimination_loss(prediction_,label_)
        statistics.update_confession_matrix(label_,prediction_)

def model_test():
    print('--- Begin evaluation ---')
    '''load models'''
    regular_model = initialize_model(suction_quality_net, suction_quality_model_state_path)
    regular_model.train(False)
    untested_model=initialize_model(suction_quality_net, suction_quality_model_state_path+untested_model_stamp)
    untested_model.train(False)
    generator = initialize_model(suction_sampler_net, suction_sampler_model_state_path)
    generator.eval()

    '''call test samples'''
    file_ids = sample_random_buffer(size=None, dict_name=suction_grasp_tracker, load_test_samples=0)

    '''data loader'''
    dataset = suction_quality_dataset(data_pool=training_buffer, file_ids=file_ids)
    data_laoder = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True  )

    '''track statistics'''
    regular_model_statistics = TrainingTracker(name='', iterations_per_epoch=len(data_laoder), samples_size=len(dataset))
    untested_model_statistics = TrainingTracker(name='', iterations_per_epoch=len(data_laoder), samples_size=len(dataset))

    '''evaluate'''
    pi = progress_indicator('Evaluation progress', max_limit=len(data_laoder))
    for i, batch in enumerate(data_laoder, 0):
        depth, normal, score, pixel_index = batch
        depth = depth.cuda().float()
        normal = normal.cuda().float()
        score = score.cuda().float()

        b = depth.shape[0]

        '''generate normals'''
        with torch.no_grad():
            generated_normals = generator(depth.clone())

        '''insert the label pose'''
        for j in range(b):
            pix_A = pixel_index[j, 0]
            pix_B = pixel_index[j, 1]
            generated_normals[j, :, pix_A, pix_B] = normal[j]

        '''update evaluation statistics'''
        testing_loss(regular_model,depth,generated_normals,b,pixel_index,score,regular_model_statistics)
        testing_loss(untested_model, depth, generated_normals, b, pixel_index, score, untested_model_statistics)
        
        pi.step(i)
    pi.end()

    '''Decision'''
    accept_upgrade=((untested_model_statistics.confession_matrix.precision()>regular_model_statistics.confession_matrix.precision())
                    or untested_model_statistics.confession_matrix.fpr()<regular_model_statistics.confession_matrix.fpr())

    print('Regular model test result: ')
    regular_model_statistics.print()
    print('New model test result: ')
    untested_model_statistics.print()
    '''upgrade'''
    if accept_upgrade:
        export_model_state(untested_model, suction_quality_model_state_path)
        print(Fore.GREEN,'Model upgrade accepted')

        print(Fore.RESET)
    else:
        print(Fore.RED,'Model update declined;')

        print(Fore.RESET)

    '''delete untested model'''
    delete_check_point(suction_quality_model_state_path+untested_model_stamp)

if __name__ == "__main__":
    for i in range(100):
        if model_exist_check(suction_quality_model_state_path+untested_model_stamp):
            with torch.no_grad():
                model_test()
        else:
            train_suction_quality(None,BATCH_SIZE=1,epochs=1,maximum_gpus=1)