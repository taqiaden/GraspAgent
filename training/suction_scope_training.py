from check_points.check_point_conventions import ModelWrapper
from dataloaders.scope_dataloader import suction_scope_dataset
from lib.loss.D_loss import binary_l1
from lib.report_utils import progress_indicator
from models.scope_net import scope_net_vanilla, suction_scope_module_key
import torch
from torch import nn
from lib.IO_utils import   custom_print
from torch.utils.data.distributed import DistributedSampler
from records.training_satatistics import TrainingTracker


print=custom_print
weight_decay = 0.000001
mes_loss=nn.MSELoss()
learning_rate=5*1e-5
batch_size=4
workers=2
epochs=10000

bce_loss=nn.BCELoss()

def train():
    suction_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=suction_scope_module_key)
    suction_scope.ini_model(train=True)

    '''optimizers'''
    suction_scope.ini_sgd_optimizer(learning_rate=learning_rate)


    dataset=suction_scope_dataset()
    data_loader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, num_workers=workers, shuffle=True)


    for epoch in range(epochs):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))
        statistics=TrainingTracker(name=suction_scope_module_key,iterations_per_epoch=len(data_loader),samples_size=len(dataset))


        for i, batch in enumerate(data_loader, 0):
            input,label,file_ids=batch
            input=input.to('cuda').float()
            label=label.to('cuda').float()

            suction_scope.model.zero_grad()

            predictions=suction_scope.model(input)

            '''update confession matrix'''
            statistics.update_confession_matrix(label,predictions)

            loss=bce_loss(predictions,label)
            loss.backward()
            suction_scope.optimizer.step()

            statistics.running_loss += binary_l1(predictions, label).mean().item()
            pi.step(i)

        '''export models'''
        suction_scope.export_model()
        '''export optimizers'''
        suction_scope.export_optimizer()
        '''save dictionary'''
        statistics.save()
        statistics.print()

        pi.end()

if __name__ == "__main__":
    train()
