from dataloaders.scope_dataloader import gripper_scope_dataset
from lib.loss.D_loss import  binary_l1, binary_smooth_l1
from lib.optimizer import load_opt, export_optm
from lib.report_utils import progress_indicator
from models.scope_net import scope_net_vanilla, gripper_scope_model_state_path
import torch
from torch import nn
from lib.IO_utils import   custom_print
from lib.models_utils import initialize_model_state, export_model_state
from torch.utils.data.distributed import DistributedSampler

from records.training_satatistics import TrainingTracker

gripper_scope_optimizer_path=r'gripper_scope_optimizer'
print=custom_print
weight_decay = 0.000001
mes_loss=nn.MSELoss()
learning_rate=5*1e-4
batch_size=4
workers=2
epochs=10000

sigmoid = nn.Sigmoid()
bce_loss=nn.BCELoss(reduction='none')

def train():
    model = initialize_model_state(scope_net_vanilla(in_size=6), gripper_scope_model_state_path)
    model.train(True)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
    #                              betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = load_opt(optimizer, gripper_scope_optimizer_path)

    dataset=gripper_scope_dataset()
    data_loader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    for epoch in range(epochs):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))
        statistics=TrainingTracker(name='',iterations_per_epoch=len(data_loader),samples_size=len(dataset))

        for i, batch in enumerate(data_loader, 0):
            transformation,label,target_index=batch
            transformation=transformation.to('cuda').float()
            label=label.to('cuda').float()

            model.zero_grad()

            predictions=model(transformation)

            # if np.random.rand()>0.95:
            #     print(f'l={label[0].item()},  p={predictions[0].item()}')

            '''update confession matrix'''
            statistics.update_confession_matrix(label,predictions)

            # positive_mask=label>0.5
            # positive_loss = (binary_l1(predictions[positive_mask], label[positive_mask]) ** 2) * 0.5
            # negative_loss = binary_smooth_l1(predictions[~positive_mask], label[~positive_mask])
            # loss=torch.cat([positive_loss,negative_loss])

            loss = binary_smooth_l1(predictions, label)

            statistics.labels_with_zero_loss+=(loss<=0.0).sum()
            loss=loss.mean()

            loss.backward()
            optimizer.step()

            statistics.running_loss+=binary_l1(predictions, label).mean().item()

            pi.step(i)
        statistics.print()

        '''export models'''
        export_model_state(model, gripper_scope_model_state_path)
        '''export optimizers'''
        export_optm(optimizer, gripper_scope_optimizer_path)

        pi.end()

if __name__ == "__main__":
    train()