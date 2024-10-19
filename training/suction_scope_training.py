from dataloaders.scope_dataloader import suction_scope_dataset
from lib.loss.D_loss import binary_l1, binary_smooth_l1
from lib.optimizer import export_optm, load_opt
from lib.report_utils import progress_indicator
from models.scope_net import scope_net_vanilla, suction_scope_model_state_path
import torch
from torch import nn
from lib.IO_utils import   custom_print
from lib.models_utils import initialize_model_state, export_model_state
from torch.utils.data.distributed import DistributedSampler

suction_scope_optimizer_path=r'suction_scope_optimizer'
print=custom_print
weight_decay = 0.000001
mes_loss=nn.MSELoss()
learning_rate=1*1e-5
batch_size=64
workers=2
epochs=10000

def train():
    model = initialize_model_state(scope_net_vanilla(in_size=6), suction_scope_model_state_path)
    model.train(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    optimizer = load_opt(optimizer, suction_scope_optimizer_path)

    dataset=suction_scope_dataset()
    data_loader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    for epoch in range(epochs):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))
        running_loss=0.0
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        labels_with_zero_loss_counter=0

        for i, batch in enumerate(data_loader, 0):
            input,label=batch
            input=input.to('cuda').float()
            label=label.to('cuda').float()

            model.zero_grad()

            predictions=model(input)

            '''update confession matrix'''
            TP+=((label > 0.5) & (predictions>0.5)).sum()
            FP+=((label < 0.5) &( predictions>0.5)).sum()
            FN+=((label > 0.5) & (predictions<=0.5)).sum()
            TN+=((label < 0.5 )&( predictions<=0.5)).sum()

            # if np.random.rand()>0.9:
            #     print(f'l={label[0].item()},  p={predictions[0].item()}')

            positive_mask = label > 0.5
            positive_loss = (binary_l1(predictions[positive_mask], label[positive_mask]) ** 2) * 0.5
            negative_loss = binary_smooth_l1(predictions[~positive_mask], label[~positive_mask])
            loss = torch.cat([positive_loss, negative_loss])
            labels_with_zero_loss_counter+=loss.shape[0]-(loss==0).sum()
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            running_loss+=binary_l1(predictions[positive_mask], label[positive_mask]).mean().item()
            pi.step(i)

        print(f'Average loss = {running_loss/len(data_loader)}')
        print(f'TP={TP}, FP={FP}')
        print(f'FN={FN}, TN={TN}')
        print(f'Number of labels with zero loss = {labels_with_zero_loss_counter}, total data size= {len(dataset)}')


        '''export models'''
        export_model_state(model, suction_scope_model_state_path)
        '''export optimizers'''
        export_optm(optimizer, suction_scope_optimizer_path)

        pi.end()

if __name__ == "__main__":
    train()
