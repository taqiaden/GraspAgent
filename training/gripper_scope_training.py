from dataloaders.scope_dataloader import gripper_scope_dataset
from lib.loss.D_loss import l1_with_threshold_new
from lib.loss.regular_loss import mse_loss
from lib.report_utils import progress_indicator
from models.scope_net import scope_net_vanilla, gripper_scope_model_state_path, scope_net_attention
import torch
from torch import nn
from lib.IO_utils import   custom_print
from lib.models_utils import  initialize_model_state
from torch.utils.data.distributed import DistributedSampler


gripper_scope_optimizer_path=r'gripper_scope_optimizer'
print=custom_print
weight_decay = 0.000001
mes_loss=nn.MSELoss()
learning_rate=1*1e-5
batch_size=32
workers=2
epochs=1000

def train():
    model = initialize_model_state(scope_net_vanilla(in_size=16), gripper_scope_model_state_path)
    model.train(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    data_loader = torch.utils.data.DataLoader( gripper_scope_dataset(), batch_size=batch_size, num_workers=workers, shuffle=True)

    for epoch in range(epochs):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))
        running_loss=0.0
        for i, batch in enumerate(data_loader, 0):
            transformation,label=batch
            transformation=transformation.to('cuda').float()
            # transition=transition.to('cuda').float()
            label=label.to('cuda').float()

            model.zero_grad()

            predictions=model(transformation)

            loss=torch.clamp((1-predictions)*label-predictions*(label-1),0)
            loss=loss.mean()

            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            pi.step(i)
        print(f'Average loss = {running_loss/len(data_loader)}')

        pi.end()

if __name__ == "__main__":
    train()
