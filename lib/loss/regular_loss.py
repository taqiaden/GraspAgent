from torch import nn

bce_loss=nn.BCELoss()
l1_loss=nn.L1Loss()
mse_loss=nn.MSELoss()
l1_loss_vec=nn.L1Loss(reduction='none')
mse_loss_vec=nn.MSELoss(reduction='none')