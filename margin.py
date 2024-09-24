from tkinter import Tk, Canvas, mainloop, Label

import torch
from PIL import ImageTk, Image
from torch import nn

from lib.loss.D_loss import l1_with_threshold_new

smooth_l1_loss=nn.SmoothL1Loss(beta=.50)
mse=nn.MSELoss()
l1=torch.tensor([1])
l2=torch.tensor([0.0])


print(smooth_l1_loss(l1,l2))
print(mse(l1,l2))
print(l1_with_threshold_new(l1, l2,with_smooth=True))
print(l1_with_threshold_new(l1, l2,with_smooth=False)**2)
