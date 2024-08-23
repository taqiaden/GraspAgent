import torch
from torch import nn

smooth_l1_loss=nn.SmoothL1Loss(beta=1.0)
def custom_loss(prediction,label):

    one_label_mask=label==1
    positive_prediction_mask=prediction>1.

    mask=positive_prediction_mask & one_label_mask

    label[mask]=(prediction[mask]).detach().clone()

    zero_label_mask=~one_label_mask
    negative_prediction_mask=prediction<0.
    mask=negative_prediction_mask & zero_label_mask
    label[mask]=(prediction[mask]).detach().clone()

    loss=smooth_l1_loss(prediction,label)
    return loss

def l1_with_threshold(prediction,label,with_smooth=True):

    one_label_mask=label==1

    l1_P=torch.clamp(1-prediction[one_label_mask],0)
    l1_N=torch.clamp(prediction[~one_label_mask],0)

    l1=0
    if l1_P.shape[0]>0:
        if with_smooth:
            l1+=smooth_l1_loss(l1_P,torch.zeros_like(l1_P))
        else:
            l1+=l1_P

    if l1_N.shape[0]>0:
        if with_smooth:
            l1 += smooth_l1_loss(l1_N, torch.zeros_like(l1_N))
        else:
            l1 += l1_N

    return l1

def l1_with_threshold_new(prediction,label,with_smooth=True):
    loss=torch.clamp((1-prediction)*label-prediction*(label-1),0)
    if with_smooth:
        loss= smooth_l1_loss(loss, torch.zeros_like(loss))
    return loss


