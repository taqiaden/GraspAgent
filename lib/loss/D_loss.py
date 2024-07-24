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