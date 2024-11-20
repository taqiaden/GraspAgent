import torch
import torch.nn.functional as F

def triplet_loss(anchor,positive,negative,margin=0.5):
    anchor=anchor.transpose(1,2)
    positive=positive.transpose(1,2)
    negative=negative.transpose(1,2)

    positive_distance=1-F.cosine_similarity(anchor,positive,dim=-1)
    negative_distance=1-F.cosine_similarity(anchor,negative,dim=-1)

    loss=torch.ReLU(positive_distance-negative_distance+margin).mean()
    return loss