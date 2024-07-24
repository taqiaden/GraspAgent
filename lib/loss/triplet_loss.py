import torch
import torch.nn.functional as F

def triplet_loss(anchor,positive,negative,margin=0.5):
    anchor=anchor.transpose(1,2)
    positive=positive.transpose(1,2)
    negative=negative.transpose(1,2)

    # print(anchor.shape)
    # print(positive)
    # print(negative)

    positive_distance=1-F.cosine_similarity(anchor,positive,dim=-1)
    negative_distance=1-F.cosine_similarity(anchor,negative,dim=-1)

    # zero_loss=torch.tensor(0.0, device=anchor.device)

    # print(f'positive distance={positive_distance}')
    # print(f'negative distance={negative_distance}')

    loss=torch.relu(positive_distance-negative_distance+margin).mean()
    return loss