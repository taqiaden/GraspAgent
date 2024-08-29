import numpy as np
import torch

def random_one_hot(batch_size,number_of_classes):
    mask = torch.full((batch_size, number_of_classes), 0.).to('cuda')
    random_index = torch.randint(0, number_of_classes, (batch_size, 1)).to('cuda')
    mask = mask.scatter_(1, random_index, 1)
    return mask

def mask_features(noisy_one_hot,features):
    # noisy_one_hot [b,c], features [b,X,c] , b batch size, c classes, X a stack of features to be filtered
    one_hot,indexes=clean_one_hot(noisy_one_hot)
    one_hot = one_hot[:, None, :]
    masked_features=one_hot*features
    masked_features=masked_features[torch.arange(0,indexes.shape[0]),:,indexes] # remove zeros
    return masked_features

def indexes_to_one_hot(indexes,n_classes):
    # creating a 2D array filled with 0's

    one_hot_array = np.zeros((indexes.shape[0], n_classes), dtype=float)

    # replacing 0 with a 1 at the index of the original array
    one_hot_array[np.arange(indexes.shape[0]), indexes] = 1
    return one_hot_array

def one_hot(bin_label,bin_num):
    bin_onehot = torch.cuda.FloatTensor(bin_label.size(0), bin_num).zero_() if bin_label.is_cuda else torch.FloatTensor(bin_label.size(0), bin_num).zero_()
    bin_onehot.scatter_(1, bin_label.view(-1, 1).long(), 1)
    return  bin_onehot

def one_hot_to_indexes(one_hot):
    # input shape [b,c]
    indexes = torch.argmax(one_hot, dim=-1,keepdim=True).long().squeeze(-1)
    return indexes

def clean_one_hot(noisy_data):
    indexes=torch.argmax(noisy_data,dim=-1)
    one_hot=torch.zeros_like(noisy_data).scatter_(1,indexes.unsqueeze(1),1.)
    return one_hot

def one_hot_to_unit_regression(one_hot,number_of_bins):
    # input shape [b,c]
    bin = torch.argmax(one_hot, dim=-1,keepdim=True)
    bin_size = 1 / number_of_bins
    unit_regression=(bin.float()+0.5)*bin_size
    return unit_regression
def unit_regression_to_one_hot(unit_regression,number_of_bins):
    # input shape [b,1]
    bin_size=1/number_of_bins
    clamped_unit_regression = torch.clamp(unit_regression, 0, 1 - 1e-5)
    bin_index = (clamped_unit_regression / bin_size).floor().long()
    bin_onehot = torch.cuda.FloatTensor(bin_index.size(0), number_of_bins).zero_() if unit_regression.is_cuda else torch.FloatTensor(bin_index.size(0), number_of_bins).zero_()
    bin_onehot.scatter_(1, bin_index.view(-1, 1).long(), 1)
    return bin_onehot

def unit_regression_to_indexes(unit_regression,number_of_bins):
    # input shape [b,1]
    bin_size=1/number_of_bins
    clamped_unit_regression = torch.clamp(unit_regression, 0, 1 - 1e-5)
    bin_index = (clamped_unit_regression / bin_size).floor().long().squeeze(-1)
    return bin_index