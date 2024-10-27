import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

def std_normalize(vector):
    std_=torch.std(vector,dim=-1,keepdim=True)
    mean_=torch.mean(vector,dim=-1,keepdim=True)
    normalized_vec=(vector-mean_)/std_
    return normalized_vec

def similarity_check(vector1,vector2):
    vector2=vector2[None,:,:]

    similarity = 1-F.cosine_similarity(vector1[:,:, None, :] , vector2[:,None, :, :], dim=-1)
    args = torch.argmin(similarity, dim=-1)

    return args

def estimate_umber_of_clusteres(data):
    model = DBSCAN(eps=2.5, min_samples=2)
    model.fit_predict(data)
    n=len(set(model.labels_))
    return n

def entropy(labels,bin_size=2):
    labels = np.round(labels, decimals=bin_size) * (10**bin_size)
    labels=(10**bin_size)+labels
    labels = labels.astype(int)
    ps = np.bincount(labels) / len(labels)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def moving_mean(old_mean,new_value,window_size=100):
    new_mean=( (old_mean*window_size)+new_value ) / (window_size+1)
    return new_mean

def moving_momentum(old_momentum,new_value,decay_rate,exponent=2):
    new_momentum=decay_rate*old_momentum+(1-decay_rate)*(new_value**exponent)
    return new_momentum
