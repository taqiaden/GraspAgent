import torch
from colorama import Fore

from Configurations.config import check_points_extension
from lib.models_utils import load_optimizer_state, export_model_state


def load_opt(optimizer,path):
    try:
        if isinstance(optimizer,torch.optim.Adam):
            optimizer=load_optimizer_state(optimizer, path+'_ADAM'+check_points_extension)
        elif isinstance(optimizer,torch.optim.SGD):
            optimizer=load_optimizer_state(optimizer, path+'_SGD'+check_points_extension)
        else:
            print(Fore.RED,'Unknown optimizer type',Fore.RESET)
    except Exception as e:
        print(Fore.RED,str(e),Fore.RESET)

    return optimizer

def export_optm(optimizer,path):
    if isinstance(optimizer, torch.optim.Adam):
        export_model_state(optimizer, path+'_ADAM')
    elif isinstance(optimizer, torch.optim.SGD):
        export_model_state(optimizer, path+'_SGD')
    else:
        print(Fore.RED, 'Unknown optimizer type', Fore.RESET)

def decay_lr(min_lr,max_lr,model,exponent=1):
    parameters = []
    if min_lr==0 and max_lr==0:
        parameters += [
            {'params':  model.parameters() , 'lr': min_lr}]
        return parameters
    n_layers = 0
    assert max_lr>=min_lr
    decay=not max_lr==min_lr
    for _ in model.named_parameters():
        n_layers += 1
    for idx, (name, param) in enumerate(model.named_parameters()):
        lr_ = min_lr + ((max_lr - min_lr) / (n_layers ** exponent)) * (idx ** exponent) if decay else max_lr
        parameters += [
            {'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad], 'lr': lr_}]
    return parameters
def decayed_optimizer(model_,lr_list=None,lr_=None,decay_rate=0.5,use_RMSprop=False,use_sgd=False,weight_decay=None):
    weight_decay_=weight_decay if weight_decay is not None else 0.
    parameters = []
    lr = lr_
    lr_i=decay_rate*lr
    if isinstance(model_,list):
        for i,mode_item in enumerate(model_):
            if lr_list is not None:
                lr= lr_list[i]

            parameters += decay_lr(min_lr=decay_rate*lr , max_lr=lr, model=mode_item)
    else:
        parameters += decay_lr(min_lr=lr_i, max_lr=lr, model=model_)
    # optimizer = torch.optim.SGD(parameters, lr=lr,  weight_decay=WEIGHT_DECAY)
    if use_RMSprop: optimizer = torch.optim.RMSprop(parameters, lr=lr,weight_decay=weight_decay_)
    elif use_sgd: optimizer = torch.optim.SGD(parameters, lr=lr,  weight_decay=weight_decay_)
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay_)
    return optimizer