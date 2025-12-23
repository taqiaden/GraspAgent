import atexit
import io
import os
import re
import signal
import threading
import time

import smbclient
import torch
from colorama import Fore

from Configurations.config import check_points_directory, check_points_extension
from lib.IO_utils import save_data_to_server, safe_save_to_smb
from registration import camera

def reshape_for_layer_norm(tensor,camera=camera,reverse=False):
    if reverse==False:
        channels=tensor.shape[1]
        tensor=tensor.permute(0,2,3,1).reshape(-1,channels)
        return tensor
    else:
        batch_size=int(tensor.shape[0]/(camera.width*camera.height))
        channels=tensor.shape[-1]
        tensor=tensor.reshape(batch_size,camera.height,camera.width,channels).permute(0,3,1,2)
        return tensor

def print_layers(model):
    total = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            n_params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            total += n_params
            print(f"{name:40s} | {str(module):30s} | params = {n_params:,}")
    print("-" * 80)
    print(f"Total trainable parameters: {total:,}")

def view_parameters_value(model,iterations=None):
    print('Model parameters value:')
    i=0
    for name, param in model.named_parameters():
        if iterations is not None:
            if iterations==i:break
        # print(name)
        print(name,': ',(param.data).abs().max())
        i += 1

def same_models(model1,model2):
    state1=get_model_state(model1).__str__()
    state2=get_model_state(model2).__str__()
    return state1==state2

def set_device():
    # run on Cuda if available
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print('cuda is not available')
    return device

def is_same_state(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        return False
    for key in model_state_dict_1:
        if not torch.equal(model_state_dict_1[key], model_state_dict_2[key]):
            return False
    return True

def number_of_parameters(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_model_state(model):
    checkpoint_dict = {}
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        checkpoint_dict['state_dict'] = model.module.state_dict()
    else:
        checkpoint_dict['state_dict'] = model.state_dict()
    return checkpoint_dict

save_threads = []
def async_safe_save(data, path):
    t = threading.Thread(target=save_data_to_server, args=(data, path))
    t.start()
    save_threads.append(t)

@atexit.register
def wait_for_saves():
    print("Waiting for pending saves to finish...")
    for t in save_threads:
        t.join()
def export_model_state(model,path):
    checkpoint_dict = get_model_state(model)

    full_path=check_points_directory+path+check_points_extension

    if os.path.isdir(check_points_directory):
        print(f'check points exported to: {full_path}')
        torch.save(checkpoint_dict,full_path)
    else:
        async_safe_save(checkpoint_dict,full_path)


def get_model_time_stamp(path):
    full_path=check_points_directory+path+check_points_extension
    if os.path.isdir(check_points_directory):
        return os.path.getmtime(full_path)
    else:
        return smbclient.path.getmtime(full_path)


def delete_check_point(path):
    full_path = check_points_directory + path + check_points_extension
    # print('save check point to {}'.format(path))
    if os.path.isdir(check_points_directory):
        os.remove(full_path)
    else:
        smbclient.remove(full_path)

def model_exist_check(path):
    full_path = check_points_directory + path + check_points_extension
    if os.path.exists(full_path):
        return True
    else:
        return smbclient.path.exists(full_path)


def is_remote_path(path: str) -> bool:
    """
    Heuristically check if a path is remote (server) or local.
    """
    # SMB/UNC paths
    if path.startswith(r"\\") or path.startswith("//"):
        return True

    # NFS or SCP style: host:/path
    if re.match(r"^[\w\.-]+:/", path):  # e.g., "server:/data" or "user@host:/path"
        return True

    # URL-based remote paths
    if re.match(r"^(smb|nfs|ssh|ftp|http|https)://", path):
        return True

    return False

def load_dictionary(file_name,wait=False):
    full_path=check_points_directory+file_name
    c=0
    while True:
        try:
            if os.path.exists(full_path):
                # local path
                return torch.load(full_path, map_location='cpu')
            elif is_remote_path(full_path) and smbclient.path.exists(full_path):
                # on server
                with smbclient.open_file(full_path, mode='rb') as file:
                    buffer=io.BytesIO(file.read())
                    return torch.load(buffer, map_location='cpu')
            break
        except Exception as e:
            if  wait:
                if c == 0:
                    print(Fore.RED, f'Unable to access the checkpoint file, path: {full_path}', Fore.RESET)
                    print(str(e))
                c+=1
            else:
                print(Fore.RED, f'Error while loading the checkpoint file: {str(e)}', Fore.RESET)
                return

def load_optimizer_state(optimizer,optimizer_state_path):
    pretrained_dict = load_dictionary(optimizer_state_path)
    if pretrained_dict:
        optimizer.load_state_dict(pretrained_dict['state_dict'])
    else:
        print(Fore.RED, f'Warning: optimizer state dictionary is not found, path : {optimizer_state_path}', Fore.RESET)
    return optimizer

def initialize_model_state(model,model_state_path,wait=False):
    pretrained_dict=load_dictionary(model_state_path+check_points_extension,wait=wait)

    if pretrained_dict:
        model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    else:
        print(Fore.RED,f'Warning: model state dictionary is not found, path : {model_state_path}',Fore.RESET)
    return model

def initialize_model(model_obj,path,wait=False):
    try:
        # print(path)
        model_obj = initialize_model_state(model_obj, path,wait=wait)
    except Exception as e:
        print(Fore.RED, 'Load state dictionary exception,  ', str(e), Fore.RESET)
    return model_obj

def view_grad_val(model):
    for name,param in model.named_parameters():
        if param.grad is not None:
            print(param.grad)
        else:
            print(f'no gradient for {name}')

def activate_parameters_training(module_list,activate):
    for p in module_list.parameters():
        p.requires_grad = activate

def update_model_state(model,model_state_path):
    # update with the new model if available
    if os.path.exists(model_state_path) or smbclient.path.exists(model_state_path):
        return initialize_model_state(model,model_state_path)
    else:return model

def model_on_cuda_check(model):
    result=(next(model.parameters()).is_cuda)
    return result

def check_model_device(model):
    print(next(model.parameters()).device)
