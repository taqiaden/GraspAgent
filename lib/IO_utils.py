import io
import json
import os
import pickle
import re
import sys

import cv2
import numpy as np
import smbclient
import torch
import trimesh
from colorama import Fore
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

result=0

def space_key_pressed(): ## starts listener modulefg h
    # THIS FUNCTION DOES NOT WORK OVER SSH
    from pynput.keyboard import Listener, Key
    print('-------- Press space key to consider it as success grasp \\')
    global result
    result=0
    def print_key(*key):  ## prints key that is pressed
        global result
        # key is a tuple, so access the key(char) from key[1]
        print('\nYou Entered {0}'.format(key))
        if key[0] == Key.space:
            print('yes!')
            result=1
            return False
        else:
            result=0
            return False
    with Listener(on_press=print_key) as listener:
        listener.join()
    return result
def save_pickle(path,obj):
    # pickled_tuple=pickle.dumps(tuples)
    with open(path,'wb') as file:
        pickle.dump(obj,file)

def save_pickle_to_server2(path,obj):
    with smbclient.open_file(path,mode="wb") as f:
        pickle.dump(obj,f)

def save_pickle_to_server(path,tuples):
    pickled_tuple=pickle.dumps(tuples)
    with smbclient.open_file(path,mode="wb") as f:
        pickle.dump(pickled_tuple,f)
def load_pickle(path):
    with open(path,'rb') as file:
        tuple=pickle.load(file)
    return tuple

# def load_pickle_from_server(path):
#     with smbclient.open_file(path,'rb') as file:
#         tuple=pickle.load(file)
#     return tuple
def save_dict(dictionary,path):
    with open(path, 'w') as f:
        f.write(json.dumps(dictionary))

def load_dict(path):
    if os.path.exists(path):
        with open(path) as f:
            dictionary = json.loads(f.read())
        return dictionary
    else: return None
def save_trimesh_mesh_as_ply(mesh,path):
    result = trimesh.exchange.ply.export_ply(mesh, encoding='ascii')
    output_file = open(path, "wb+")
    output_file.write(result)
    output_file.close()


def cuda_space_info():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print('Cuda space info:')
    print(f'   total    : {info.total}')
    print(f'   free     : {info.free}')
    print(f'   used     : {info.used}')
    # model.eval()

def custom_print(*args,end='\n',probability=1,color=None):
    if np.random.rand() < probability:
        if color:print(color)
        for item in args:
            sys.stdout.write(str(item))
        sys.stdout.write(end)
        sys.stdout.flush()
        if color:print(Fore.RESET)

def copy_from_server_to_local(source,target):
    with smbclient.open_file(source,mode='rb') as f:
        buffer=io.BytesIO(f.read())
        with open(target,mode='wb') as w:
            w.write(buffer.read())

def move_single_labeled_sample(source_dataset, target_dataset,pc_filename=None,label_file_name=None):
    # first copy to target
    pc,label=source_dataset.load_labeled_data(pc_filename= pc_filename,label_filename=label_file_name)
    idx=source_dataset.get_index(pc_filename) if pc_filename is not None else source_dataset.get_index(label_file_name)
    target_dataset.save_labeled_data(pc,label,idx)

    # second delete from source
    source_dataset.remove_labeled_data(idx)


def remove_all_files(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            return False
    return True

def save_to_file(index,path):
    # save to txt
    with open(path, 'w') as f:
        f.write(str(index))

def load_file(path):
    # load from txt
    with open(path, 'r') as f:
        index = int(f.read())
    return index
def get_sample_index(path):
    file_name = os.path.splitext(path)[0]
    index = re.findall(r'\d+', file_name)[0]
    return index

def load_from_server(path):
    with smbclient.open_file(path, mode='rb') as f:
        file=f.read()
    return file

def load_pickle_from_server(path,allow_pickle=True):
    with smbclient.open_file(path, mode='rb') as f:
        file=f.read()
        file_like_object= io.BytesIO(file)
        # buffer = io.BytesIO(f.read())
        # pickle=f.read()
        npy=np.load(file_like_object,allow_pickle=allow_pickle)
    return npy

def load_pickle_from_server2(path):
    with smbclient.open_file(path, mode='rb') as f:
        pickle_obj=pickle.load(f)
    return pickle_obj

def save_pickle_to_server(path,data):
    pickled=pickle.dumps(data)
    with smbclient.open_file(path,mode="wb") as f:
        f.write(pickled)

def save_to_server(path,data,binary_mode=True):
    mode_="wb" if binary_mode else "w"
    with smbclient.open_file(path,mode=mode_) as f:
        f.write(data)

def save_image_to_server(path,data):
    data = data * 255
    data = data.astype(np.uint8)
    image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    _, buff = cv2.imencode('.jpg', image)
    byte_buffer = buff.tobytes()
    save_to_server(path, byte_buffer)

def save_data_to_server(data,path):
    # try:
        #method 1
        # pickled=pickle.dumps(data)
        # with smbclient.open_file(path,mode="wb") as f:
        #     f.write(pickled)
        # method 2
    buffer=io.BytesIO()
    torch.save(data,buffer)
    buffer.seek(0)
    with smbclient.open_file(path,mode='wb') as f:
        f.write(buffer.read())

    # except OSError:
    #     print('save data failed')

def save_data_to_local(data,path):
    # try:
        #method 1
        # pickled=pickle.dumps(data)
        # with smbclient.open_file(path,mode="wb") as f:
        #     f.write(pickled)
        # method 2
    buffer=io.BytesIO()
    torch.save(data,buffer)
    buffer.seek(0)
    with os.open_file(path,mode='wb') as f:
        f.write(buffer.read())

def npy_to_csv(filepath,npy):
    np.savetxt(filepath,npy,delimiter=",")

def get_balance_counter(dataset):
    balance_counter=np.array([0,0,0,0])
    assert dataset.length_of_label_container()==len(dataset),f'{dataset.length_of_label_container()},  {len(dataset)}'
    filenames=dataset.get_label_names()
    for filename in filenames:
        try:
            label = dataset.load_label(filename)
        except:
            continue
        balance_counter[0] += label[4] * label[3]
        balance_counter[1] += label[4] * (1 - label[3])
        balance_counter[2] += label[23] * label[3]
        balance_counter[3] += label[23] * (1 - label[3])
    return balance_counter

def unbalance_check(label,balance_counter,allowance=0):
    if label[4] == 1:
        # is grasp data
        if balance_counter[0] < balance_counter[1]-allowance and label[3] == 0:
            # more negative samples and the current draw is negative sample
            return 1
        elif balance_counter[1] <balance_counter[0]-allowance and label[3] == 1:
            # more positive samples and the current draw is negative positive
            return -1
        else:
            # No balance problem
            return 0
    elif label[23] == 1:
        # is suction data
        if balance_counter[2] < balance_counter[3]-allowance and label[3] == 0:
            return 1
        else:
            return 0
    else:
        assert False, 'Data is not recognized as either grasp or suction'

def update_balance_counter(balance_counter,is_grasp,score,n=1):
    # [3]: score
    # [4]: grasp
    # [23]: suction
    balance_counter[0] +=  score * n if is_grasp else 0
    balance_counter[1] +=  (1 - score) * n if is_grasp else 0
    balance_counter[2] +=  score * n if is_grasp==False else 0
    balance_counter[3] +=  (1 - score) * n if is_grasp==False else 0
    return balance_counter

