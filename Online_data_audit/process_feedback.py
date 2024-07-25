import numpy as np
from colorama import Fore
from lib.dataset_utils import online_data
from lib.IO_utils import load_file, save_to_file

sampling_last_index_path='dataset/last_index.txt'
online_data=online_data()

def verify_label_is_in_standard_form(label):
    assert len(label)==28, f'label size is {len(label)}, the standard size is 27'
    list_of_types = [float, float, float, int, int, float, float,
                     float, float, float, float, float, float, float,
                     float, float, float
                    , float, float, float, float, np.float64,
                     np.float64, int, int, int, int,int]
    for i in range(len(label)):
        if i==3:
            # the socre can be saved in either float and int
            assert np.float32 == type(label[i]) or int == type(label[i]), f'label[{i}] type is {type(label[i])}, expected type is {np.float32} or {int}'
        else:
            assert list_of_types[i]==type(label[i]), f'label[{i}] type is {type(label[i])}, expected type is {list_of_types[i]}'

def standard_label_structure(width, distance, transformation, normal, center_point, grasp, suction, score,state):
    label = center_point.tolist() + [score]

    transformation = transformation.reshape(-1)

    if grasp:
        label = label + [grasp] + transformation.tolist() + [width, distance]
        label = label + [0] * 4

    if suction:
        label = label + [0] * 19
        label = label + [suction] + normal.tolist()

    label=label+[state]
    # verify_label_is_in_standard_form(label)

    return np.array(label)

def save_new_data_sample(full_pc,width, distance, transformation, normal, center_point, grasp, suction, score ,state,dataset=online_data):

    index = load_file(sampling_last_index_path)+1

    point_data=full_pc
    label = standard_label_structure(width, distance, transformation, normal, center_point, grasp, suction, score,state)
    dataset.save_labeled_data(point_data,np.asarray(label),str(index).zfill(6))

    if score==1:print(Fore.GREEN,'Save new data, award =', score, Fore.RESET)
    else: print(Fore.YELLOW,'Save new data, award =', score,Fore.RESET)

    save_to_file(index, sampling_last_index_path)

    # tabulated data:
    # [0:3]: center_point
    # [3]: score
    # [4]: grasp
    # [5:21]: rotation_matrix
    # [21]: width
    # [22]: distance
    # [23]: suction
    # [24:27]: pred_normal
