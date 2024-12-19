import os.path
import pickle

from colorama import Fore


def save_dict( dictionary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dict( file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            loaded_dict = pickle.load(file)
            return loaded_dict
    else:
        print(Fore.RED,'Dictionary file is not found',Fore.RESET)
        return {}




