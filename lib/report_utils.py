import datetime
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore

from lib.IO_utils import custom_print


base_directory=r'records/'

print=custom_print

def save_error_log(msg):
    with open('error_record.log',"a") as f:
    # f=File('error_record.log')
        now =datetime.datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        f.write(f'error date_time: {date_time}+\n')
        f.write(msg+'\n')

class progress_indicator():
    def __init__(self,msg,max_limit):
        self.progress_percentage = 00
        self.max_limit=max_limit
        self.ended=False
        print(Fore.LIGHTMAGENTA_EX,msg,' ', self.progress_percentage , '%',Fore.RESET,end='')
    def step(self,i):
        if not self.ended:
            if i>=self.max_limit:
                str_ = '100%'
                print('\b\b\b' + str_)
                self.ended=True
            else:
                progress_percentage = math.floor((i + 1) * 100 / self.max_limit)
                str_ = str(progress_percentage).zfill(2) + '%'
                if str_=='100%':
                    self.ended=True
                    print('\b\b\b' + str_)
                else:
                    print('\b\b\b' + str_, end='')
    def end(self):
        if not self.ended:
            str_ = '100%'
            print('\b\b\b\b' + str_)

class wait_indicator():
    def __init__(self,msg):
        print(msg+' \\', end='')
        self.i=0
        self.skip_counter=0
    def step(self,seconds=0.5,skip_times=0):
        if self.skip_counter==skip_times:
            self.skip_counter = 0
        else:
            self.skip_counter += 1
            return
        time.sleep(seconds)
        if self.i == 0:
            print('\b/', end='')
            self.i = 1
        else:
            print('\b' + '\\', end='')
            self.i = 0
    def end(self):
        print('')

def plot_list(plot_list):
    data = np.array(plot_list)
    plt.plot(plot_list)
    plt.show()

def distribution_summary(npy,data_name=None):
    max_score = np.max(npy)
    min_score = np.min(npy)
    average = np.average(npy)
    std = np.std(npy)
    print(data_name if data_name else 'Data',' distribution: ')
    print(f'     Max = {max_score} , Min = {min_score}')
    print(f'     Ave = {average} , Std = {std}')
    return max_score,min_score,average,std

def save_new_data_point(data,file_name):
    # if not os.path.exists(file_name):
    #     with open(file_name, 'w') as file:
    #         file.write()
    with open(base_directory+file_name,"a") as f:
        f.write(f'{data}\n')

class counter_progress():
    def __init__(self,msg,counter=None):
        print(msg,end='')
        if counter: print(counter,end='')

    def step(self,counter):
        if counter == 1:
            print(counter, end='')
        else:
            for i in range(len(str(counter - 1))):
                print('\b', end='')
            print(counter, end='')

    def end(self):
        print('')

if __name__ == '__main__':
    save_new_data_point(torch.Tensor([3, 4, 5, 6]), 'my_file.txt')
    save_new_data_point(torch.Tensor([3, 4, 5, 6]).numpy(), 'my_file.txt')

