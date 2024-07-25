import math
from datetime import datetime
from colorama import Fore
from GAGAN import train_generator
from records.records_managment import clear_records
from suction_D_training import train_suction
from    gripper_D_training import train_gripper
from lib.math_utils import seeds
from training.opening_GAN_training import train_opening_GAN

time_seed=math.floor(datetime.now().timestamp())

if __name__ == "__main__":
    while True:
        seeds(time_seed)
        clear_records()
        # train_opening_GAN(n_samples=3)
        # train_generator(n_samples=2)
        # train_gripper(n_samples=2)
        # train_suction(n_samples=2)
        # exit()
        for i in range(1000):
            try:
                train_opening_GAN(n_samples=1000)
            except Exception as e:
                print(Fore.RED, str(e), Fore.RESET)
        for i in range(1):
            try:
                train_generator(n_samples=1000)
            except Exception as e:
                print(Fore.RED, str(e), Fore.RESET)
        for i in range(1):
            try:
                train_gripper(n_samples=1000)
            except Exception as e:
                print(Fore.RED, str(e), Fore.RESET)
        for i in range(1):
            try:
                train_suction(n_samples=1000)
            except Exception as e:
                print(Fore.RED, str(e), Fore.RESET)
