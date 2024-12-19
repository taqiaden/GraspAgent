import math
from datetime import datetime

from colorama import Fore

from lib.math_utils import seeds
from training.Grasp_GAN_training import train_Grasp_GAN
from training.gripper_quality_training import train_gripper_quality
from training.suction_quality_training import train_suction_quality

time_seed=math.floor(datetime.now().timestamp())

if __name__ == "__main__":
    while True:
        seeds(time_seed)

        for i in range(0):
            try:
                print(Fore.LIGHTMAGENTA_EX,'Train gripper sampler', Fore.RESET)
                train_Grasp_GAN(600, BATCH_SIZE=2, epochs=1, maximum_gpus=None)
            except Exception as e:
                print(Fore.RED, str(e), Fore.RESET)
        for i in range(1):
            try:
                print(Fore.LIGHTMAGENTA_EX,'Train suction quality net', Fore.RESET)
                train_suction_quality(3000, BATCH_SIZE=4, epochs=3, maximum_gpus=None,learning_rate=5*1e-5)
            except Exception as e:
                print(Fore.RED, str(e), Fore.RESET)
        for i in range(1):
            try:
                print(Fore.LIGHTMAGENTA_EX,'Train gripper quality net', Fore.RESET)
                train_gripper_quality(3000, BATCH_SIZE=4, epochs=3, maximum_gpus=None,learning_rate=5*1e-5)
            except Exception as e:
                print(Fore.RED, str(e), Fore.RESET)

