import asyncio
from concurrent.futures import ProcessPoolExecutor

import numpy as np

print('running async test')

def say_boo():
    i = 0
    while True:
        print('...boo {0}'.format(i))
        i += 1


def say_baa():
    i = 0
    while True:
        print('...baa {0}'.format(i))
        i += 1

if __name__ == "__main__":
    x=[1,2,3,4,5,6]
