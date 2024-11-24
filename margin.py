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
    x=np.array([1,2,30,4,5,6])
    y=np.array([1,2,3,4,5,606])
    z=np.array([10,12,3,4,5,606])

    x[-1]=y[-1]
    z=x+y
    z[0]=9
    print(z)
    print(x)
    print(y)
