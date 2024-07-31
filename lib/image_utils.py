import cv2
import torch
from PIL import Image
import numpy as np
from colorama import Fore


def check_image_similarity(old_image, new_image):
    res = np.sum((old_image.astype("float") - new_image.astype("float")) ** 2)
    res /= (old_image.shape[0] * old_image.shape[1])
    limit=30
    safety_threshold=50
    print(f'Image res = {res}')
    if res > limit+safety_threshold:
        return 1
    elif res<limit:
        return 0
    else:
        print(Fore.RED,'Warning: Can not confirm the grasp success',Fore.RESET)
        return None

def resize_image(im,size):
    if im.shape==size:return im
    if isinstance(im,torch.Tensor): im=im.numpy()
    image = Image.fromarray(im)
    # method 1, will keep the aspect ratio
    # image.thumbnail(size, Image.ANTIALIAS)
    # method 2 , will change the aspect ratio
    image=image.resize((size[1],size[0]),Image.ANTIALIAS)
    image = np.asarray(image)
    return image

def view_image(image,title=''):
    cv2.imshow(title, image)
    cv2.waitKey(0)