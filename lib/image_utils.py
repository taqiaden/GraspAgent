import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from colorama import Fore


def depth_to_gray_scale(depth,view=False,convert_to_three_channels=True,colorize=False):
    processed_gray_image=np.copy(depth)
    non_zero_min=np.min(depth[np.nonzero(depth)])
    processed_gray_image[processed_gray_image==0.0]+=non_zero_min
    max_=np.max(processed_gray_image)
    range=max_-non_zero_min
    processed_gray_image-=non_zero_min
    processed_gray_image/=range
    if convert_to_three_channels:
        processed_gray_image=np.concatenate([processed_gray_image,processed_gray_image,processed_gray_image],axis=-1)
    if colorize:
        processed_gray_image[:,:,2]=1-(processed_gray_image[:,:,2]**np.random.uniform(0.1,3))
        processed_gray_image[:,:,1]=1-(processed_gray_image[:,:,1]**np.random.uniform(0.1,2))
        processed_gray_image[:,:,0]=1-(processed_gray_image[:,:,0]**np.random.uniform(0.1,1))

    if view:view_image(processed_gray_image)

    return  processed_gray_image
def check_image_similarity(old_image, new_image):
    res = np.sum((old_image.astype("float") - new_image.astype("float")) ** 2)
    res /= (old_image.shape[0] * old_image.shape[1])
    limit=30
    safety_threshold=20
    print(f'Image res = {res}')
    if res > limit+safety_threshold:
        return 1
    elif res<limit:
        return 0
    else:
        print(Fore.RED,'Warning: Can not confirm the action success',Fore.RESET)
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
    # cv2.imshow(title, image)
    # cv2.waitKey(0)
    plt.imshow(image)
    plt.show()

