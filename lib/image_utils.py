import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from colorama import Fore


def imflatfield(I, sigma):
    """Python equivalent imflatfield implementation
       I format must be BGR uint8"""
    A = I.astype(np.float32) / 255  # A = im2single(I);
    Ihsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)  # Ihsv = rgb2hsv(A);
    A = Ihsv[:, :, 2]  # A = Ihsv(:,:,3);

    filterSize = int(2 * np.ceil(2 * sigma) + 1);  # filterSize = 2*ceil(2*sigma)+1;

    # shading = imgaussfilt(A, sigma, 'Padding', 'symmetric', 'FilterSize', filterSize); % Calculate shading
    shading = cv2.GaussianBlur(A, (filterSize, filterSize), sigma, borderType=cv2.BORDER_REFLECT)

    meanVal = np.mean(A)  # meanVal = mean(A(:),'omitnan')

    # % Limit minimum to 1e-6 instead of testing using isnan and isinf after division.
    shading = np.maximum(shading, 1e-6)  # shading = max(shading, 1e-6);

    B = A * meanVal / shading  # B = A*meanVal./shading;

    # % Put processed V channel back into HSV image, convert to RGB
    Ihsv[:, :, 2] = B  # Ihsv(:,:,3) = B;

    B = cv2.cvtColor(Ihsv, cv2.COLOR_HSV2BGR)  # B = hsv2rgb(Ihsv);

    B = np.round(np.clip(B * 255, 0, 255)).astype(np.uint8)  # B = im2uint8(B);

    return B

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
    limit=100
    safety_threshold=50
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

