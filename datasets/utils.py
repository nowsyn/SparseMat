import torch
import numpy as np
import random
import math
import cv2

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
from skimage import io, transform, color


def get_random_patch(mask, crop_size):
    new_h, new_w = mask.shape[:2]
    crop_size = min(crop_size, min(new_w, new_h)-1)
    crop_size_hf = crop_size // 2
    maskf = mask / 255.
    ys, xs = np.where(np.logical_and(maskf>0.05, maskf<0.95))[:2]
    if len(ys)>0:
        rand_ind = random.randint(0, len(ys)-1)
        cy = min(max(ys[rand_ind], crop_size_hf), new_h-crop_size_hf)
        cx = min(max(xs[rand_ind], crop_size_hf), new_w-crop_size_hf)
        x1, y1 = cx - crop_size_hf, cy - crop_size_hf
        x2, y2 = x1 + crop_size, y1 + crop_size
    else:
        x1, y1 = new_w // 2 - crop_size_hf, new_h // 2 - crop_size_hf
        x2, y2 = x1 + crop_size, y1 + crop_size
    return (x1,y1,x2,y2)


def convert_color_space(image, flag=3):
    if flag == 3:
        image = image / 255.0
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        if image.shape[2]==1:
            tmpImg[:] = 2 * np.tile(image[:,:,None],(1,1,3)) - 1
        else:
            tmpImg[:] = 2 * image[:] - 1

    elif flag == 2: # with rgb and Lab colors
        tmpImg = np.zeros((image.shape[0],image.shape[1],6))
        tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
        if image.shape[2]==1:
            tmpImgt[:,:,0] = image[:,:,0]
            tmpImgt[:,:,1] = image[:,:,0]
            tmpImgt[:,:,2] = image[:,:,0]
        else:
            tmpImgt = image
        tmpImgtl = color.rgb2lab(tmpImgt)

        # nomalize image to range [0,1]
        tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
        tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
        tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
        tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
        tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
        tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

        # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
        tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
        tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
        tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

    elif flag == 1: #with Lab color
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))

        if image.shape[2]==1:
            tmpImg[:,:,0] = image[:,:,0]
            tmpImg[:,:,1] = image[:,:,0]
            tmpImg[:,:,2] = image[:,:,0]
        else:
            tmpImg = image

        tmpImg = color.rgb2lab(tmpImg)

        # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

    else: # with rgb color
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        image = image/np.max(image)
        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

    return tmpImg
