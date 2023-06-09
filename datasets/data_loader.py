#  data loader
from __future__ import print_function, division
import os
import glob
import numpy as np
import random
import math
import cv2
from PIL import Image
from skimage import io, transform, color

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset, DataLoader
from torchvision import transforms, utils

from .utils import convert_color_space, get_random_patch


def imread(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class RandomFlip(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, sample):
        # randomly flip
        if random.random() >= 0.5:
            pos = sample['pos']
            x1 = 1. - pos[..., 2]
            x2 = 1. - pos[..., 0]
            pos[..., 0] = x1
            pos[..., 2] = x2
            sample['pos'] = pos
            sample['hr_image'] = sample['hr_image'][:,::-1].copy()
            sample['lr_image'] = sample['lr_image'][:,::-1].copy()
            sample['hr_label'] = sample['hr_label'][:,::-1].copy()
            sample['hr_unknown'] = sample['hr_unknown'][:,::-1].copy()
            if 'hr_fg' in sample:
                sample['hr_fg'] = sample['hr_fg'][:,::-1].copy()
                sample['hr_bg'] = sample['hr_bg'][:,::-1].copy()
        return sample


class Rescale(object):
    def __init__(self, cfg):
        assert isinstance(cfg.aug.rescale_size,(int,tuple))
        self.output_size = cfg.aug.rescale_size

    def __call__(self,sample):
        h, w = sample['hr_image'].shape[:2]
        sample['origin_h'] = h
        sample['origin_w'] = w
        if isinstance(self.output_size,int):
            ratio = self.output_size / min(h,w)
            new_h, new_w = ratio*h, ratio*w
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        sample['lr_image'] = cv2.resize(sample['hr_image'], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return sample


class RescaleT(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_size = cfg.test.max_size
        self.output_size = cfg.test.rescale_size
        assert isinstance(self.output_size,(int,tuple))

    def get_dst_size(self, origin_size, output_size=None, stride=32, max_size=1920):
        h, w = origin_size
        if output_size is None:
            ratio = max_size / max(h,w)
            if ratio>=1:
                new_h, new_w = h, w
            else:
                new_h, new_w = int(math.ceil(ratio*h)), int(math.ceil(ratio*w))
        elif isinstance(output_size,int):
            if output_size>=max_size:
                ratio = output_size / max(h,w)
            else:
                ratio = output_size / min(h,w)
            new_h, new_w = int(math.ceil(ratio*h)), int(math.ceil(ratio*w))
        else:
            new_h, new_w = output_size
        new_h = new_h - new_h % 32
        new_w = new_w - new_w % 32
        return (new_h, new_w)

    def __call__(self,sample):
        h, w = sample['hr_image'].shape[:2]
        sample['origin_h'] = h
        sample['origin_w'] = w
        new_h, new_w = self.get_dst_size((h,w), self.output_size, 32)
        sample['lr_image'] = cv2.resize(sample['hr_image'], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return sample


class RandomCrop(object):

    def __init__(self, cfg):
        # low-resolution full image
        output_size = cfg.aug.crop_size
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        # full-resolution patch
        patch_crop_size = cfg.aug.patch_crop_size
        assert isinstance(patch_crop_size, (tuple, list))
        self.patch_crop_size = patch_crop_size

        patch_load_size = cfg.aug.patch_load_size
        assert isinstance(patch_load_size, int)
        self.patch_load_size = patch_load_size

        self.cfg = cfg

    def random_crop(self, sample):
        h, w = sample['lr_image'].shape[:2]
        new_h, new_w = self.output_size
        ly1 = np.random.randint(0, h - new_h)
        lx1 = np.random.randint(0, w - new_w)
        ly2 = ly1 + new_h
        lx2 = lx1 + new_w

        oh, ow = sample['hr_image'].shape[:2]
        ratio_h = oh / float(h)
        ratio_w = ow / float(w)
        hx1, hy1 = int(lx1*ratio_w), int(ly1*ratio_h)
        hx2, hy2 = int(lx2*ratio_w), int(ly2*ratio_h)
        return (lx1,ly1,lx2,ly2), (hx1,hy1,hx2,hy2)

    def __call__(self,sample):
        (lx1,ly1,lx2,ly2), (hx1,hy1,hx2,hy2) = self.random_crop(sample)
        sample['lr_image'] = sample['lr_image'][ly1:ly2, lx1:lx2]
        sample['hr_image'] = sample['hr_image'][hy1:hy2, hx1:hx2]
        sample['hr_label'] = sample['hr_label'][hy1:hy2, hx1:hx2]
        sample['hr_unknown'] = sample['hr_unknown'][hy1:hy2, hx1:hx2]

        # random crop from high-resolution input
        h, w = sample['hr_label'].shape[:2]
        random_crop_size = random.choice(self.patch_crop_size)
        px1,py1,px2,py2 = get_random_patch(sample['hr_label'], random_crop_size)
        pos = np.array([px1/w,py1/h,px2/w,py2/h]).astype(np.float32)
        pos = np.clip(pos, 0, 1)
        sample['pos'] = pos

        load_size = (self.patch_load_size, self.patch_load_size)
        sample['hr_image'] = cv2.resize(sample['hr_image'][py1:py2, px1:px2], load_size, interpolation=cv2.INTER_LINEAR)
        sample['hr_label'] = cv2.resize(sample['hr_label'][py1:py2, px1:px2], load_size, interpolation=cv2.INTER_LINEAR)
        sample['hr_unknown'] = cv2.resize(sample['hr_unknown'][py1:py2, px1:px2], load_size, interpolation=cv2.INTER_NEAREST)

        if 'hr_fg' in sample:
            sample['hr_fg'] = sample['hr_fg'][hy1:hy2, hx1:hx2]
            sample['hr_fg'] = cv2.resize(sample['hr_fg'][py1:py2, px1:px2], load_size, interpolation=cv2.INTER_LINEAR)
            sample['hr_bg'] = sample['hr_bg'][hy1:hy2, hx1:hx2]
            sample['hr_bg'] = cv2.resize(sample['hr_bg'][py1:py2, px1:px2], load_size, interpolation=cv2.INTER_LINEAR)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, cfg):
        self.color_space = cfg.train.color_space

    def __call__(self, sample):
        sample['hr_label'] = sample['hr_label'] / 255.
        sample['hr_label'] = torch.from_numpy(sample['hr_label'][None].astype(np.float32))
        sample['hr_unknown'] = torch.from_numpy(sample['hr_unknown'][None].astype(np.float32))

        sample['hr_image'] = convert_color_space(sample['hr_image'], flag=self.color_space)
        sample['hr_image'] = torch.from_numpy(sample['hr_image'].transpose((2,0,1)).astype(np.float32))
        sample['lr_image'] = convert_color_space(sample['lr_image'], flag=self.color_space)
        sample['lr_image'] = torch.from_numpy(sample['lr_image'].transpose((2,0,1)).astype(np.float32))

        if 'pos' in sample:
            sample['pos'] = torch.from_numpy(sample['pos'].astype(np.float32))
        if 'hr_fg' in sample:
            sample['hr_fg'] = convert_color_space(sample['hr_fg'], flag=self.color_space)
            sample['hr_fg'] = torch.from_numpy(sample['hr_fg'].transpose((2,0,1)).astype(np.float32))
            sample['hr_bg'] = convert_color_space(sample['hr_bg'], flag=self.color_space)
            sample['hr_bg'] = torch.from_numpy(sample['hr_bg'].transpose((2,0,1)).astype(np.float32))
        return sample


class CustomDataset(Dataset):
    def __init__(self,cfg, is_training, img_name_list, lbl_name_list,
                 fg_name_list=None, bg_name_list=None, transform=None):

        self.cfg = cfg
        self.is_training = is_training

        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.fg_name_list = fg_name_list # for composition loss only!!!!!
        self.bg_name_list = bg_name_list # for composition loss only!!!!!

        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):

        sample = {}
        sample['hr_image'] = imread(self.image_name_list[idx])
        sample['hr_label'] = imread(self.label_name_list[idx])[:,:,0]

        unknown = generate_unknown_label(sample['hr_label'], fixed=(not self.is_training))
        mask = (unknown==0) | (unknown==1)
        unknown[mask==1] = 0
        unknown[mask==0] = 1
        sample['hr_unknown'] = unknown

        if self.is_training and len(self.fg_name_list) == len(self.image_name_list):
            fg = imread(self.fg_name_list[idx])
            bg = imread(self.bg_name_list[idx])
            sample['hr_fg'] = fg
            sample['hr_bg'] = bg

        if self.transform:
            sample = self.transform(sample)

        return sample


def generate_unknown_label(alpha, ksize=3, iterations=5, fixed=False):
    oH, oW = alpha.shape[:2]
    if not fixed:
        ksize_range=(3, 9)
        iter_range=(1, 15)
        ksize = random.randint(ksize_range[0], ksize_range[1])
        iterations = random.randint(iter_range[0], iter_range[1])
    else:
        ksize = 5
        iterations = 5
        ratio = 1280. / max(oH,oW)
        alpha = cv2.resize(alpha, None, fx=ratio, fy=ratio)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape) + 128
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    trimap = trimap.astype(np.uint8)
    if trimap.shape[0] != oH or trimap.shape[1] != oW:
        trimap = cv2.resize(trimap, (oW,oH), interpolation=cv2.INTER_NEAREST)
    return trimap
