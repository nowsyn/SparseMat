import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import upas, batch_slice
from model.lpn import LPN
from model.shm import SHM


class SparseMat(nn.Module):
    def __init__(self, cfg):
        super(SparseMat, self).__init__()
        self.cfg = cfg
        in_ch = cfg.model.in_channel
        hr_ch = cfg.model.hr_channel
        self.lpn = LPN(in_ch, hr_ch)
        self.shm = SHM(inc=4)
        self.stride = cfg.model.dilation_kernel
        self.dilate_op = nn.MaxPool2d(self.stride, stride=1, padding=self.stride//2)
        self.max_n_pixel = cfg.model.max_n_pixel
        self.cascade = cfg.test.cascade

    @torch.no_grad()
    def generate_sparse_inputs(self, img, lr_pred, mask):
        lr_pred = (lr_pred - 0.5) / 0.5
        x = torch.cat((img, lr_pred), dim=1)
        indices = torch.where(mask.squeeze(1)>0)
        x = x.permute(0,2,3,1)
        x = x[indices]
        indices = torch.stack(indices, dim=1)
        return x, indices

    def dilate(self, alpha, stride=15):
        mask = torch.logical_and(alpha>0.01, alpha<0.99).float()
        mask = self.dilate_op(mask)
        return mask

    def forward(self, input_dict):
        xlr = input_dict['lr_image']
        xhr = input_dict['hr_image']

        lr_pred, ctx = self.lpn(xlr)
        lr_pred = lr_pred.clone().detach()
        ctx = ctx.clone().detach()

        lr_pred = batch_slice(lr_pred, input_dict['pos'], xhr.size()[2:])
        lr_pred = upas(lr_pred, xhr)
        if 'hr_unknown' in input_dict:
            mask = input_dict['hr_unknown']
        else:
            mask = self.dilate(lr_pred)

        sparse_inputs, coords = self.generate_sparse_inputs(xhr, lr_pred, mask=mask)
        pred_list = self.shm(sparse_inputs, lr_pred, coords, xhr.size(0), mask.size()[2:], ctx=ctx)
        return pred_list

    def generate_sparsity_map(self, lr_pred, curr_img, last_img):
        mask_s = self.dilate(lr_pred)
        if last_img is not None:
            diff = (curr_img - last_img).abs().mean(dim=1, keepdim=True)
            shared = torch.logical_and(
                F.conv2d(diff, torch.ones(1,1,9,9,device=diff.device), padding=4) < 0.05,
                F.conv2d(diff, torch.ones(1,1,1,1,device=diff.device), padding=0) < 0.001,
            ).float()
            mask_t = self.dilate_op(1 - shared)
            mask = mask_s * mask_t
            mask = self.dilate_op(mask)
        else:
            shared = torch.zeros_like(mask_s)
            mask_t = torch.ones_like(mask_s)
            mask = mask_s * mask_t
        return mask, mask_s, mask_t, shared

    def inference(self, hr_img, lr_img=None, last_img=None, last_pred=None):
        h, w = hr_img.shape[-2:]

        if lr_img is None:
            nh = 512. / min(h,w) * h
            nh = math.ceil(nh / 32) * 32
            nw = 512. / min(h,w) * w
            nw = math.ceil(nw / 32) * 32
            lr_img = F.interpolate(hr_img, (int(nh), int(nw)), mode="bilinear")

        lr_pred, ctx = self.lpn(lr_img)
        lr_pred_us = upas(lr_pred, hr_img)
        mask, mask_s, mask_t, shared = self.generate_sparsity_map(lr_pred_us, hr_img, last_img)
        n_pixel = mask.sum().item()

        if n_pixel <= self.max_n_pixel:
            sparse_inputs, coords = self.generate_sparse_inputs(hr_img, lr_pred_us, mask)
            preds = self.shm(sparse_inputs, lr_pred_us, coords, hr_img.size(0), mask.size()[2:], ctx=ctx)
            hr_pred_sp = preds[-1]
            if last_pred is not None:
                hr_pred = hr_pred_sp * mask + lr_pred_us * (1-mask) * (1-shared) + last_pred * (1-mask) * shared
            else:
                hr_pred = hr_pred_sp * mask + lr_pred_us * (1-mask)
        elif self.cascade:
            print("Cascading is used.")
            for scale in [0.25, 0.5, 1.0]:
                hr_img_ds = F.interpolate(hr_img, None, scale_factor=scale, mode="bilinear")
                lr_pred_us = upas(lr_pred, hr_img_ds)
                mask_s = self.dilate(lr_pred_us)
                if mask_s.sum() > self.max_n_pixel:
                    break
                sparse_inputs, coords = self.generate_sparse_inputs(hr_img_ds, lr_pred_us, mask_s)
                preds = self.shm(sparse_inputs, lr_pred_us, coords, hr_img_ds.size(0), mask_s.size()[2:], ctx=ctx)
                hr_pred_sp = preds[-1]
                hr_pred = hr_pred_sp * mask_s + lr_pred_us * (1-mask_s)
                lr_pred = hr_pred
        else:
            print("Rescaling is used.")
            scale = math.sqrt(self.max_n_pixel / float(n_pixel))
            nh = int(scale * h)
            nw = int(scale * w)
            nh = math.ceil(nh / 8) * 8
            nw = math.ceil(nw / 8) * 8

            hr_img_ds = F.interpolate(hr_img, (nh, nw), mode="bilinear")
            lr_pred_us = upas(lr_pred, hr_img_ds)
            mask_s = self.dilate(lr_pred_us)

            sparse_inputs, coords = self.generate_sparse_inputs(hr_img_ds, lr_pred_us, mask_s)
            preds = self.shm(sparse_inputs, lr_pred_us, coords, hr_img_ds.size(0), mask_s.size()[2:], ctx=ctx)
            hr_pred_sp = preds[-1]
            hr_pred = hr_pred_sp * mask_s + lr_pred_us * (1-mask_s)
        return hr_pred
