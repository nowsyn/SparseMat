import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F


def grid_images(pred_dict, input_dict, show_all=False):
    lr_image = input_dict['lr_image'] * 0.5 + 0.5
    lr_label = input_dict['lr_label_mat'].expand_as(lr_image)
    lr_mask = input_dict['lr_label_unk'].expand_as(lr_image)
    lr_pred = pred_dict['coarse']
    if lr_pred.shape[2:] != lr_image.shape[2:]:
        lr_pred = F.interpolate(lr_pred, lr_image.shape[2:], mode="bilinear", align_corners=False)
    lr_pred = lr_pred.expand_as(lr_image)

    h, w = lr_image.size(2), lr_image.size(3)

    tmps = []
    if show_all:
        extra_keys = ['global_seg', 'global_mat', 'errormap', 'classmap']
        for key in extra_keys:
            if key in pred_dict:
                if key == 'errormap':
                   tmp = pred_dict[key]
                   if tmp.size(2) != h or tmp.size(3) != w:
                       tmp = F.interpolate(tmp, (h,w), mode='nearest')
                elif key == 'classmap':
                   tmp = torch.argmax(pred_dict[key], dim=1, keepdim=True).float() / 2.
                   # if tmp.min() < 0:
                   #     tmp = (tmp + 1) / 2.
                   # if tmp.size(2) != h or tmp.size(3) != w:
                   #     tmp = F.interpolate(tmp, (h,w), mode='nearest')
                   # tmp = tmp.repeat(1,3,1,1).float()
                else:
                   tmp = pred_dict[key][0]
                   if tmp.size(2) != h or tmp.size(3) != w:
                       tmp = F.interpolate(tmp, (h,w), mode='bilinear', align_corners=False)
                tmp = tmp.expand_as(lr_image)
                tmps.append(tmp)

    if 'fine' in pred_dict:
        hr_image = input_dict['hr_image'] * 0.5 + 0.5
        hr_label = input_dict['hr_label_mat']
        hr_pred = pred_dict['fine'].expand_as(hr_image)
        if hr_image.size(2) != h or hr_image.size(3) != w:
            hr_image = F.interpolate(hr_image, (h,w), mode='bilinear', align_corners=False)
            hr_label = F.interpolate(hr_label, (h,w), mode='bilinear', align_corners=False)
            hr_pred = F.interpolate(hr_pred, (h,w), mode='bilinear', align_corners=False)
        hr_label = hr_label.expand_as(hr_image)
        grid = torch.cat([lr_image, lr_label, lr_mask, lr_pred]+tmps+[hr_image, hr_label, hr_pred], dim=3)
    else:
        grid = torch.cat([lr_image, lr_label, lr_mask, lr_pred]+tmps, dim=3)
    grid = F.interpolate(grid, scale_factor=0.5, mode='bilinear', align_corners=False)
    n,c,h,w = grid.size()
    grid = grid.permute(1,0,2,3).contiguous().view(c,n*h,w)
    # np_img = cv2.cvtColor(np.transpose(grid.data.cpu().numpy(), (1,2,0))*255, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('tmp/tmp.png', np_img)
    return grid


def save_preds(viz_dir, img, lbl, res1, res2):
    res1 = F.interpolate(res1, (res2.size(2), res2.size(3)))
    img_color = np.transpose(img.data.cpu().numpy(), (0,2,3,1))[:,:,:,::-1]*255
    lbl_color = np.tile(lbl.squeeze().data.cpu().numpy()[:,:,:,None], (1,1,1,3)) * 255
    res1_color = np.tile(torch.clamp(res1,0,1).squeeze().data.cpu().numpy()[:,:,:,None], (1,1,1,3)) * 255
    res2_color = np.tile(torch.clamp(res2,0,1).squeeze().data.cpu().numpy()[:,:,:,None], (1,1,1,3)) * 255
    shows = []
    for i in range(img_color.shape[0]):
        shows.append(np.concatenate((img_color[i], lbl_color[i], res1_color[i], res2_color[i]), axis=1))
    shows = np.concatenate(shows, axis=0)
    ratio = 1200.0 / shows.shape[1]
    shows = cv2.resize(shows, None, fx=ratio, fy=ratio)
    cv2.imwrite(os.path.join(viz_dir,"viz.png"), shows)


def save_labels(labels, save_dir):
    n, c, h, w = labels.shape
    labels = labels[:,0].data.cpu().numpy()

    template = np.zeros((h*n,w,3))

    for i in range(n):
        label_color = idx_to_colormat(labels[i])
        template[h*i:h*(i+1), w*0:w*1] = label_color

    cv2.imwrite(os.path.join(save_dir, "viz.png"), template)


def save_raw_labels(labels, save_dir):
    h, w = labels.shape[:2]
    template = np.zeros((h,w,3))

    label_color = idx_to_colormat(labels)
    cv2.imwrite(os.path.join(save_dir, "viz.png"), label_color)
