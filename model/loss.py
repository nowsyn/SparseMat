import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lap_pyramid_loss import LapLoss


def matting_loss(p, d, mask=None, with_lap=False):
    assert p.shape == d.shape

    if mask is not None:
        loss = torch.sqrt((p - d) ** 2 + 1e-10) * mask
        loss = loss.sum() / (mask.sum() + 1)
    else:
        loss = torch.sqrt((p - d) ** 2 + 1e-10)
        loss = loss.mean()

    if with_lap:
        lap_loss = LapLoss(5, device=torch.device('cuda'))
        loss = loss + lap_loss(p, d)
    return loss


def composition_loss(alpha, img, fg, bg, mask):
    comp = alpha * fg + (1. - alpha) * bg
    diff = (comp - img) * mask
    loss = torch.sqrt(diff ** 2 + 1e-12)
    loss = loss.sum() / (mask.sum() + 1.) / 3.
    return loss


def losses(pred_list, input_dict, alpha_loss_weights=[1.0, 1.0, 1.0, 1.0], with_composition_loss=False, composition_loss_weight=1.0):
    label = input_dict['hr_label']
    mask = input_dict['hr_unknown']

    loss_dict = {}

    alpha_loss = 0.
    for i, pred in enumerate(pred_list):
        stride = label.size(2) / pred.size(2)
        pred = F.interpolate(pred, scale_factor=stride, mode='bilinear', align_corners=False)
        alpha_loss += matting_loss(pred, label, mask, with_lap=True) * alpha_loss_weights[i]
    loss_dict['alpha_loss'] = alpha_loss

    if with_composition_loss:
        comp_loss = composition_loss(pred_list[-1], input_dict['hr_image'],
            input_dict['hr_fg'], input_dict['hr_bg'], mask) * composition_loss_weight
        loss_dict['comp_loss'] = comp_loss

    loss = 0.
    for k, v in loss_dict.items():
        if k.endswith('loss'):
            loss += v
    loss_dict['loss'] = loss
    return loss_dict
