import os
import argparse
import numpy as np
import cv2
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SparseMat
from utils import load_config


def load_checkpoint(net, pretrained_model):
    net_state_dict = net.state_dict()
    state_dict = torch.load(pretrained_model)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    filtered_state_dict = OrderedDict()
    for k,v in state_dict.items():
        if k.startswith('module'):
            nk = '.'.join(k.split('.')[1:])
        else:
            nk = k
        filtered_state_dict[nk] = v
    net.load_state_dict(filtered_state_dict)
    print('load pretrained weight from {} successfully'.format(pretrained_model))


def preprocess(image):
    image = (image / 255. - 0.5) / 0.5
    image = torch.from_numpy(image[None]).permute(0,3,1,2)
    h, w = image.shape[2:]
    nh = math.ceil(h / 8) * 8
    nw = math.ceil(w / 8) * 8
    image = F.interpolate(image, (nh, nw), mode="bilinear")
    return image.float().cuda()


def run_single_image(net, input_path, save_dir):
    filename = input_path.split('/')[-1]
    image = cv2.imread(input_path)
    origin_h, origin_w = image.shape[:2]
    tensor = preprocess(image)
    with torch.no_grad():
        pred = net.inference(tensor)
    pred = F.interpolate(pred, (origin_h, origin_w), align_corners=False, mode="bilinear")
    pred_alpha = (pred * 255).squeeze().data.cpu().numpy().astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, filename), pred_alpha)
    return pred


def run_multiple_images(net, input_path, save_dir):
    for item in os.listdir(input_path):
        run_single_image(net, os.path.join(input_path, item), save_dir)


def run_video(net, input_path, save_dir):
    filename = input_path.split('/')[-1]
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(save_dir, filename), fourcc, fps, (width, height))

    last_frame = None
    last_pred = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tensor = preprocess(frame)
        with torch.no_grad():
            pred = net.inference(tensor, last_img=last_frame, last_pred=last_pred)
        pred = F.interpolate(pred, (height, width), align_corners=False, mode="bilinear")
        pred_alpha = (pred * 255).squeeze().data.cpu().numpy().astype(np.uint8)
        writer.write(np.tile(pred_alpha[:,:,None], (1,1,3)))
        last_frame = tensor
        last_pred = pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, metavar='FILE', help='path to config file')
    parser.add_argument('--input', type=str, metavar='PATH', help='path to input path')
    parser.add_argument('--save_dir', type=str, metavar='PATH', help='path to save path')

    args = parser.parse_args()
    cfg = load_config(args.config)

    os.makedirs(args.save_dir, exist_ok=True)

    net = SparseMat(cfg)

    if torch.cuda.is_available():
        net.cuda()
    else:
        exit()

    load_checkpoint(net, cfg.test.checkpoint)

    net.eval()

    if args.input.endswith(".mp4"):
        run_video(net, args.input, args.save_dir)
    elif args.input.endswith(".jpg") or args.input.endswith(".png"):
        run_single_image(net, args.input, args.save_dir)
    else:
        run_multiple_images(net, args.input, args.save_dir)


if __name__ == "__main__":
    main()
