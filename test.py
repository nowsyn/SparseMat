import os
import argparse
import numpy as np
import cv2
from collections import OrderedDict
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import SparseMat
from utils import load_config, get_logger
from datasets import RescaleT, ToTensor, CustomDataset


def get_timestamp():
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    return dt_string


def load_checkpoint(net, pretrained_model, logger):
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
    logger.info('load pretrained weight from {} successfully'.format(pretrained_model))


def load_test_filelist(test_data_path):
    test_images = []
    test_labels = []
    for line in open(test_data_path).read().splitlines():
        splits = line.split(',')
        img_path, mat_path = splits
        test_labels.append(mat_path)
        test_images.append(img_path)
    return test_images, test_labels


def compute_metrics(pred, gt):
    assert pred.size(0)==1 and pred.size(1)==1
    if pred.shape[2:] != gt.shape[2:]:
        pred = F.interpolate(pred, gt.shape[2:], mode='bilinear', align_corners=False)
    mad = (pred-gt).abs().mean()
    mse = ((pred-gt)**2).mean()
    return mse, mad


def save_preds(pred, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    pred = pred.squeeze().data.cpu().numpy() * 255
    imgname = filename.split('/')[-1].split('.')[0] + '.png'
    cv2.imwrite(os.path.join(save_dir, imgname), pred)


def test(cfg, net, dataloader, filenames, logger):
    net.eval()

    mse_list = []
    mad_list = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input_dict = {}
            for k, v in data.items():
                input_dict[k] = v.cuda()

            pred = net.inference(input_dict['hr_image'])
            origin_h = input_dict['origin_h']
            origin_w = input_dict['origin_w']
            pred = F.interpolate(pred, (origin_h, origin_w), align_corners=False, mode="bilinear")

            if cfg.test.save:
                save_preds(pred, cfg.test.save_dir, filenames[i])

            gt = input_dict['hr_label']
            mse, mad = compute_metrics(pred, gt)
            mse_list.append(mse.item())
            mad_list.append(mad.item())

            logger.info('[ith:%d/%d] mad:%.5f mse:%.5f' % (i, len(dataloader), mad.item(), mse.item()))

    avg_mad = np.array(mad_list).mean()
    avg_mse = np.array(mse_list).mean()
    logger.info('avg_mad:%.5f avg_mse:%.5f' % (avg_mad.item(), avg_mse.item()))


def main():
    parser = argparse.ArgumentParser(description='HM')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', help='use distributed training')
    parser.add_argument('-c', '--config', type=str, metavar='FILE', help='path to config file')
    parser.add_argument('-p', '--phase', default="train", type=str, metavar='PHASE', help='train or test')

    args = parser.parse_args()
    cfg = load_config(args.config)
    device_ids = range(torch.cuda.device_count())

    dataset = cfg.data.dataset
    model_name = cfg.model.arch
    exp_name = args.config.split('/')[-1].split('.')[0]
    timestamp = get_timestamp()

    cfg.log.log_dir = os.path.join(os.getcwd(), 'log', model_name, dataset, exp_name+os.sep)
    cfg.log.log_path = os.path.join(cfg.log.log_dir, "log_eval.txt")
    os.makedirs(cfg.log.log_dir, exist_ok=True)

    if cfg.test.save_dir is None:
        cfg.test.save_dir = os.path.join(cfg.log.log_dir, 'vis')
        os.makedirs(cfg.test.save_dir, exist_ok=True)

    logger = get_logger(cfg.log.log_path)
    logger.info('[LogPath] {}'.format(cfg.log.log_dir))

    test_images, test_labels = load_test_filelist(cfg.data.filelist_test)

    test_transform = transforms.Compose([
        RescaleT(cfg),
        ToTensor(cfg)
    ])

    test_dataset = CustomDataset(
        cfg,
        is_training=False,
        img_name_list=test_images,
        lbl_name_list=test_labels,
        transform=test_transform
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.test.num_workers
    )

    net = SparseMat(cfg)

    if torch.cuda.is_available():
        net.cuda()
    else:
        exit()

    load_checkpoint(net, cfg.test.checkpoint, logger)
    test(cfg, net, test_dataloader, test_images, logger)


if __name__ == "__main__":
    main()
