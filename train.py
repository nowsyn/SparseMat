import os
import shutil
import argparse
import numpy as np
import cv2
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from datasets import Rescale, RescaleT, RandomFlip, RandomCrop, ToTensor, CustomDataset
from model import SparseMat, losses
from utils import load_config, grid_images, get_logger


def get_timestamp():
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    return dt_string


def adjust_learning_rate(optimizer, epoch, epoch_decay, init_lr, min_lr=1e-6):
    for param_group in optimizer.param_groups:
        lr = max(init_lr * (0.1 ** (epoch // epoch_decay)), min_lr)
        param_group['lr'] = lr


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
    net.load_state_dict(filtered_state_dict, strict=False)
    logger.info('load pretrained weight from {} successfully'.format(pretrained_model))


def save_checkpoint(cfg, net, optimizer, epoch, iterations, running_loss, best_mad, is_best=False):
    state_dict = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iterations + 1,
        'running_loss': running_loss,
        'best_mad': best_mad,
    }
    save_path = os.path.join(cfg.log.log_dir, "ckpt_e{}.pth".format(epoch))
    torch.save(state_dict, save_path)

    latest_path = os.path.join(cfg.log.log_dir, "ckpt_latest.pth")
    shutil.copy(save_path, latest_path)

    if is_best:
        best_path = os.path.join(cfg.log.log_dir, "ckpt_best.pth")
        shutil.copy(save_path, best_path)


def save_preds(pred, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    pred = pred.squeeze().data.cpu().numpy() * 255
    imgname = filename.split('/')[-1].split('.')[0] + '.png'
    cv2.imwrite(os.path.join(save_dir, imgname), pred)


def load_filelist(data_path):
    images = []
    labels = []
    fgs = []
    bgs = []
    for line in open(data_path).read().splitlines():
        splits = line.split(',')
        if len(splits) == 4:
            img_path, lbl_path, fg_path, bg_path = splits
            images.append(img_path)
            labels.append(lbl_path)
            fgs.append(fg_path)
            bgs.append(bg_path)
        else:
            img_path, lbl_path = splits
            images.append(img_path)
            labels.append(lbl_path)
    return images, labels, fgs, bgs


def compute_metrics(pred, gt):
    if pred.shape[2:] != gt.shape[2:]:
        pred = F.interpolate(pred, gt.shape[2:], mode='bilinear', align_corners=False)
    mad = (pred-gt).abs().mean()
    mse = ((pred-gt)**2).mean()
    return mad, mse


def train(cfg, net, optimizer, criterion, dataloader, writer, logger, epoch, iterations, best_mad):
    net.train()
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        iterations += 1

        input_dict = {}
        for k, v in data.items():
            input_dict[k] = v.cuda()

        optimizer.zero_grad()
        pred_list = net(input_dict)
        loss_dict = criterion(pred_list, input_dict)
        loss_dict['loss'].backward()
        optimizer.step()

        running_loss += loss_dict['loss'].item()

        cur_lr = optimizer.param_groups[0]['lr']

        if iterations % cfg.log.print_frq == 0:
            for k,v in loss_dict.items():
                writer.add_scalar('loss/'+k, loss_dict[k].item(), iterations)
            writer.add_scalar('loss/running_loss', running_loss/(i+1), iterations)
            writer.add_image('train/images', torch.cat(torch.unbind(pred_list[-1], dim=0), dim=1), global_step=iterations)
            if 'comp_loss' in loss_dict:
                logger.info('[epo:%d/%d][iter:%d/%d] lr:%5f loss:%.3f alpha_loss:%.3f comp_Loss:%.3f running_loss:%.3f' % (
                    epoch, cfg.train.epoch, (i+1), len(dataloader), cur_lr, loss_dict['loss'],
                    loss_dict['alpha_loss'], loss_dict['comp_loss'],
                    running_loss/(i+1)))
            else:
                logger.info('[epo:%d/%d][iter:%d/%d] lr:%5f loss:%.3f running_loss:%.3f' % (
                    epoch, cfg.train.epoch, (i+1), len(dataloader), cur_lr, loss_dict['loss'], running_loss/(i+1)))

        # comment this line if memory is sufficient
        torch.cuda.empty_cache()

    return iterations, running_loss


def test(cfg, net, dataloader, writer, logger, epoch, filenames):
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

            gt = input_dict['hr_label']
            mad, mse = compute_metrics(pred, gt)
            mse_list.append(mse.item())
            mad_list.append(mad.item())

            logger.info('[ith:%d/%d] mad:%.5f mse:%.5f' % (i, len(dataloader), mad.item(), mse.item()))

    avg_mad = np.array(mad_list).mean()
    avg_mse = np.array(mse_list).mean()
    logger.info('[epo:%d/%d][ith:%d/%d] mad:%.3f mse:%.5f' % (epoch, cfg.train.epoch, i, len(dataloader), mad.item(), mse.item()))
    return avg_mad


def main():
    parser = argparse.ArgumentParser(description='HM')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', help='use distributed training')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate or not')
    parser.add_argument('-c', '--config', type=str, metavar='FILE', help='path to config file')
    parser.add_argument('-p', '--phase', default="train", type=str, metavar='PHASE', help='train or test')

    args = parser.parse_args()
    cfg = load_config(args.config)
    best_mad = 1e12
    device_ids = range(torch.cuda.device_count())

    dataset = cfg.data.dataset
    model_name = cfg.model.arch
    exp_name = args.config.split('/')[-1].split('.')[0]
    timestamp = get_timestamp()

    cfg.log.log_dir = os.path.join(os.getcwd(), 'log', model_name, dataset, exp_name+os.sep)
    cfg.log.viz_dir = os.path.join(cfg.log.log_dir, "tensorboardx", timestamp)
    cfg.log.log_path = os.path.join(cfg.log.log_dir, "log.txt")
    os.makedirs(cfg.log.log_dir, exist_ok=True)
    os.makedirs(cfg.log.viz_dir, exist_ok=True)

    if cfg.test.save_dir is None:
        cfg.test.save_dir = os.path.join(cfg.log.log_dir, 'vis')
        os.makedirs(cfg.test.save_dir, exist_ok=True)

    writer = SummaryWriter(cfg.log.viz_dir)
    logger = get_logger(cfg.log.log_path)

    logger.info('[LogPath] {}'.format(cfg.log.log_dir))
    logger.info('[VizPath] {}'.format(cfg.log.viz_dir))

    train_images, train_labels, train_fgs, train_bgs = load_filelist(cfg.data.filelist_train)
    test_images, test_labels, test_fgs, test_bgs = load_filelist(cfg.data.filelist_val)

    train_transform = transforms.Compose([
        Rescale(cfg),
        RandomCrop(cfg),
        RandomFlip(cfg),
        ToTensor(cfg)
    ])

    test_transform = transforms.Compose([
        RescaleT(cfg),
        ToTensor(cfg)
    ])

    train_dataset = CustomDataset(
        cfg, True,
        img_name_list=train_images,
        lbl_name_list=train_labels,
        fg_name_list=train_fgs,
        bg_name_list=train_bgs,
        transform=train_transform
    )
    test_dataset = CustomDataset(
        cfg, False,
        img_name_list=test_images,
        lbl_name_list=test_labels,
        fg_name_list=test_fgs,
        bg_name_list=test_bgs,
        transform=test_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=cfg.train.num_workers
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=cfg.test.num_workers
    )

    net = SparseMat(cfg)
    criterion = partial(
        losses,
        alpha_loss_weights=cfg.loss.alpha_loss_weights,
        with_composition_loss=cfg.loss.with_composition_loss,
        composition_loss_weight=cfg.loss.composition_loss_weight,
    )

    load_checkpoint(net.lpn, cfg.train.pretrained_model, logger)

    if torch.cuda.is_available():
        net.cuda()
    else:
        exit()

    if len(device_ids)>0:
        net = torch.nn.DataParallel(net)
        net_without_dp = net.module
    else:
        net_without_dp = net

    logger.info("---define optimizer...")
    optimizer = optim.Adam(
        net.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.train.beta1, cfg.train.beta2),
        eps=1e-08,
        weight_decay=0,
    )

    logger.info("---start training...")
    iterations = 0
    running_loss = 0.0

    resume_checkpoint = os.path.join(cfg.log.log_dir, 'ckpt_latest.pth')
    if (args.evaluate or cfg.train.resume) and os.path.exists(resume_checkpoint):
        state_dict = torch.load(resume_checkpoint)
        if state_dict['epoch'] < cfg.train.epoch:
            logger.info("Resume checkpoint from {}".format(resume_checkpoint))
            if 'best_mad' in state_dict:
                best_mad = state_dict['best_mad']
            if 'epoch' in state_dict:
                cfg.train.start_epoch = state_dict['epoch']
            filtered_state_dict = OrderedDict()
            for k,v in state_dict['state_dict'].items():
                if k.startswith('module'):
                    nk = '.'.join(k.split('.')[1:])
                else:
                    nk = k
                filtered_state_dict[nk] = v
            net.module.load_state_dict(filtered_state_dict, strict=True)

    if args.evaluate:
        test(cfg, net_without_dp, test_dataloader, writer, logger, cfg.train.start_epoch, test_images)
        exit()

    for epoch in range(cfg.train.start_epoch, cfg.train.epoch):
        iterations, running_loss = train(cfg, net, optimizer, criterion, train_dataloader, writer, logger, epoch+1, iterations, best_mad)
        mad = test(cfg, net_without_dp, test_dataloader, writer, logger, epoch+1, test_images)
        if mad < best_mad:
            best_mad = min(mad, best_mad)
            save_checkpoint(cfg, net_without_dp, optimizer, epoch+1, iterations, running_loss, best_mad, is_best=True)
        else:
            save_checkpoint(cfg, net_without_dp, optimizer, epoch+1, iterations, running_loss, best_mad, is_best=False)
        adjust_learning_rate(optimizer, epoch, cfg.train.epoch_decay, cfg.train.lr, min_lr=cfg.train.min_lr)


if __name__ == "__main__":
    main()
