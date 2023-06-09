import numpy as np
from collections import OrderedDict
from scipy.ndimage import morphology

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv(conv):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def _init_norm(norm):
    if norm.weight is not None:
        nn.init.constant_(norm.weight, 1)
        nn.init.constant_(norm.bias, 0)


def _generate_random_trimap(x, dist=(1,30), is_training=True):
    fg = (x>0.999).type(torch.float)
    un = (x>=0.001).type(torch.float) - fg
    un_np = (un*255).squeeze(1).data.cpu().numpy().astype(np.uint8)
    if is_training:
        thresh = np.random.randint(dist[0], dist[1])
    else:
        thresh = (dist[0] + dist[1]) // 2
    un_np = [(morphology.distance_transform_edt(item==0) <= thresh) for item in un_np]
    un_np = np.array(un_np)
    un = torch.from_numpy(un_np).unsqueeze(1).to(x.device)
    trimap = fg
    trimap[un>0] = 0.5
    return trimap


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar,mode='bilinear'):
    src = F.interpolate(src,size=tar.shape[2:],mode=mode,align_corners=False if mode=='bilinear' else None)
    return src
upas = _upsample_like


def batch_slice(tensor, pos, size, mode='bilinear'):
    n, c, h, w = tensor.shape
    patchs = []
    for i in range(n):
        # x1, y1, x2, y2 = torch.clamp(pos[i], 0, 1)
        x1, y1, x2, y2 = pos[i]
        x1 = int(x1.item() * w)
        y1 = int(y1.item() * h)
        x2 = int(x2.item() * w)
        y2 = int(y2.item() * h)
        patch = tensor[i:i+1, :, y1:y2, x1:x2].contiguous()
        patch = F.interpolate(patch, (size[0], size[1]), align_corners=False if mode=='bilinear' else None, mode=mode)
        patchs.append(patch)
    return torch.cat(patchs, dim=0)


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


## copy weight from old tensor to new tensor
def copy_weight(ws, wd):

    assert len(ws.shape)==4 or len(ws.shape)==1

    if len(ws.shape) == 4 and ws.shape[2]==ws.shape[3] and ws.shape[3]<=7:
        cout1, cin1, kh, kw = ws.shape
        cout2, cin2, kh, kw = wd.shape
        weight = torch.zeros((cout2, cin2, kh, kw)).float().to(ws.device)
        cout3 = min(cout1, cout2)
        cin3 = min(cin1, cin2)
        weight[:cout3, :cin3] = ws[:cout3, :cin3]
    elif len(ws.shape) == 4:
        kh, kw, cin1, cout1 = ws.shape # (3,3,4,64)
        kh, kw, cin2, cout2 = wd.shape
        print(ws.shape, wd.shape)
        weight = torch.zeros((kh, kw, cin2, cout2)).float().to(ws.device)
        cout3 = min(cout1, cout2)
        cin3 = min(cin1, cin2)
        weight[:, :, :cin3, :cout3] = ws[:, :, :cin3, :cout3]
    else:
        cout1, = ws.shape
        cout2, = wd.shape
        cout3 = min(cout1, cout2)
        weight = torch.zeros((cout3,)).float().to(ws.device)
        weight[:cout3] = ws[:cout3]
    return weight


## only works for models with same architecture
def load_pretrained_weight(model, ckpt_path, copy=True):
   ckpt = torch.load(ckpt_path)
   filtered_ckpt = OrderedDict()
   model_ckpt = model.state_dict()
   for k,v in ckpt.items():
       if k in model_ckpt:
           if v.shape==model_ckpt[k].shape:
               filtered_ckpt[k] = v
           elif copy:
               filtered_ckpt[k] = copy_weight(v, model_ckpt[k])
   model.load_state_dict(filtered_ckpt, strict=False)
   return model
