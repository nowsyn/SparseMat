import os
from functools import reduce
from collections import OrderedDict

import torch
import torch.nn as nn

from model.utils import load_pretrained_weight
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import MobileNetV3


class BaseBackbone(nn.Module):
    """ Superclass of Replaceable Backbone Model for Semantic Estimation
    """

    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels

        self.model = None
        self.enc_channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class MobileNetV2Backbone(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels, with_norm=True):
        super(MobileNetV2Backbone, self).__init__(in_channels)

        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None, with_norm=with_norm)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x):
        x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        enc2x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        enc4x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        enc8x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        enc16x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained_model/mobilenetv2_human_seg.ckpt'
        self.model = load_pretrained_weight(self.model, ckpt_path)
        print('load pretrained weight from {} successfully'.format(ckpt_path))


class MobileNetV3LargeBackbone(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels, with_norm=True):
        super(MobileNetV3LargeBackbone, self).__init__(in_channels)

        self.model = MobileNetV3(self.in_channels, num_classes=None, with_norm=with_norm)
        self.enc_channels = [16, 24, 40, 112, 1280]

    def forward(self, x, priors=None):
        x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        enc2x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        enc4x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        enc8x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 13)), x)
        enc16x = x
        x = reduce(lambda x, n: self.model.features[n](x), list(range(13, 17)), x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]
