import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class CategoricalCounting(nn.Module):
    def __init__(self, cls_num=8):
        super(CategoricalCounting, self).__init__()
        self.ccm_cfg = [576, 576, 576, 288, 288, 288]
        self.in_channels = 576
        self.conv1 = nn.Conv2d(288, self.in_channels, kernel_size=1)
        self.ccm = make_layers(self.ccm_cfg, in_channels=self.in_channels, d_rate=2)
        self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(288, cls_num)

    def forward(self, features, spatial_shapes=None):
        features = features.transpose(1, 2)
        bs, c, hw = features.shape
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]

        v_feat = features[:, :, 0:h * w].view(bs, 288, h, w)
        x = self.conv1(v_feat)
        x = self.ccm(x)
        out = self.output(x)
        out = out.squeeze(3)
        out = out.squeeze(2)
        out = self.linear(out)
        # print(out.shape)
        # print(x.shape)

        return out, x


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
