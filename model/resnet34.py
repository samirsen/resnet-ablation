# Resnet34 model for ablation analysis with random layer perturbation
# Author: Samir Sen, 2019

import torch
import torch.nn as nn


# From torch model utils (https://github.com/pytorch/vision/blob/master/torchvision/utils.py)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def build_skip_connectors(in_channels=64, out_channels=128, layers = [3, 4, 6, 3]):
    skip_layers = []

    downsample = None
    stride = 1
    for i, num_blocks in enumerate(layers):
        next_in_channels = out_channels
        next_out_channels = out_channels * 2

        if i > 0:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )

        curr_block_layers = []
        curr_block_layers.append(ResBlock(64, out_channels, stride, downsample))

        for _ in range(1, num_blocks):
            curr_block_layers.append(block(out_channels, out_channels))

        skip_layers += curr_block_layers
        in_channels = next_in_channels
        out_channels = next_out_channels

    return nn.Sequential(*skip_layers)


def build_featurizer(in_channel, out_channel):
    return [
        nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2,
                              padding=3, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    ]

def build_base_resnet(in_channels=3, out_channels=64, num_classes=100, perturb=False):
    res_layers = []
    res_layers += build_featurizer(in_channels, out_channels)
    res_layers += build_skip_connectors(out_channels, out_channels*2)
    res_layers += [nn.AdaptiveAvgPool2d((1, 1))]
    res_layers += [nn.Linear(512, num_classes)]

    if perturb:
        pass

    return nn.Sequential(*res_layers)


def build_res_block(in_channels, out_channels):
    return nn.Sequential(
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        conv3x3(out_channels, out_channels),
        nn.BatchNorm2d(out_channels)
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        Super(ResBlock, self).__init__()

        self.expansion = 1
        self.downsample = downsample
        self.res_block = build_res_block(in_channels, out_channels)

    def forward(self, x):
        identity = x

        out = self.res_block(x)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = nn.ReLU(inplace=True)(out)

        return out


class Resnet(nn.Module):
    def __init__(self, num_classes=100, init_weight=False):
        super(Resnet, self).__init__()

        self._resnet = build_base_resnet()
        if init_weight:
            self.init_weights()

    def forward(self, x):
        out = self._resnet(x)

        return out


    def init_weights(self):
        for param in self.modules():
            if isinstance(param, nn.Conv2d):
                nn.init.kaiming_normal_(param.weight, mode='fan_out', nonlinearity='relu')
                if param.bias is not None:
                    nn.init.constant_(param.bias, 0)
            elif isinstance(param, nn.BatchNorm2d):
                nn.init.constant_(param.weight, 1)
                nn.init.constant_(param.bias, 0)
            elif isinstance(param, nn.Linear):
                nn.init.normal_(param.weight, 0, 0.01)
                nn.init.constant_(param.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            elif isinstance(param, BasicBlock):
                nn.init.constant_(param.bn2.weight, 0)
