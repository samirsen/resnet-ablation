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



def build_featurizer(in_channel, out_channel):
    return [
        nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2,
                              padding=3, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    ]

def build_res_block(in_channels, out_channels):
    return nn.Sequential(
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        conv3x3(out_channels, out_channels),
        nn.BatchNorm2d(out_channels)
    )


def build_base_resnet(in_channels, out_channels, block, num_classes=100, perturb=False):
    res_layers = []
    res_layers += build_featurizer(3, 64)
    res_layers += build_skip_connectors(block)
    res_layers += [nn.AdaptiveAvgPool2d((1, 1))]
    res_layers += [nn.Linear(512 * block.expansion, num_classes)]

    if perturb:
        pass

    return nn.Sequential(*res_layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1, downsample=None):
        Super(ResBlock, self).__init__()

        self.expansion = expansion
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


        if init_weight: self.init_weights()



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
