# VGG 19 Model, Introducing random perturbations to layer activations
# Author: Samir Sen, 2019 

import torch
import torch.nn as nn


def FullyConnected(in_channel, out_channel, num_classes):
    return nn.Sequential(
        nn.Linear(in_channel, out_channel),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(out_channel, out_channel),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(out_channel, num_classes)
    )

def conv4x4(in_channels, out_channels):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ]

def conv2x2(in_channels, out_channels):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ]

def build_base_vgg(in_channel=3):
    vgg_layers = []
    vgg_layers += conv2x2(in_channel, 64)
    vgg_layers += conv2x2(64, 128)
    vgg_layers += conv4x4(128, 256)
    vgg_layers += conv4x4(256, 512)
    vgg_layers += conv4x4(512, 512)

    return vgg_layers


class VGG(nn.Module):
    def __init__(self, num_classes=100, init_weight=False):
        super(VGG, self).__init__()

        self._vgg = build_base_vgg()
        self.pooling = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = FullyConnected(512 * 7 * 7, 4096, num_classes)
        if init_weight: self.init_weights()

    def forward(self, x):
        feats = self._vgg(x)
        pool = self.pooling(feats)
        out = self.classifier(out.view(out.size(0), -1))
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


if __name__ == '__main__':
    model = VGG()
    pass # TODO: Example of data loading
