# Resnet34 model for ablation analysis with random layer perturbation
# Author: Samir Sen, 2019

import torch
import torch.nn as nn



def build_base_resnet():
    pass


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
