# Investigating random perturbation effects in serial and residual networks
# What we find is that serial networks performance fluctuates with more
# volatility in comparison with residual networks with skip connections.
#
# TODO: Add visualization for final layer activations.
#
# Author: Samir Sen, 2019

import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
from utils import *
from model.vgg19 import VGG, FullyConnected
from model.resnet34 import Resnet

import matplotlib.pyplot as plt


save_path = 'checkpoints/'
figures = 'figures/'
use_cuda = torch.cuda.is_available()

best_acc = 0
num_classes = 100
num_epochs = 300

# vgg = VGG()
vgg = models.vgg19(pretrained=True)
vgg_model = load_model2(vgg)
if use_cuda: vgg_model = vgg_model.cuda()

print('   Total params: %.2fM' % (num_params(vgg)))

for i, param in vgg_model.named_parameters():
    param.requires_grad = False

vgg_model.classifier[6] = nn.Sequential(
                      nn.Linear(512 * 7 * 7, 256),
                      nn.ReLU(),
                      nn.Linear(256, num_classes),
                      nn.LogSoftmax(dim=1))

vgg_model = model.to('cuda')

print_params(vgg_model)
trainingset, trainloader = load_data()

criterion = nn.CrossEntropyLoss()
trainable_params = list(filter(lambda p: p.requires_grad, vgg_model.parameters()))
optimizer = optim.SGD(trainable_params, lr=0.001, momentum=0.9)

iter = 0
train_loss = []
train_acc = []
train_acc_epoch = []
for epoch in range(num_epochs):
    for data, targets in trainloader:

        out = vgg_model(data)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _, pred = torch.max(out.data, 1)
        total_train += targets.size(0)
        correct_train += pred.eq(targets.data).sum().item()
        train_accuracy = 100 * correct_train / total_train

        iter += 1
        if iter % 10:
            train_loss.append(loss)
            train_acc.append(train_accuracy)

        if iter % 100:
            torch.save(vgg_model, save_path)
            print('Epoch {}, train Loss: {:.3f}'.format(epoch, loss.item()), "Training Accuracy: %d %%" % (train_accuracy))

    train_acc_epoch.append(train_accuracy)


plt.plot(train_loss)
plt.savefig(save_path + figures)
