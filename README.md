# resnet-ablation
Ablation analysis of deep residual networks robustness to layer perturbation


# We start by training Resnet34 and VGG19 models
VGG19 is a serial network while Resnet 34 contains Residual blocks - adding "skip connections" within the base serial architecture. The advantage of this is avoiding accuracy loss due to overfitting over more layers, and intractability in backpropagating error to all parameters (vanishing gradient). The serial network has a difficult time learning identity mappings between layers (due to the non-linearity present at each activation). 

Skip connections have been found to occur in the brain, as noted in http://vpnl.stanford.edu/papers/GrillSpectorKayWeiner-DeepLearningChaper2017.pdf . "...most current deep CNNs such as AlexNet are strictly serial, which does not respect the biological reality that the ventral face network contains bypass routes that skip stages of the processing
hierarchy. Therefore, incorporating recent empirical findings of bypass routes into the architectures of CNNs may (1) make CNNs more accurate models of the human visual system and (2) could advance our understanding regarding hypothesized benefits of specific architecture features such as bypass routes." We try to examine this hypothesis by introducing random layer perturbations in the VGG and Resnet Architectures and map the effect on inference results.

Resnet Training Loss                                                                     
<img width="400" height="250" alt="resnet_loss" src="https://user-images.githubusercontent.com/18583460/59377186-8f006a00-8d06-11e9-9f91-268dbc22a262.png"> 

VGG19 Training Loss
<img width="400" alt="cifar_loss" src="https://user-images.githubusercontent.com/18583460/59377966-23b79780-8d08-11e9-901b-2feaea2c7cda.png">

VGG19 Train/Val Accuracy
<img width="400" alt="cifar_acc_vgg" src="https://user-images.githubusercontent.com/18583460/59378036-45b11a00-8d08-11e9-899a-410b61fd82a6.png">
