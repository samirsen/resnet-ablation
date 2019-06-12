# resnet-ablation
Ablation analysis of deep residual networks robustness to layer perturbation


# We start by training Resnet34 and VGG19 models
VGG19 is a serial network while Resnet 34 contains Residual blocks - adding "skip connections" within the base serial architecture. The advantage of this is avoiding accuracy loss due to overfitting over more layers, and intractability in backpropagating error to all parameters (vanishing gradient). The serial network has a difficult time learning identity mappings between layers (due to the non-linearity present at each activation). 

Skip connections have been found to occur in the brain, as noted in http://vpnl.stanford.edu/papers/GrillSpectorKayWeiner-DeepLearningChaper2017.pdf . "...most current deep CNNs such as AlexNet are strictly serial, which does not respect the biological reality that the ventral face network contains bypass routes that skip stages of the processing
hierarchy. Therefore, incorporating recent empirical findings of bypass routes into the architectures of CNNs may (1) make CNNs more accurate models of the human visual system and (2) could advance our understanding regarding hypothesized benefits of specific architecture features such as bypass routes." We try to examine this hypothesis by introducing random layer perturbations in the VGG and Resnet Architectures and map the effect on inference results.

Resnet Training Loss                                                                     
<img width="300" height="200" alt="resnet_loss" src="https://user-images.githubusercontent.com/18583460/59377186-8f006a00-8d06-11e9-9f91-268dbc22a262.png"> 

VGG19 Training Loss

<img width="300" height="200" alt="cifar_loss" src="https://user-images.githubusercontent.com/18583460/59377966-23b79780-8d08-11e9-901b-2feaea2c7cda.png">

VGG19 Train/Val Accuracy

<img width="300" height="200" alt="cifar_acc_vgg" src="https://user-images.githubusercontent.com/18583460/59378036-45b11a00-8d08-11e9-899a-410b61fd82a6.png">

We trained the models, reimplementing base PyTorch model classes and adding capabilities to perturb the models during inference layer-wise. As a baseline, we found the following train / validation accuracies in the network and saved the best model weights for each model. We see at the onset, the two models fair approximately similarly to each other. 

|  | Train Acc | Test Acc |
| :---         |     :---:      |          ---: |
| VGG19 (Serial)   | 88.72     | 84.48    |
| Resnet34     | 89.22       | 85.13      |

We also visualize the activations of the unperturbed VGG19 and Resnet34 network upon training at the first several layers, which look similar:

VGG19

<img width="200" alt="Screen Shot 2019-06-12 at 12 10 45 PM" src="https://user-images.githubusercontent.com/18583460/59379919-c07c3400-8d0c-11e9-97c5-5da4763d006f.png">

Resnet34

<img width="200" alt="Screen Shot 2019-06-12 at 12 11 08 PM" src="https://user-images.githubusercontent.com/18583460/59379993-f28d9600-8d0c-11e9-9aed-48eba69bed2c.png">

Next, we experimentally perturb each of the networks at inference time and observe the effects on validation accuracy and activations among the last layers. Perturbation was a extension of the model's forward function. At each n Conv layers (experiment with n=1, 2, 3), we introduce Gaussian noise to the layer outputs with probability p. 

N = 1 perturbed layers

|  | Train Acc | Test Acc |
| :---         |     :---:      |          ---: |
| VGG19 (Serial)   | --     | 83.79    |
| Resnet34     | --      | 85.02      |

N = 2 Perturbed layers

|  | Train Acc | Test Acc |
| :---         |     :---:      |          ---: |
| VGG19 (Serial)   | --     | 78.26    |
| Resnet34     | --      | 84.94      |

N = 3 Perturbed Layers

|  | Train Acc | Test Acc |
| :---         |     :---:      |          ---: |
| VGG19 (Serial)   | --     |  71.47  |
| Resnet34     | --      | 82.85      |

What we find is that the residual connections have allowed the Resnet34 architecture to be more robust to noise perturbations in its layers compared wit the fully serial VGG network. In future work, it will be interesting to observe various forms of perturbation (those which are relevant in human visual processing) and the effect on network activations and testing accuracy. 
