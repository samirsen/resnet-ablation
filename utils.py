import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

pretrained_weights = {
    'vgg': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'resnet': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
}

def load_model(model, arch='vgg'):
    state_dict = load_state_dict_from_url(pretrained_weights[arch], progress=True)
    model.load_state_dict(state_dict)

    return model

def load_data():
    pass 
