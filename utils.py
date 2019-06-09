import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


imagenet_pretrained = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'

def load_model(model, weight_file=imagenet_pretrained):
    state_dict = load_state_dict_from_url(weight_file, progress=True)
    model.load_state_dict(state_dict)

    return model
