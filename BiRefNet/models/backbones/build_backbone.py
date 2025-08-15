import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg16, vgg16_bn, VGG16_Weights, VGG16_BN_Weights, resnet50, ResNet50_Weights
from models.backbones.pvt_v2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b5
from models.backbones.swin_v1 import swin_v1_t, swin_v1_s, swin_v1_b, swin_v1_l
from config import Config


config = Config()

def build_backbone(bb_name, pretrained=True, params_settings=''):
    if bb_name == 'vgg16':
        bb_net = list(vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None).children())[0]
        bb = nn.Sequential(OrderedDict({'conv1': bb_net[:10], 'conv2': bb_net[10:17], 'conv3': bb_net[17:24], 'conv4': bb_net[24:31]}))
    elif bb_name == 'vgg16bn':
        bb_net = list(vgg16_bn(weights=VGG16_BN_Weights.DEFAULT if pretrained else None).children())[0]
        bb = nn.Sequential(OrderedDict({'conv1': bb_net[:14], 'conv2': bb_net[14:24], 'conv3': bb_net[24:34], 'conv4': bb_net[34:44]}))
    elif bb_name == 'resnet50':
        bb_net = list(resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None).children())
        bb = nn.Sequential(OrderedDict({'conv1': nn.Sequential(*bb_net[0:4], bb_net[4]), 'conv2': bb_net[5], 'conv3': bb_net[6], 'conv4': bb_net[7]}))
    else:
        bb = eval('{}({})'.format(bb_name, params_settings))
        if pretrained:
            bb = load_weights(bb, bb_name)
    return bb

def load_weights(model, model_name):
    save_model = torch.load(config.weights[model_name], map_location='cpu', weights_only=True)
    
    model_dict = model.state_dict()
    # print(f"Loaded keys: {list(save_model.keys())[:5]} ...")  # preview first few keys
    # print(f"Expected keys (from model): {list(model.state_dict().keys())[:5]} ...")

    # Try flat loading
    state_dict = {k: v for k, v in save_model.items() if k in model_dict and v.size() == model_dict[k].size()}

    # If no match found, try nested structure (e.g., 'state_dict', 'model', etc.)
    if not state_dict:
        for key in save_model:
            if isinstance(save_model[key], dict):
                nested_dict = save_model[key]
                state_dict = {k: v for k, v in nested_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                if state_dict:
                    print(f'Found matching weights under "{key}"')
                    break

    if not state_dict:
        print('❌ No matching weights found. Please check weight file structure.')
        return model
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model
