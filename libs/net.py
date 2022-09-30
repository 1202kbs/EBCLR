from libs.resnets import ResNet18, ResNet34, ResNet50
from libs.models import MLP

import torch.nn as nn

def get_act(net_config):

    if net_config['act'] == 'relu':
        return nn.ReLU()
    elif net_config['act'] == 'lrelu':
        return nn.LeakyReLU(0.2)
    else:
        raise NotImplementedError

def init_enc(net_config, device):

    if net_config['encoder'] == 'resnet18':
        enc = ResNet18(nc=net_config['nc'], use_bn=net_config['use_bn'], use_sn=net_config['use_sn'], use_wn=net_config['use_wn'], act=get_act(net_config)).cuda(device)
        feature_dim = 512
    elif net_config['encoder'] == 'resnet34':
        enc = ResNet34(nc=net_config['nc'], use_bn=net_config['use_bn'], use_sn=net_config['use_sn'], use_wn=net_config['use_wn'], act=get_act(net_config)).cuda(device)
        feature_dim = 512
    elif net_config['encoder'] == 'resnet50':
        enc = ResNet50(nc=net_config['nc'], use_bn=net_config['use_bn'], use_sn=net_config['use_sn'], use_wn=net_config['use_wn'], act=get_act(net_config)).cuda(device)
        feature_dim = 2048
    else:
        raise NotImplementedError

    return enc, feature_dim

def init_enc_proj(net_config, device):
    
    if net_config['encoder'] == 'resnet18':
        enc = ResNet18(nc=net_config['nc'], use_bn=net_config['use_bn'], use_sn=net_config['use_sn'], use_wn=net_config['use_wn'], act=get_act(net_config)).cuda(device)
        feature_dim = 512
    elif net_config['encoder'] == 'resnet34':
        enc = ResNet34(nc=net_config['nc'], use_bn=net_config['use_bn'], use_sn=net_config['use_sn'], use_wn=net_config['use_wn'], act=get_act(net_config)).cuda(device)
        feature_dim = 512
    elif net_config['encoder'] == 'resnet50':
        enc = ResNet50(nc=net_config['nc'], use_bn=net_config['use_bn'], use_sn=net_config['use_sn'], use_wn=net_config['use_wn'], act=get_act(net_config)).cuda(device)
        feature_dim = 2048
    else:
        raise NotImplementedError
    
    if 'proj_layers' in net_config:
        proj = MLP(feature_dim, net_config['proj_dim'], n_layers=net_config['proj_layers'], act=get_act(net_config)).cuda(device)
    else:
        proj = MLP(feature_dim, net_config['proj_dim'], n_layers=2, act=get_act(net_config)).cuda(device)
    
    return enc, proj, feature_dim