import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import torch

def Linear(in_features, out_features, bias=True, use_bn=False, use_sn=False, use_wn=False):

    layer = nn.Linear(in_features, out_features, bias=bias)

    if use_bn:
        layer = nn.Sequential(layer, nn.BatchNorm1d(out_features))

    if use_sn:
        layer = nn.utils.spectral_norm(layer)

    if use_wn:
        layer = nn.utils.weight_norm(layer)

    return layer

class LIN(nn.Module):
    
    def __init__(self, in_dim, out_dim, use_sn=False):
        super(LIN, self).__init__()
        
        self.fc1 = Linear(in_dim, out_dim, use_sn=use_sn)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

class MLP(nn.Module):
    
    def __init__(self, in_dim, out_dim, n_layers=2, use_bn=False, use_sn=False, use_wn=False, act=nn.LeakyReLU(0.2)):
        super(MLP, self).__init__()
        self.act = act

        self.fc1 = Linear(in_dim, in_dim, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn)

        # fcs = []
        # for _ in range(n_layers - 1):
        #     fcs.append(Linear(in_dim, in_dim, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn))
        #     fcs.append(self.act)
        # self.fcs = nn.Sequential(*fcs)

        self.fc2 = Linear(in_dim, out_dim, use_sn=use_sn, use_wn=use_wn)
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x