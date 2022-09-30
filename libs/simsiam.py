from libs.utils import log_softmax, load_data, get_t, get_lr_schedule, get_optimizer, load_config
from libs.net import init_enc

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np

import torchvision
import itertools
import torch
import time
import os

def save_ckpt(enc, proj, pred, opt, iteration, runtime, ckpt_path):
    
    checkpoint = {
        'iteration' : iteration,
        'runtime' : runtime,
        'opt' : opt.state_dict()
    }
    
    if isinstance(enc, nn.DataParallel):
        checkpoint['enc_state_dict'] = enc.module.state_dict()
    else:
        checkpoint['enc_state_dict'] = enc.state_dict()
    
    if isinstance(proj, nn.DataParallel):
        checkpoint['proj_state_dict'] = proj.module.state_dict()
    else:
        checkpoint['proj_state_dict'] = proj.state_dict()

    if isinstance(pred, nn.DataParallel):
        checkpoint['pred_state_dict'] = pred.module.state_dict()
    else:
        checkpoint['pred_state_dict'] = pred.state_dict()
    
    torch.save(checkpoint, ckpt_path)

def load_ckpt(enc, proj, pred, opt, ckpt_path):
    ckpt = torch.load(ckpt_path)

    if isinstance(enc, nn.DataParallel):
        enc.module.load_state_dict(ckpt['enc_state_dict'])
    else:
        enc.load_state_dict(ckpt['enc_state_dict'])

    if isinstance(proj, nn.DataParallel):
        proj.module.load_state_dict(ckpt['proj_state_dict'])
    else:
        proj.load_state_dict(ckpt['proj_state_dict'])

    if isinstance(pred, nn.DataParallel):
        pred.module.load_state_dict(ckpt['pred_state_dict'])
    else:
        pred.load_state_dict(ckpt['pred_state_dict'])

    opt.load_state_dict(ckpt['opt'])
    it = ckpt['iteration']
    rt = ckpt['runtime']
    
    return it, rt

def D(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

class Preprocess(nn.Module):
    def __init__(self, mean, std):
        super(Preprocess, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensors):
        return (tensors - self.mean) / self.std

class projection_MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

def SimSiam(train_X, enc, proj, pred, config, log_dir, start_epoch):
    
    train_n = train_X.shape[0]
    size = train_X.shape[2]
    nc = train_X.shape[1]

    mean = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()
    std = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()

    params = itertools.chain(enc.parameters(), proj.parameters(), pred.parameters())
    net = nn.Sequential(Preprocess(mean, std), enc, proj)

    t = get_t(size, config['t'])
    lr_schedule = get_lr_schedule(config)
    opt = get_optimizer(config['optim'], params)
    
    if start_epoch == 0:
        it, rt = 0, 0
    else: 
        it, rt = load_ckpt(enc, proj, pred, opt, log_dir + '/{}.pt'.format(start_epoch))
    
    enc.train()
    proj.train()
    pred.train()

    epoch_n_iter = int(np.ceil(train_n / config['bs']))
    while True:
        if it % epoch_n_iter == 0:
            train_X = train_X[torch.randperm(train_n)]

        i = it % epoch_n_iter
        it += 1
        
        s = time.time()
        
        X = train_X[i * config['bs']:(i + 1) * config['bs']]

        X_v1 = t(X).cuda()
        X_v2 = t(X).cuda()

        Z_v1, Z_v2 = net(X_v1), net(X_v2)
        P_v1, P_v2 = pred(Z_v1), pred(Z_v2)

        loss = D(P_v1, Z_v2) / 2 + D(P_v2, Z_v1) / 2

        # Update
        opt.param_groups[0]['lr'] = lr_schedule(it)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        e = time.time()
        rt += (e - s)

        if it % config['p_iter'] == 0:
            save_ckpt(enc, proj, pred, opt, it, rt, log_dir + '/curr.pt')
            print('Epoch : {:.3f} | Loss : {:.3f} | LR : {:.3e} | Time : {:.3f}'.format(it / epoch_n_iter, loss.item(), lr_schedule(it), rt))
        
        if it % (epoch_n_iter * config['s_epoch']) == 0:
            save_ckpt(enc, proj, pred, opt, it, rt, log_dir + '/{}.pt'.format(it // epoch_n_iter))

        if it >= config['its']:
            break

def run(log_dir, config_dir, start_epoch, device):

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    config = load_config(config_dir)
    
    train_X, train_y, test_X, test_y, n_classes = load_data(config['data'])

    epoch_n_iter = int(np.ceil(train_X.shape[0] / config['bs']))

    if 'epochs' in config:
        config['its'] = epoch_n_iter * config['epochs']

    if 'p_epoch' in config:
        config['p_iter'] = epoch_n_iter * config['p_epoch']

    enc, feature_dim = init_enc(config['net'], device)
    proj = projection_MLP(feature_dim).cuda(device)
    pred = prediction_MLP().cuda(device)

    print('Running SimSiam from epoch {}'.format(start_epoch))

    with torch.cuda.device(device):
        SimSiam(train_X, enc, proj, pred, config, log_dir, start_epoch)

    print('Finished!')