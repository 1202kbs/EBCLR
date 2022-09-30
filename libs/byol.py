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

def save_ckpt(enc_o, proj_o, pred_o, enc_t, proj_t, opt, iteration, runtime, save_all, ckpt_path):
    
    checkpoint = {
        'iteration' : iteration,
        'runtime' : runtime,
        'opt' : opt.state_dict()
    }

    if save_all:

        if isinstance(enc_t, nn.DataParallel):
            checkpoint['enc_t_state_dict'] = enc_t.module.state_dict()
        else:
            checkpoint['enc_t_state_dict'] = enc_t.state_dict()
        
        if isinstance(proj_t, nn.DataParallel):
            checkpoint['proj_t_state_dict'] = proj_t.module.state_dict()
        else:
            checkpoint['proj_t_state_dict'] = proj_t.state_dict()
    
    if isinstance(enc_o, nn.DataParallel):
        checkpoint['enc_state_dict'] = enc_o.module.state_dict()
    else:
        checkpoint['enc_state_dict'] = enc_o.state_dict()
    
    if isinstance(proj_o, nn.DataParallel):
        checkpoint['proj_state_dict'] = proj_o.module.state_dict()
    else:
        checkpoint['proj_state_dict'] = proj_o.state_dict()

    if isinstance(pred_o, nn.DataParallel):
        checkpoint['pred_state_dict'] = pred_o.module.state_dict()
    else:
        checkpoint['pred_state_dict'] = pred_o.state_dict()
    
    torch.save(checkpoint, ckpt_path)

def load_ckpt(enc_o, proj_o, pred_o, enc_t, proj_t, opt, ckpt_path):
    ckpt = torch.load(ckpt_path)

    if isinstance(enc_o, nn.DataParallel):
        enc_o.module.load_state_dict(ckpt['enc_state_dict'])
    else:
        enc_o.load_state_dict(ckpt['enc_state_dict'])

    if isinstance(proj_o, nn.DataParallel):
        proj_o.module.load_state_dict(ckpt['proj_state_dict'])
    else:
        proj_o.load_state_dict(ckpt['proj_state_dict'])

    if isinstance(pred_o, nn.DataParallel):
        pred_o.module.load_state_dict(ckpt['pred_state_dict'])
    else:
        pred_o.load_state_dict(ckpt['pred_state_dict'])

    if isinstance(enc_t, nn.DataParallel):
        enc_t.module.load_state_dict(ckpt['enc_t_state_dict'])
    else:
        enc_t.load_state_dict(ckpt['enc_t_state_dict'])

    if isinstance(proj_t, nn.DataParallel):
        proj_t.module.load_state_dict(ckpt['proj_t_state_dict'])
    else:
        proj_t.load_state_dict(ckpt['proj_t_state_dict'])

    opt.load_state_dict(ckpt['opt'])
    it = ckpt['iteration']
    rt = ckpt['runtime']
    
    return it, rt

def D(x, y):
    return (2 - 2 * F.cosine_similarity(x, y.detach(), dim=-1)).mean()

class Preprocess(nn.Module):
    def __init__(self, mean, std):
        super(Preprocess, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensors):
        return (tensors - self.mean) / self.std

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
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

def BYOL(train_X, enc_o, proj_o, pred_o, enc_t, proj_t, config, log_dir, start_epoch):
    
    train_n = train_X.shape[0]
    size = train_X.shape[2]
    nc = train_X.shape[1]

    mean = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()
    std = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()
    
    params = itertools.chain(enc_o.parameters(), proj_o.parameters(), pred_o.parameters())
    net_o = nn.Sequential(Preprocess(mean, std), enc_o, proj_o)
    net_t = nn.Sequential(Preprocess(mean, std), enc_t, proj_t)

    t = get_t(size, config['t'])
    lr_schedule = get_lr_schedule(config)
    opt = get_optimizer(config['optim'], params)
    
    if start_epoch == 0:
        it, rt = 0, 0
    else: 
        it, rt = load_ckpt(enc_o, proj_o, pred_o, enc_t, proj_t, opt, log_dir + '/{}.pt'.format(start_epoch))

    enc_o.train()
    proj_o.train()
    pred_o.train()

    enc_t.train()
    proj_t.train()

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

        Z_v1_o, Z_v2_o = net_o(X_v1), net_o(X_v2)
        P_v1_o, P_v2_o = pred_o(Z_v1_o), pred_o(Z_v2_o)

        with torch.no_grad():
            Z_v1_t, Z_v2_t = net_t(X_v1), net_t(X_v2)

        loss = D(P_v1_o, Z_v2_t) / 2 + D(P_v2_o, Z_v1_t) / 2

        # Update
        opt.param_groups[0]['lr'] = lr_schedule(it)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Online network update
        with torch.no_grad():

            for p_o, p_t in zip(enc_o.parameters(), enc_t.parameters()):
                p_t.data = p_t.data * config['momentum'] + p_o.detach().data * (1 - config['momentum'])

            for p_o, p_t in zip(proj_o.parameters(), proj_t.parameters()):
                p_t.data = p_t.data * config['momentum'] + p_o.detach().data * (1 - config['momentum'])
        
        e = time.time()
        rt += (e - s)

        if it % config['p_iter'] == 0:
            save_ckpt(enc_o, proj_o, pred_o, enc_t, proj_t, opt, it, rt, False, log_dir + '/curr.pt')
            print('Epoch : {:.3f} | Loss : {:.3f} | LR : {:.3e} | Time : {:.3f}'.format(it / epoch_n_iter, loss.item(), lr_schedule(it), rt))
        
        if it % (epoch_n_iter * config['s_epoch']) == 0:
            save_ckpt(enc_o, proj_o, pred_o, enc_t, proj_t, opt, it, rt, True, log_dir + '/{}.pt'.format(it // epoch_n_iter))

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

    enc_o, feature_dim = init_enc(config['net'], device)
    proj_o = MLP(feature_dim).cuda(device)
    pred_o = MLP(256).cuda(device)

    enc_t, feature_dim = init_enc(config['net'], device)
    proj_t = MLP(feature_dim).cuda(device)

    for p_o, p_t in zip(enc_o.parameters(), enc_t.parameters()):
        p_t.data.copy_(p_t.data)
        p_t.data.requires_grad = False

    for p_o, p_t in zip(proj_o.parameters(), proj_t.parameters()):
        p_t.data.copy_(p_t.data)
        p_t.data.requires_grad = False

    print('Running BYOL from epoch {}'.format(start_epoch))

    with torch.cuda.device(device):
        BYOL(train_X, enc_o, proj_o, pred_o, enc_t, proj_t, config, log_dir, start_epoch)

    print('Finished!')