from libs.utils import log_softmax, load_data, get_t, get_lr_schedule, get_optimizer, load_config
from libs.net import init_enc_proj

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

def save_ckpt(enc, proj, opt, iteration, runtime, ckpt_path):
    
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
    
    torch.save(checkpoint, ckpt_path)

def load_ckpt(enc, proj, opt, ckpt_path):
    ckpt = torch.load(ckpt_path)

    if isinstance(enc, nn.DataParallel):
        enc.module.load_state_dict(ckpt['enc_state_dict'])
    else:
        enc.load_state_dict(ckpt['enc_state_dict'])

    if isinstance(proj, nn.DataParallel):
        proj.module.load_state_dict(ckpt['proj_state_dict'])
    else:
        proj.load_state_dict(ckpt['proj_state_dict'])

    opt.load_state_dict(ckpt['opt'])
    it = ckpt['iteration']
    rt = ckpt['runtime']
    
    return it, rt

class SimCLR_Loss(nn.Module):

    def __init__(self, temperature):
        super(SimCLR_Loss, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        batch_size = z_i.shape[0]

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)
        
        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

def SimCLR(train_X, enc, proj, config, log_dir, start_epoch):
    
    train_n = train_X.shape[0]
    size = train_X.shape[2]
    nc = train_X.shape[1]

    mean = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()
    std = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()
    
    simclr_loss = SimCLR_Loss(config['temperature'])
    
    params = itertools.chain(enc.parameters(), proj.parameters())
    net = lambda x : proj(enc((x - mean) / std))

    t = get_t(size, config['t'])
    lr_schedule = get_lr_schedule(config)
    opt = get_optimizer(config['optim'], params)
    
    if start_epoch == 0:
        it, rt = 0, 0
    else: 
        it, rt = load_ckpt(enc, proj, opt, log_dir + '/{}.pt'.format(start_epoch))

    enc.train()
    proj.train()

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

        loss = simclr_loss(net(X_v1), net(X_v2))

        # Update
        opt.param_groups[0]['lr'] = lr_schedule(it)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        e = time.time()
        rt += (e - s)

        if it % config['p_iter'] == 0:
            save_ckpt(enc, proj, opt, it, rt, log_dir + '/curr.pt')
            print('Epoch : {:.3f} | Loss : {:.3f} | LR : {:.3e} | Time : {:.3f}'.format(it / epoch_n_iter, loss.item(), lr_schedule(it), rt))
        
        if it % (epoch_n_iter * config['s_epoch']) == 0:
            save_ckpt(enc, proj, opt, it, rt, log_dir + '/{}.pt'.format(it // epoch_n_iter))

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

    enc, proj, _ = init_enc_proj(config['net'], device)

    print('Running SimCLR from epoch {}'.format(start_epoch))

    with torch.cuda.device(device):
        SimCLR(train_X, enc, proj, config, log_dir, start_epoch)

    print('Finished!')