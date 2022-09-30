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

def save_ckpt(enc_q, proj_q, enc_k, proj_k, queue, opt, iteration, runtime, save_all, ckpt_path):
    
    checkpoint = {
        'iteration' : iteration,
        'runtime' : runtime,
        'opt' : opt.state_dict()
    }

    if save_all:
        checkpoint['queue'] = queue

        if isinstance(enc_k, nn.DataParallel):
            checkpoint['enc_k_state_dict'] = enc_k.module.state_dict()
        else:
            checkpoint['enc_k_state_dict'] = enc_k.state_dict()
        
        if isinstance(proj_k, nn.DataParallel):
            checkpoint['proj_k_state_dict'] = proj_k.module.state_dict()
        else:
            checkpoint['proj_k_state_dict'] = proj_k.state_dict()
    
    if isinstance(enc_q, nn.DataParallel):
        checkpoint['enc_state_dict'] = enc_q.module.state_dict()
    else:
        checkpoint['enc_state_dict'] = enc_q.state_dict()
    
    if isinstance(proj_q, nn.DataParallel):
        checkpoint['proj_state_dict'] = proj_q.module.state_dict()
    else:
        checkpoint['proj_state_dict'] = proj_q.state_dict()
    
    torch.save(checkpoint, ckpt_path)

def load_ckpt(enc_q, proj_q, enc_k, proj_k, queue, opt, ckpt_path):
    ckpt = torch.load(ckpt_path)

    if isinstance(enc_q, nn.DataParallel):
        enc_q.module.load_state_dict(ckpt['enc_state_dict'])
    else:
        enc_q.load_state_dict(ckpt['enc_state_dict'])

    if isinstance(proj_q, nn.DataParallel):
        proj_q.module.load_state_dict(ckpt['proj_state_dict'])
    else:
        proj_q.load_state_dict(ckpt['proj_state_dict'])

    if isinstance(enc_k, nn.DataParallel):
        enc_k.module.load_state_dict(ckpt['enc_k_state_dict'])
    else:
        enc_k.load_state_dict(ckpt['enc_k_state_dict'])

    if isinstance(proj_k, nn.DataParallel):
        proj_k.module.load_state_dict(ckpt['proj_k_state_dict'])
    else:
        proj_k.load_state_dict(ckpt['proj_k_state_dict'])

    queue.data = ckpt['queue'].data

    opt.load_state_dict(ckpt['opt'])
    it = ckpt['iteration']
    rt = ckpt['runtime']
    
    return it, rt

def shuffled_idx(batch_size):
    shuffled_idxs = torch.randperm(batch_size).long().cuda()
    reverse_idxs = torch.zeros(batch_size).long().cuda()
    value = torch.arange(batch_size).long().long().cuda()
    reverse_idxs.index_copy_(0, shuffled_idxs, value)
    return shuffled_idxs, reverse_idxs

def MoCo(train_X, enc_q, proj_q, enc_k, proj_k, config, log_dir, start_epoch):
    
    train_n = train_X.shape[0]
    size = train_X.shape[2]
    nc = train_X.shape[1]

    mean = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()
    std = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    
    queue = torch.randn(config['queue_size'], config['net']['proj_dim']).cuda()
    
    params_q = itertools.chain(enc_q.parameters(), proj_q.parameters())
    net_q = lambda x : proj_q(enc_q((x - mean) / std))

    params_k = itertools.chain(enc_k.parameters(), proj_k.parameters())
    net_k = lambda x : proj_k(enc_k((x - mean) / std))

    t = get_t(size, config['t'])
    lr_schedule = get_lr_schedule(config)
    opt = get_optimizer(config['optim'], params_q)
    
    if start_epoch == 0:
        it, rt = 0, 0
    else: 
        it, rt = load_ckpt(enc_q, proj_q, enc_k, proj_k, queue, opt, log_dir + '/{}.pt'.format(start_epoch))

    enc_q.train()
    proj_q.train()

    enc_k.train()
    proj_k.train()

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

        Z_v1 = F.normalize(net_q(X_v1), dim=1)

        with torch.no_grad():

            for p_q, p_k in zip(enc_q.parameters(), enc_k.parameters()):
                p_k.data = p_k.data * config['momentum'] + p_q.detach().data * (1 - config['momentum'])

            for p_q, p_k in zip(proj_q.parameters(), proj_k.parameters()):
                p_k.data = p_k.data * config['momentum'] + p_q.detach().data * (1 - config['momentum'])

            shuffled_idxs, reverse_idxs = shuffled_idx(X.shape[0])
            X_v2 = X_v2[shuffled_idxs]
            Z_v2 = F.normalize(net_k(X_v2), dim=1)
            Z_v2 = Z_v2[reverse_idxs]

        Z_mem = F.normalize(queue.clone().detach(), dim=1)

        pos = torch.bmm(Z_v1.view(Z_v1.shape[0],1,-1), Z_v2.view(Z_v2.shape[0],-1,1)).squeeze(-1)
        neg = torch.mm(Z_v1, Z_mem.transpose(1,0))

        logits = torch.cat((pos, neg), dim=1) / config['temperature']
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = criterion(logits, labels)

        # Update
        opt.param_groups[0]['lr'] = lr_schedule(it)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Update queue
        queue = queue[Z_v2.shape[0]:]
        queue = torch.cat((queue, Z_v2.detach().clone()), dim=0)
        
        e = time.time()
        rt += (e - s)

        if it % config['p_iter'] == 0:
            save_ckpt(enc_q, proj_q, enc_k, proj_k, queue, opt, it, rt, False, log_dir + '/curr.pt')
            print('Epoch : {:.3f} | Loss : {:.3f} | LR : {:.3e} | Time : {:.3f}'.format(it / epoch_n_iter, loss.item(), lr_schedule(it), rt))
        
        if it % (epoch_n_iter * config['s_epoch']) == 0:
            save_ckpt(enc_q, proj_q, enc_k, proj_k, queue, opt, it, rt, True, log_dir + '/{}.pt'.format(it // epoch_n_iter))

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

    enc_q, proj_q, _ = init_enc_proj(config['net'], device)
    enc_k, proj_k, _ = init_enc_proj(config['net'], device)

    for p_q, p_k in zip(enc_q.parameters(), enc_k.parameters()):
        p_k.data.copy_(p_q.data)
        p_k.data.requires_grad = False

    for p_q, p_k in zip(proj_q.parameters(), proj_k.parameters()):
        p_k.data.copy_(p_q.data)
        p_k.data.requires_grad = False

    print('Running MoCo from epoch {}'.format(start_epoch))

    with torch.cuda.device(device):
        MoCo(train_X, enc_q, proj_q, enc_k, proj_k, config, log_dir, start_epoch)

    print('Finished!')