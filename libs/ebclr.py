from libs.utils import MSGLD, dist, logsumexp, log_softmax, load_data, get_t, get_optimizer, get_lr_schedule, load_config
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

def save_ckpt(buffer, enc, proj, opt, iteration, runtime, save_buffer, ckpt_path):
    
    checkpoint = {
        'iteration' : iteration,
        'runtime' : runtime,
        'opt' : opt.state_dict()
    }

    if save_buffer:
        checkpoint['buffer'] = buffer.buffer
        checkpoint['counter'] = buffer.counter
    
    if isinstance(enc, nn.DataParallel):
        checkpoint['enc_state_dict'] = enc.module.state_dict()
    else:
        checkpoint['enc_state_dict'] = enc.state_dict()
    
    if isinstance(proj, nn.DataParallel):
        checkpoint['proj_state_dict'] = proj.module.state_dict()
    else:
        checkpoint['proj_state_dict'] = proj.state_dict()
    
    torch.save(checkpoint, ckpt_path)

def load_ckpt(enc, proj, buffer, opt, ckpt_path):
    ckpt = torch.load(ckpt_path)

    if isinstance(enc, nn.DataParallel):
        enc.module.load_state_dict(ckpt['enc_state_dict'])
    else:
        enc.load_state_dict(ckpt['enc_state_dict'])

    if isinstance(proj, nn.DataParallel):
        proj.module.load_state_dict(ckpt['proj_state_dict'])
    else:
        proj.load_state_dict(ckpt['proj_state_dict'])

    buffer.buffer = ckpt['buffer']
    buffer.counter = ckpt['counter']

    opt.load_state_dict(ckpt['opt'])
    
    return ckpt['iteration'], ckpt['runtime']

def transform_data(x, t, bs):
    n_iter = int(np.ceil(x.shape[0] / bs))
    return torch.cat([t(x[j * bs:(j + 1) * bs]) for j in range(n_iter)], dim=0)

def similarity_f(x, y, normalize):

    if normalize:
        x = x / (x.norm(dim=1, keepdim=True) + 1e-10)
        y = y / (y.norm(dim=1, keepdim=True) + 1e-10)

    return -(x[:,None] - y[None]).square().sum(dim=2)

class SimCLR_Loss(nn.Module):

    def __init__(self, normalize, temperature):
        super(SimCLR_Loss, self).__init__()
        self.normalize = normalize
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

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

        sim = similarity_f(z, z, self.normalize) / self.temperature

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

class Buffer:

    def __init__(self, train_X, t, buffer_config):
        self.train_X = train_X
        self.config = buffer_config
        self.t = t

        self.train_n = train_X.shape[0]
        self.size = train_X.shape[2]
        self.nc = train_X.shape[1]

        self.idx = 0

        self.counter = torch.ones(size=[self.config['size'],1,1,1]) * 5.0
        self.init_buffer()

    def __get_sample__(self, n_samples):
        r_idx = torch.randint(self.train_X.shape[0], size=[n_samples])
        samples = self.train_X[r_idx]
        samples = transform_data(samples, self.t, self.config['bs'])
        return samples

    def __get_rand__(self, n_samples):
        if 'CD_ratio' in self.config:
            samples = self.__get_sample__(n_samples)
            return (self.config['CD_ratio'] * samples + (1.0 - self.config['CD_ratio']) * torch.rand_like(samples)) * 2.0 - 1.0
        else:
            return torch.rand(size=[n_samples, self.nc, self.size, self.size]) * 2.0 - 1.0

    def init_buffer(self):
        self.buffer = self.__get_rand__(self.config['size'])

    def sample(self, n_samples):
        self.idx = torch.randint(self.config['size'], size=[n_samples])
        sample = self.buffer[self.idx]
        count = self.counter[self.idx].clone()

        r_idx = torch.randint(n_samples, size=[int(n_samples * self.config['rho'])])
        sample[r_idx] = self.__get_rand__(n_samples)[r_idx]
        count[r_idx] = 0.0

        self.counter[self.idx] = (count + 1.0)

        return sample.cuda(), count.cuda()

    def update(self, samples):
        self.buffer[self.idx] = samples.detach().clone().cpu()

def shuffled_idx(batch_size):
    shuffled_idxs = torch.randperm(batch_size).long().cuda()
    reverse_idxs = torch.zeros(batch_size).long().cuda()
    value = torch.arange(batch_size).long().long().cuda()
    reverse_idxs.index_copy_(0, shuffled_idxs, value)
    return shuffled_idxs, reverse_idxs

def EBCLR(train_X, enc, proj, config, log_dir, start_epoch):
    
    train_n = train_X.shape[0]
    size = train_X.shape[2]
    nc = train_X.shape[1]

    mean = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()
    std = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda()
    
    simclr_loss = SimCLR_Loss(config['normalize'], config['temperature'])
    logits = lambda x, y : similarity_f(x, y, config['normalize']) / config['temperature']
    log_pdf = lambda x, y : logsumexp(logits(x, y))

    params = itertools.chain(enc.parameters(), proj.parameters())
    net = lambda x : proj(enc(x))

    t = get_t(size, config['t'])
    buffer = Buffer(train_X, t, config['buffer'])
    lr_schedule = get_lr_schedule(config)
    opt = get_optimizer(config['optim'], params)
    sgld = MSGLD(config['sgld'])
    
    if start_epoch == 0:
        it, rt = 0, 0
    else: 
        it, rt = load_ckpt(enc, proj, buffer, opt, log_dir + '/{}.pt'.format(start_epoch))
    
    enc.train()
    proj.train()
    
    epoch_n_iter = int(np.ceil(train_n / config['bs']))
    while True:
        if it % epoch_n_iter == 0:
            idx = torch.randperm(train_n)
            V1 = transform_data(train_X, t, config['bs'])[idx]
            V2 = transform_data(train_X, t, config['bs'])[idx]
        
        i = it % epoch_n_iter
        it += 1
        
        s = time.time()

        X_v1 = V1[i * config['bs']:(i + 1) * config['bs']].cuda() * 2.0 - 1.0
        X_v2 = V2[i * config['bs']:(i + 1) * config['bs']].cuda() * 2.0 - 1.0

        if config['net']['use_bn']:
            shuffled_idxs, reverse_idxs = shuffled_idx(X_v2.shape[0])
            Z_v1, Z_v2 = net(X_v1), net(X_v2[shuffled_idxs])[reverse_idxs]
        else:
            Z_v1, Z_v2 = net(X_v1), net(X_v2)

        if 'neg_bs' in config:
            X_init, count = buffer.sample(config['neg_bs'])
        else:
            X_init, count = buffer.sample(X_v1.shape[0])
        
        X_n = sgld(lambda x : log_pdf(net(x), Z_v2.detach().clone()), X_init, count)
        buffer.update(X_n)
        Z_n = net(X_n)

        log_pdf_d = log_pdf(Z_v1, Z_v2)
        log_pdf_n = log_pdf(Z_n, Z_v2)
        gen_loss = -log_pdf_d.mean() + log_pdf_n.mean() + config['lmda1'] * (Z_v1.square().sum() + Z_n.square().sum())

        loss = config['lmda2'] * gen_loss + simclr_loss(Z_v1, Z_v2)

        # Update
        opt.param_groups[0]['lr'] = lr_schedule(it)
        opt.zero_grad()
        loss.backward()

        if config['optim']['optimizer'] == 'adam':
            for i, param in enumerate(params):
                std = opt.state_dict()['state'][i]['exp_avg_sq'].sqrt()
                param.grad = clamp(param.grad, - 3.0 * std, 3.0 * std)
        
        opt.step()
        
        e = time.time()
        rt += (e - s)

        if it % config['p_iter'] == 0:
            save_ckpt(buffer, enc, proj, opt, it, rt, False, log_dir + '/curr.pt')

            pdf_d = torch.exp(log_pdf_d).mean().item()
            pdf_n = torch.exp(log_pdf_n).mean().item()

            print('Epoch : {:.3f} | Data PDF : {:.3e} | Noise PDF : {:.3e} | LR : {:.3e} | Time : {:.3f}'.format(it / epoch_n_iter, pdf_d, pdf_n, lr_schedule(it), rt))
        
        if it % (epoch_n_iter * config['s_epoch']) == 0:
            save_ckpt(buffer, enc, proj, opt, it, rt, config['save_buffer'], log_dir + '/{}.pt'.format(it // epoch_n_iter))

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

    print('Starting EBCLR from epoch {}'.format(start_epoch))

    with torch.cuda.device(device):
        flag = EBCLR(train_X, enc, proj, config, log_dir, start_epoch)

    print('Finished!')