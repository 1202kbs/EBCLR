from libs.utils import log_softmax, load_config, load_data
from libs.net import init_enc_proj
from libs.models import LIN

from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np

import itertools
import torch

# Use this function to get linear evaluation accuracy
def eval_acc(ckpt_dir, config_dir, eval_config, data_config, device):
    
    config = load_config(config_dir)

    train_X, train_y, test_X, test_y, n_classes = load_data(data_config)

    if eval_config['standardize']:
        nc = train_X.shape[1]
        mean = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda(device)
        std = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda(device)
    else:
        mean = 0.5
        std = 0.5

    enc, _, _ = init_enc_proj(config['net'], device)
    enc.load_state_dict(torch.load(ckpt_dir)['enc_state_dict'])
    net = nn.Sequential(Preprocess(mean, std), enc)
    net.eval()

    lin = LIN(512, n_classes).cuda(device)
    
    with torch.cuda.device(device):
        train(lin, net, train_X, train_y, test_X, test_y, n_classes, eval_config)

# Use this function to get linear transfer accuracy
def transfer_acc(ckpt_dir, config_dir, eval_config, source_data_config, target_data_config, device):

    config = load_config(config_dir)

    if eval_config['standardize']:
        train_X, train_y, test_X, test_y, n_classes = load_data(source_data_config)
        nc = train_X.shape[1]
        mean = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda(device)
        std = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda(device)
    else:
        mean = 0.5
        std = 0.5

    train_X, train_y, test_X, test_y, n_classes = load_data(target_data_config)

    enc, _, _ = init_enc_proj(config['net'], device)
    enc.load_state_dict(torch.load(ckpt_dir)['enc_state_dict'])
    net = nn.Sequential(Preprocess(mean, std), enc)
    net.eval()

    lin = LIN(512, n_classes).cuda(device)
    
    with torch.cuda.device(device):
        train(lin, net, train_X, train_y, test_X, test_y, n_classes, eval_config)

# Use this function to get knn evaluation accuracy
def eval_knn_acc(ckpt_dir, config_dir, eval_config, data_config, device):
    config = load_config(config_dir)
    train_X, train_y, test_X, test_y, n_classes = load_data(data_config)
    
    if eval_config['standardize']:
        nc = train_X.shape[1]
        mean = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda(device)
        std = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda(device)
    else:
        mean = 0.5
        std = 0.5

    enc, _, _ = init_enc_proj(config['net'], device)
    enc.load_state_dict(torch.load(ckpt_dir)['enc_state_dict'])
    net = nn.Sequential(Preprocess(mean, std), enc)
    net.eval()
    
    with torch.cuda.device(device):
        test_n = test_X.shape[0]
        train_Z, test_Z = embed_dataset(net, train_X, test_X)
        
        acc = 0
        N = int(np.ceil(test_n / eval_config['bs']))
        for i in tqdm(range(N)):
            Z = test_Z[i * eval_config['bs']:(i + 1) * eval_config['bs']].cuda()
            y = test_y[i * eval_config['bs']:(i + 1) * eval_config['bs']].cuda()
            
            D = (Z[:,None] - train_Z[None].cuda()).norm(dim=2).cpu()
            w, inds = D.topk(eval_config['K'], dim=1, largest=False)
            
            v = train_y[inds]
            a = v.reshape(-1)
            a = F.one_hot(a, num_classes=n_classes)
            a = a.reshape(v.shape[0],v.shape[1],n_classes)
            weight_pred = a / w[...,None]
            weight_pred = weight_pred.sum(dim=1)
            wp = weight_pred.argmax(dim=1)
            acc += (wp.cuda() == y).sum()

        print(acc / ((i + 1) * eval_config['bs']))

# Use this function to get knn transfer accuracy
def eval_knn_transfer_acc(ckpt_dir, config_dir, eval_config, source_data_config, target_data_config, device):
    config = load_config(config_dir)
    
    if eval_config['standardize']:
        train_X, train_y, test_X, test_y, n_classes = load_data(source_data_config)
        nc = train_X.shape[1]
        mean = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda(device)
        std = train_X.transpose(1,0).flatten(start_dim=1).mean(dim=1).reshape(1,nc,1,1).cuda(device)
    else:
        mean = 0.5
        std = 0.5
    
    train_X, train_y, test_X, test_y, n_classes = load_data(target_data_config)
    
    enc, _, _ = init_enc_proj(config['net'], device)
    enc.load_state_dict(torch.load(ckpt_dir)['enc_state_dict'])
    net = nn.Sequential(Preprocess(mean, std), enc)
    net.eval()
    
    with torch.cuda.device(device):
        test_n = test_X.shape[0]
        train_Z, test_Z = embed_dataset(net, train_X, test_X)
        
        acc = 0
        N = int(np.ceil(test_n / eval_config['bs']))
        for i in tqdm(range(N)):
            Z = test_Z[i * eval_config['bs']:(i + 1) * eval_config['bs']].cuda()
            y = test_y[i * eval_config['bs']:(i + 1) * eval_config['bs']].cuda()
            
            D = (Z[:,None] - train_Z[None].cuda()).norm(dim=2).cpu()
            w, inds = D.topk(eval_config['K'], dim=1, largest=False)
            
            v = train_y[inds]
            a = v.reshape(-1)
            a = F.one_hot(a, num_classes=n_classes)
            a = a.reshape(v.shape[0],v.shape[1],n_classes)
            weight_pred = a / w[...,None]
            weight_pred = weight_pred.sum(dim=1)
            wp = weight_pred.argmax(dim=1)
            acc += (wp.cuda() == y).sum()

        print(acc / ((i + 1) * eval_config['bs']))

def train(lin, net, train_X, train_y, test_X, test_y, n_classes, eval_config):
    
    train_n = train_X.shape[0]
    train_Z, test_Z = embed_dataset(net, train_X, test_X)
    opt = optim.Adam(lin.parameters(), lr=eval_config['lr'])
    
    for epoch in range(eval_config['epochs']):
        for i in range(int(np.ceil(train_n / eval_config['bs']))):
            
            Z = train_Z[i * eval_config['bs']:(i + 1) * eval_config['bs']].cuda()
            y = train_y[i * eval_config['bs']:(i + 1) * eval_config['bs']].cuda()
            
            log_p = lin(Z)
            
            loss = -(F.one_hot(y, num_classes=n_classes) * log_softmax(log_p)).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if (epoch + 1) % eval_config['p_epoch'] == 0:
            print('Epoch {} | Accuracy : {:.3f}'.format(epoch + 1, accuracy(lin, test_Z, test_y)))

class Preprocess(nn.Module):
    def __init__(self, mean, std):
        super(Preprocess, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensors):
        return (tensors - self.mean) / self.std

def embed_dataset(net, train_X, test_X, bs=64):
    
    train_n = train_X.shape[0]
    test_n = test_X.shape[0]
    nc = train_X.shape[1]
    
    train_Z = []
    for i in tqdm(range(int(np.ceil(train_n / bs)))):
        batch = train_X[i * bs:(i + 1) * bs].cuda()
        train_Z.append(net(batch).detach().cpu())
    train_Z = torch.cat(train_Z, dim=0)
    
    test_Z = []
    for i in tqdm(range(int(np.ceil(test_n / bs)))):
        batch = test_X[i * bs:(i + 1) * bs].cuda()
        test_Z.append(net(batch).detach().cpu())
    test_Z = torch.cat(test_Z, dim=0)
    
    return train_Z, test_Z

def accuracy(clf, test_Z, test_y, bs=500):
    
    test_n = test_Z.shape[0]

    acc = 0
    for i in range(int(np.ceil(test_n / bs))):
        Z = test_Z[i * bs:(i + 1) * bs].cuda()
        y = test_y[i * bs:(i + 1) * bs].cuda()
        log_p = clf(Z)
        acc += (torch.argmax(log_p, dim=1) == y).sum() / test_n
    
    return acc.item()