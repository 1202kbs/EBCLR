from libs.gaussian_blur import GaussianBlur

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np

import torchvision
import torch
import yaml

class MSGLD:

    def __init__(self, config):
        self.config = config

    def __get_std__(self, count):
        return self.config['min_std'] + (self.config['max_std'] - self.config['min_std']) * torch.maximum(1.0 - count / self.config['threshold'], torch.zeros_like(count).cuda())

    def __call__(self, log_pdf, init, count):
        out = init.detach().clone().requires_grad_(True)
        for i in range(self.config['iter']):
            lp = log_pdf(out).sum()
            lp.backward()
            out.data = out + self.config['lr'] * torch.clamp(out.grad, -self.config['tau'], self.config['tau']) + self.__get_std__(count) * torch.randn_like(out)
            out.grad.zero_()
        return out.detach().clone()

def logsumexp(log_p):
    m, _ = torch.max(log_p, dim=1, keepdim=True)
    return torch.log(torch.exp(log_p - m).sum(dim=1, keepdim=True)) + m

def log_softmax(log_p):
    return log_p - logsumexp(log_p)

def softmax(log_p):
    m, _ = torch.max(log_p, dim=1, keepdim=True)
    f = torch.exp(log_p - m)
    return f / f.sum(dim=1, keepdim=True)

def get_t(img_size, t_config):
    
    t = []
    
    if 'crop_scale' in t_config:
        t.append(transforms.RandomResizedCrop(size=img_size, scale=tuple(t_config['crop_scale'].values())))
    
    if 'flip_p' in t_config:
        t.append(transforms.RandomHorizontalFlip(p=t_config['flip_p']))
    
    if 'jitter' in t_config:
        jitter = transforms.ColorJitter(t_config['jitter']['b'], t_config['jitter']['c'], t_config['jitter']['s'], t_config['jitter']['h'])
        t.append(transforms.RandomApply([jitter], p=t_config['jitter_p']))
    
    if 'gray_p' in t_config:
        t.append(transforms.RandomGrayscale(p=t_config['gray_p']))
    
    if 'blur_scale' in t_config:
        blur = GaussianBlur(kernel_size=int(t_config['blur_scale'] * img_size))
        t.append(transforms.RandomApply([blur], p=1.0))
    
    if 'noise_std' in t_config:
        t.append(lambda x : x + torch.randn_like(x) * t_config['noise_std'])
    
    return transforms.Compose(t)

def get_optimizer(optim_config, params):
    if optim_config['optimizer'] == 'sgd':
        return optim.SGD(params, lr=optim_config['init_lr'], momentum=0.9, weight_decay=optim_config['weight_decay'] if ('weight_decay' in optim_config) else 0.0) 
    elif optim_config['optimizer'] == 'adam':
        return optim.Adam(params, lr=optim_config['init_lr'], weight_decay=optim_config['weight_decay'] if ('weight_decay' in optim_config) else 0.0)
    else:
        raise NotImplementedError

def get_lr_schedule(config):
    
    if config['optim']['lr_schedule'] == 'const':
        return lambda it : config['optim']['init_lr']
    elif config['optim']['lr_schedule'] == 'cosine':
        return lambda it : config['optim']['init_lr'] * np.cos(0.5 * np.pi * it / config['its'])
    else:
        raise NotImplementedError

def inner(x, y, normalize=False):
    x = x.flatten(start_dim=1)
    y = y.flatten(start_dim=1)

    if normalize:
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)

    d = (x * y).sum(dim=1, keepdim=True)
    return d

def dist(x, y, normalize=False):
    x = x.flatten(start_dim=1)
    y = y.flatten(start_dim=1)
    
    if normalize:
        x = x / (x.norm(dim=1, keepdim=True) + 1e-10)
        y = y / (y.norm(dim=1, keepdim=True) + 1e-10)

    d = (x - y).square().sum(dim=1, keepdim=True)
    return d

def generate_views(x, t, n_samples):
    views = []
    shape = list(x.shape[1:])
    for _ in range(n_samples):
        views.append(t(x)[:,None])
    views = torch.cat(views, dim=1).reshape([-1] + shape)
    return views

def load_data(data_config, download=False):

    root = data_config['data_dir']
    dataset = data_config['dataset']
    
    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=[transforms.ToTensor()])
        testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=[transforms.ToTensor()])

        train_X, train_y = trainset.data[:,None] / 255, trainset.targets
        test_X, test_y = testset.data[:,None] / 255, testset.targets
        n_classes = 10
    elif dataset == 'fmnist':
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=[transforms.ToTensor()])
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=[transforms.ToTensor()])

        train_X, train_y = trainset.data[:,None] / 255, trainset.targets
        test_X, test_y = testset.data[:,None] / 255, testset.targets
        n_classes = 10
    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=[transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=[transforms.ToTensor()])

        train_X = torch.tensor(trainset.data).permute(0,3,1,2) / 255
        train_y = torch.tensor(trainset.targets, dtype=int)

        test_X = torch.tensor(testset.data).permute(0,3,1,2) / 255
        test_y = torch.tensor(testset.targets, dtype=int)
        n_classes = 10
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=[transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=[transforms.ToTensor()])

        train_X = torch.tensor(trainset.data).permute(0,3,1,2) / 255
        train_y = torch.tensor(trainset.targets, dtype=int)

        test_X = torch.tensor(testset.data).permute(0,3,1,2) / 255
        test_y = torch.tensor(testset.targets, dtype=int)
        n_classes = 100

    train_n = train_X.shape[0]
    idx = torch.randperm(train_n)
    train_X = train_X[idx][:int(train_n * data_config['data_ratio'])]
    train_y = train_y[idx][:int(train_n * data_config['data_ratio'])]
    
    return train_X, train_y, test_X, test_y, n_classes

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config