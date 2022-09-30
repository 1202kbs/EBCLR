import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_bn=False, use_sn=False, use_wn=False, avg_pool=False):

    if avg_pool:
        layer = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False if use_bn else bias)
    else:
        layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False if use_bn else bias)

    if use_bn:
        layer = nn.Sequential(layer, nn.BatchNorm2d(out_channels))

    if use_sn:
        layer = nn.utils.spectral_norm(layer)

    if use_wn:
        layer = nn.utils.weight_norm(layer)

    if avg_pool and stride > 1:
        layer = nn.Sequential(layer, nn.AvgPool2d(stride, stride))

    return layer

def Linear(in_features, out_features, bias=True, use_bn=False, use_sn=False, use_wn=False):

    layer = nn.Linear(in_features, out_features, bias=bias)

    if use_bn:
        layer = nn.Sequential(layer, nn.BatchNorm1d(out_features))

    if use_sn:
        layer = nn.utils.spectral_norm(layer)

    if use_wn:
        layer = nn.utils.weight_norm(layer)

    return layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn=False, use_sn=False, use_wn=False, avg_pool=False, act=nn.LeakyReLU(0.2)):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool)
        self.act = act

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool)
            )

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_bn=False, use_sn=False, use_wn=False, avg_pool=False, act=nn.LeakyReLU(0.2)):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=True, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool)
        self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True, use_sn=use_sn, use_bn=use_bn, use_wn=use_wn, avg_pool=avg_pool)
        self.act = act

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool)
            )

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, nc, block, num_blocks, use_bn=False, use_sn=False, use_wn=False, avg_pool=False, act=nn.LeakyReLU(0.2)):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=True, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool, act=act)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool, act=act)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool, act=act)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use_bn=use_bn, use_sn=use_sn, use_wn=use_wn, avg_pool=avg_pool, act=act)
        self.act = act

    def _make_layer(self, block, planes, num_blocks, stride, use_bn, use_sn, use_wn, avg_pool, act):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn, use_sn, use_wn, avg_pool, act))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return out


def ResNet18(nc=3, use_bn=False, use_sn=False, use_wn=False, avg_pool=False, act=nn.LeakyReLU(0.2)):
    return ResNet(nc, BasicBlock, [2, 2, 2, 2], use_bn, use_sn, use_wn, avg_pool, act)


def ResNet34(nc=3, use_bn=False, use_sn=False, use_wn=False, avg_pool=False, act=nn.LeakyReLU(0.2)):
    return ResNet(nc, BasicBlock, [3, 4, 6, 3], use_bn, use_sn, use_wn, avg_pool, act)


def ResNet50(nc=3, use_bn=False, use_sn=False, use_wn=False, avg_pool=False, act=nn.LeakyReLU(0.2)):
    return ResNet(nc, Bottleneck, [3, 4, 6, 3], use_bn, use_sn, use_wn, avg_pool, act)


def ResNet101(nc=3, use_bn=False, use_sn=False, use_wn=False, avg_pool=False, act=nn.LeakyReLU(0.2)):
    return ResNetnc, (Bottleneck, [3, 4, 23, 3], use_bn, use_sn, use_wn, avg_pool, act)


def ResNet152(nc=3, use_bn=False, use_sn=False, use_wn=False, avg_pool=False, act=nn.LeakyReLU(0.2)):
    return ResNet(nc, Bottleneck, [3, 8, 36, 3], use_bn, use_sn, use_wn, avg_pool, act)