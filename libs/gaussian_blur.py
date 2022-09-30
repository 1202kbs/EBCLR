from torchvision import transforms

import torch.nn as nn
import numpy as np
import torch

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radius = kernel_size // 2
        kernel_size = radius * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radius

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radius),
            self.blur_h,
            self.blur_v
        )

    def __call__(self, img):
        
        # img = img[None]

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            # img = img.squeeze()

        return img