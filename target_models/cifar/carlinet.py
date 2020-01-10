'''
Carlini's net for CIFAR-10 (https://arxiv.org/pdf/1608.04644.pdf)
'''

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['carlinet']


class CarliNet(nn.Module):
    def __init__(self):
        super(CarliNet, self).__init__()

        self.conv2d_1 = nn.Conv2d(3, 64, 3)
        self.conv2d_2 = nn.Conv2d(64, 64, 3)
        self.conv2d_3 = nn.Conv2d(64, 128, 3)
        self.conv2d_4 = nn.Conv2d(128, 128, 3)

        self.dense_1 = nn.Linear(3200, 256)
        self.dense_2 = nn.Linear(256, 256)
        self.dense_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = F.relu(x)
        x = self.conv2d_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2d_3(x)
        x = F.relu(x)
        x = self.conv2d_4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        # carlini's keras model data format: (N, H, W, C)
        # pytorch data format: (N, C, H, W)
        # we need to transpose pytorch data format into keras data format, to make sure the flatten operator
        # has the same effect.
        x = x.transpose(1, 2).transpose(2, 3).contiguous().view(x.shape[0], -1)
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dense_2(x)
        x = F.relu(x)
        x = self.dense_3(x)
        return x


def carlinet():
    return CarliNet()
