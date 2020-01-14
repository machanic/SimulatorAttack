'''
Carlini's net for CIFAR-10 (https://arxiv.org/pdf/1608.04644.pdf)
'''

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['carlinet']


class CarliNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CarliNet, self).__init__()

        self.conv2d_1 = nn.Conv2d(in_channels, 64, 3)
        self.conv2d_2 = nn.Conv2d(64, 64, 3)
        self.conv2d_3 = nn.Conv2d(64, 128, 3)
        self.conv2d_4 = nn.Conv2d(128, 128, 3)

        self.dense_1 = nn.Linear(3200, 256)
        self.dense_2 = nn.Linear(256, 256)
        self.dense_3 = nn.Linear(256, num_classes)

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
        # x = F.max_pool2d(x, (2, 2))
        x = F.adaptive_max_pool2d(x, (5,5))
        # carlini's keras model data format: (N, H, W, C)
        # pytorch data format: (N, C, H, W)
        # we need to transpose pytorch data format into keras data format, to make sure the flatten operator
        x = x.view(x.size(0), -1)
        # has the same effect.
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dense_2(x)
        x = F.relu(x)
        x = self.dense_3(x)
        return x


def carlinet(in_channels, num_classes):
    return CarliNet(in_channels, num_classes)
