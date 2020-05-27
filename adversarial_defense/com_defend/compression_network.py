from adversarial_defense.com_defend.processer import *
import numpy as np
import time
import os
import math
import cv2
from torch import nn
import torch
class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)

class ComCNN(nn.Module):
    def __init__(self, in_channels):
        super(ComCNN, self).__init__()
        self.c = nn.ModuleList()
        self.c.append(Lambda(lambda x:x-0.5))
        self.conv(in_channels,16)
        self.conv(16,32)
        self.conv(32,64)
        self.conv(64,128)
        self.conv(128,256)
        self.conv(256,128)
        self.conv(128,64)
        self.conv(64,32)
        self.conv(32,12,flag=False)


    def conv(self, nip, nop, flag=True):
        self.c.append(nn.Conv2d(nip,nop,kernel_size=3,bias=True,padding=1))
        if flag:
            self.c.append(nn.ELU())
    def forward(self, x):
        for conv in self.c:
            x=  conv(x)
        return x

class RecCNN(nn.Module):
    def __init__(self, in_channels):
        super(RecCNN, self).__init__()
        self.c = nn.ModuleList()
        self.conv(12,32)
        self.conv(32, 64)
        self.conv(64, 128)
        self.conv(128, 256)
        self.conv(256, 128)
        self.conv(128, 64)
        self.conv(64, 32)
        self.conv(32, 16)
        self.conv(16, in_channels, flag=False)
        self.c.append(nn.Sigmoid())

    def conv(self, nip, nop, flag=True):
        self.c.append(nn.Conv2d(nip,nop, kernel_size=3,bias=True,padding=1))
        if flag:
            self.c.append(nn.ELU())
    def forward(self, x):
        for conv in self.c:
            x=  conv(x)
        return x


class ComDefend(nn.Module):
    def __init__(self, in_channels, noise_dev):
        super(ComDefend, self).__init__()
        self.com_cnn = ComCNN(in_channels)
        self.rec_cnn = RecCNN(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.noise_dev = noise_dev

    def forward(self, x):
        assert x.size(1) == self.in_channels
        linear_code =self.com_cnn(x)
        noisy_code =linear_code -torch.randn_like(linear_code).cuda()  * self.noise_dev
        binary_code =self.sigmoid(noisy_code)
        # y = self.rec_cnn(binary_code)
        binary_code_test = (binary_code > 0.5).float()
        y_test =self.rec_cnn(binary_code_test)
        return y_test  # 训练的时候用倒数第二个

    def __call__(self, x):
        return self.forward(x)

    def forward_com(self, x):
        return self.com_cnn(x)

    def forward_rec(self, x):
        return self.rec_cnn(x)


