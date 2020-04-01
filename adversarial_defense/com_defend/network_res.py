import torch
from torch import nn
import numpy as np

class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)


class ModelRes(nn.Module):
    def __init__(self, in_channels, n_com=3,n_rec=3,b_com=6,b_rec=6,d_com=2,d_rec=2,com_disable=False,rec_disable=False):
        super(ModelRes, self).__init__()
        self.in_channels = in_channels
        self.f1_com, self.f2_com = self.make(n_com, b_com, d_com)
        self.f1_rec, self.f2_rec = self.make(n_rec, b_rec, d_rec)
        self.n_com = n_com
        self.n_rec = n_rec
        self.com_disable = com_disable
        self.rec_disable = rec_disable
        self.res_com_modules = self.res_com()
        self.res_rec_modules = self.res_rec()


    def make(self, n, b, d):
        f1 = [2 ** (b + i) for i in range(n)] + [2 ** (b + n - 1 - i) for i in range(n)]
        f2 = [i * (2 ** d) for i in f1]
        del f2[len(f2) // 2]
        f2_last = 32 if f1[0] > 32 else 16
        f2.append(f2_last)
        return f1, f2

    def conv2d(self, in_channel, out_channel, use_elu=True):
        if use_elu:
            return nn.Sequential(nn.Conv2d(in_channel,out_channel, kernel_size=3,stride=1,bias=True),
                                 nn.ELU())
        else:
            return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, bias=True))

    def res_com(self):
        if self.com_disable:
            self.com_modules = nn.ModuleList()
            self.minus_module = Lambda(lambda x:x-0.5)
            self.com_modules.append(self.minus_module)
            self.com_conv1 = self.conv2d(in_channel=self.in_channels, out_channel=16)
            self.com_modules.append(self.com_conv1)
            self.com_conv2 = self.conv2d(16, 32)
            self.com_modules.append(self.com_conv2)
            self.com_conv3 = self.conv2d(32, 64)
            self.com_modules.append(self.com_conv3)
            self.com_conv4 = self.conv2d(64, 128)
            self.com_modules.append(self.com_conv4)
            self.com_conv5 = self.conv2d(128, 256)
            self.com_modules.append(self.com_conv5)
            self.com_conv6 = self.conv2d(256, 128)
            self.com_modules.append(self.com_conv6)
            self.com_conv7 = self.conv2d(128, 64)
            self.com_modules.append(self.com_conv7)
            self.com_conv8 = self.conv2d(64, 32)
            self.com_modules.append(self.com_conv8)
            self.com_out = self.conv2d(32, 12, use_elu=False)
            self.com_modules.append(self.com_out)
        else:
            self.com_modules = nn.ModuleList()
            for i in range(self.n_com * 2):
                self.com_modules.append(self.res_block([self.f1_com[i], self.f1_com[i], self.f2_com[i]]))
            self.com_out =nn.Conv2d(in_channels=self.f2_com[self.n_com*2-1], out_channels=12,kernel_size=1,stride=1)
            self.com_modules.append(self.com_out)

    def res_rec(self):
        if self.com_disable:
            self.rec_modules = nn.ModuleList()
            self.rec_conv1 = self.conv2d(in_channel=12, out_channel=32)
            self.rec_modules.append(self.rec_conv1)
            self.rec_conv2 = self.conv2d(32,64)
            self.rec_modules.append(self.rec_conv2)
            self.rec_conv3 = self.conv2d(64, 128)
            self.rec_modules.append(self.rec_conv3)
            self.rec_conv4 = self.conv2d(128, 256)
            self.rec_modules.append(self.rec_conv4)
            self.rec_conv5 = self.conv2d(256, 128)
            self.rec_modules.append(self.rec_conv5)
            self.rec_conv6 = self.conv2d(128, 64)
            self.rec_modules.append(self.rec_conv6)
            self.rec_conv7 = self.conv2d(64, 32)
            self.rec_modules.append(self.rec_conv7)
            self.rec_conv8 = self.conv2d(32, 16)
            self.rec_modules.append(self.rec_conv8)
            self.rec_out = self.conv2d(16, self.in_channels, use_elu=False)
            self.rec_modules.append(self.rec_out)
        else:
            self.rec_modules = nn.ModuleList()
            for i in range(self.n_com * 2):
                self.rec_modules.append(self.res_block([self.f1_rec[i],self.f1_rec[i],self.f2_rec[i]]))
            self.rec_out =nn.Conv2d(in_channels=self.f2_com[self.n_com*2-1], out_channels=12,kernel_size=1,stride=1)
            self.rec_modules.append(self.rec_out)

    def res_block(self, filters):
        return nn.Sequential(ConvolutionalBlock(filters),
                                    IdentityBlock(filters),IdentityBlock(filters))

    def forward_com(self, x):
        for module in self.com_modules:
            x = module(x)
        return x

    def forward_rec(self, x):
        for module in self.rec_modules:
            x=  module(x)
        return x

class IdentityBlock(nn.Module):
    def __init__(self,   filters, f=3, s=1):
        super(IdentityBlock, self).__init__()
        filter_0, filter_1, filter_2, filter_3 = filters
        self.conv1 =nn.Conv2d(in_channels=filter_0,out_channels=filter_1,kernel_size=1,stride=1)
        self.conv1_relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(filter_1)
        self.conv2 = nn.Conv2d(in_channels=filter_1, out_channels=filter_2, kernel_size=f, stride=1,
                               padding=f // 2)  # pad = SAME
        self.conv2_relu = nn.ReLU()
        self.conv2_bn = nn.BatchNorm2d(filter_2)
        self.conv3 = nn.Conv2d(in_channels=filter_2, out_channels=filter_3, kernel_size=1, stride=1)  # pad = VALID
        self.conv3_bn = nn.BatchNorm2d(filter_3)
        self.last_relu = nn.ReLU()

    def forward(self,X):
        X_shortcut = X.clone()
        X = self.conv1(X)
        X = self.conv1_bn(X)
        X = self.conv1_relu(X)
        X = self.conv2(X)
        X = self.conv2_bn(X)
        X = self.conv2_relu(X)
        X = self.conv3(X)
        X = self.conv3_bn(X)
        X4 = X+ X_shortcut
        X5 =self.last_relu(X4)
        return X5

class ConvolutionalBlock(nn.Module):
    def __init__(self,   filters, f=3, s=1):
        super(ConvolutionalBlock, self).__init__()
        filter_0, filter_1, filter_2, filter_3 = filters
        self.conv1 =nn.Conv2d(in_channels=filter_0, out_channels=filter_1, kernel_size=1,stride=s)
        self.conv1_relu= nn.ReLU()
        self.conv1_bn = nn.BatchNorm2d(filter_1)
        self.conv2 =nn.Conv2d(in_channels=filter_1, out_channels=filter_2, kernel_size=f,stride=1,padding=f//2)  # pad = SAME
        self.conv2_relu = nn.ReLU()
        self.conv2_bn = nn.BatchNorm2d(filter_2)
        self.conv3 = nn.Conv2d(in_channels=filter_2, out_channels=filter_3, kernel_size=1, stride=1)  # pad = VALID
        self.conv3_bn = nn.BatchNorm2d(filter_3)
        # 直接输入X
        self.conv4 = nn.Conv2d(in_channels=filter_0, out_channels=filter_3, kernel_size=1, stride=s)  # pad = VALID
        self.conv4_bn = nn.BatchNorm2d(filter_3)
        self.last_relu = nn.ReLU()

    def forward(self, X):
        X_shortcut = X.clone()
        X = self.conv1(X)
        X = self.conv1_bn(X)
        X = self.conv1_relu(X)
        X = self.conv2(X)
        X = self.conv2_bn(X)
        X = self.conv2_relu(X)
        X = self.conv3(X)
        X = self.conv3_bn(X)
        X = self.conv3_relu(X)
        X_shortcut = self.conv4(X_shortcut)
        X_shortcut = self.conv4_bn(X_shortcut)
        X5 =X + X_shortcut
        X6 = self.last_relu(X5)
        return X6

