"""
ResNet with Denoising Blocks in PyTorch on CIAFR10.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Cihang Xie, Yuxin Wu, Laurens van der Maaten, Alan Yuille, Kaiming He
    Feature Denoising for Improving Adversarial Robustness. arXiv:1812.03411

Explanation:
[1] If 'whether_denoising' is True, a ResNet with two denoising blocks will be created.
    In contrast 'whether_denoising' is False, a normal ResNet will be created.
[2] 'filter_type' decides which denoising operation the denoising block will apply.
    Now it includes 'Median_Filter' 'Mean_Filter' and 'Gaussian_Filter'.
[3] 'ksize' means the kernel size of the filter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#import kornia
import numpy as np

class BasicBlock(nn.Module): #
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):  # 相当于resnet_bottleneck
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DenoisingBlock(nn.Module):
    def __init__(self, in_planes, ksize, filter_type, embed=True, softmax=True):
        super(DenoisingBlock, self).__init__()
        self.in_planes = in_planes
        self.ksize = ksize
        self.filter_type = filter_type # Median_Filter, Mean_Filter, Gaussian_Filter, Non_Local
        self.embed = embed
        self.softmax = softmax
        self.conv_1x1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1, padding=0)
        if self.filter_type == "NonLocal_Filter":
            self.embedding_theta = nn.Conv2d(in_planes, in_planes//2, 1, stride=1,padding=0)
            self.embedding_phi =nn.Conv2d(in_planes, in_planes//2,1, stride=1, padding=0)

    def non_local_op(self, l, embed, softmax):
        n_in, H, W = l.size(1),l.size(2),l.size(3)
        if embed:
            theta = self.embedding_theta(l)
            phi = self.embedding_phi(l)
            g = l
        else:
            theta, phi, g = l, l, l
        if n_in > H * W or softmax:
            f  = torch.einsum('niab,nicd->nabcd', theta, phi)
            if softmax:
                orig_shape = f.size()
                f = f.view(-1, H * W, H * W)
                f = f / np.sqrt(float(theta.size(1))).item()
                f = torch.softmax(f, dim=-1)
                f = f.view(orig_shape)
            f = torch.einsum('nabcd,nicd->niab', f, g)
        else:
            f = torch.einsum('nihw,njhw->nij', phi, g)
            f = torch.einsum('nij,nihw->njhw', f, theta)

        if not softmax:
            f = f / float(H * W)
        return f.view(l.size()).contiguous()


    def forward(self, x ):
        if self.filter_type == 'Median_Filter':
            x_denoised = kornia.median_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Mean_Filter':
            x_denoised = kornia.box_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Gaussian_Filter':
            x_denoised = kornia.gaussian_blur2d(x, (self.ksize, self.ksize), (0.3 * ((x.shape[3] - 1) * 0.5 - 1) + 0.8, 0.3 * ((x.shape[2] - 1) * 0.5 - 1) + 0.8))
        elif self.filter_type == "NonLocal_Filter":
            x_denoised = self.non_local_op(x, self.embed, self.softmax)
        new_x = x + self.conv_1x1(x_denoised)
        return new_x

class ResNet(nn.Module):
    def __init__(self, in_channels, block, num_blocks, num_classes=10, whether_denoising=True, filter_type="NonLocal_Filter", ksize=3):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.whether_denoising = whether_denoising
        self.in_planes = 64
        self.ksize= ksize
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) # 相当于tf版本的group_func
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, self.in_channels, 1, 1)
        # self.std = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view(1, self.in_channels, 1, 1)

    def _make_layer(self, block, planes, num_blocks, stride):         # [3, 4, 23, 3]
        strides = [stride] + [1]*(num_blocks-1)  # layer1: [1,1,1]  layer2: [2,1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        if self.whether_denoising:
            layers.append(DenoisingBlock(self.in_planes, self.ksize,self.filter_type, True, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = (x - self.mean.type(x.dtype).to(x.device)) / self.std.type(x.dtype).to(x.device)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, output_size=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DenoiseResNet18(in_channels, num_classes, whether_denoising=True, filter_type="NonLocal_Filter", ksize=3):
    return ResNet(in_channels, BasicBlock, [2, 2, 2, 2], num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

def DenoiseResNet34(in_channels, num_classes, whether_denoising=True, filter_type="NonLocal_Filter", ksize=3):
    return ResNet(in_channels, BasicBlock, [3, 4, 6, 3], num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

def DenoiseResNet50(in_channels, num_classes, whether_denoising=True, filter_type="NonLocal_Filter", ksize=3):
    return ResNet(in_channels, Bottleneck, [3, 4, 6, 3], num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

def DenoiseResNet101(in_channels, num_classes, whether_denoising=True, filter_type="NonLocal_Filter", ksize=3):
    return ResNet(in_channels, Bottleneck, [3, 4, 23, 3], num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

def DenoiseResNet152(in_channels, num_classes, whether_denoising=True, filter_type="NonLocal_Filter", ksize=3):
    return ResNet(in_channels, Bottleneck, [3, 8, 36, 3], num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)


