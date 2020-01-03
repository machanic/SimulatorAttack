import torch.nn as nn
import torch.nn.functional as F


class DropoutConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, drop=0):
        super(DropoutConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.drop = drop

    def forward(self, x):
        x = super(DropoutConv2d, self).forward(x)
        x = F.dropout(x, p=self.drop, training=True)
        return x
