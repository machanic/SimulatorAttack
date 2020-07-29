import torch
from torch.nn import functional as F
import numpy as np
from torch import nn

from square_attack_RL.rl.a3c_model import A3CModel
from square_attack_RL.rl.policy import SoftmaxPolicy


class DilatedConvBlock(nn.Module):
    def __init__(self, d_factor, weight=None, bias=None):
        super(DilatedConvBlock, self).__init__()
        self.diconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=d_factor,dilation=d_factor,
                                bias=True)
        if weight is not None and bias is not None:
            self.diconv.weight.fill_(weight.data)
            self.diconv.bias.fill_(bias.data)

    def forward(self, x):
        h = F.relu(self.diconv(x))
        return h


class MyFCN(A3CModel):
    def __init__(self, input_shape, n_actions):
        super(MyFCN, self).__init__()
        self.in_channels = input_shape[0]
        self.conv1 = nn.Conv2d(self.in_channels,64,3,stride=1,padding=1,bias=True,)
        self.diconv2 = DilatedConvBlock(2)
        self.diconv3 = DilatedConvBlock(3)
        self.diconv4 = DilatedConvBlock(4)
        self.diconv5_pi = DilatedConvBlock(3)
        self.diconv6_pi = DilatedConvBlock(2)
        self.conv7_pi = SoftmaxPolicy(nn.Conv2d(64, n_actions, 3, stride=1, padding=1, bias=True))
        self.diconv5_V = DilatedConvBlock(3)
        self.diconv6_V = DilatedConvBlock(2)
        self.conv7_V = nn.Conv2d(64,64,3,stride=1,padding=1,bias=True)
        conv_out_size = self._get_value_conv_out_shape(input_shape)
        self.fc_V_1 = nn.Linear(conv_out_size,256)
        self.relu_1 = nn.ReLU()
        self.fc_V_2 = nn.Linear(256, 1)  # value network是一个值

    def init_weights(self, m):
        if hasattr(m,"weight"):
            torch.nn.init.kaiming_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)

    def _get_value_conv_out_shape(self, shape):
        x = torch.zeros(1, *shape)  # fake image
        h = F.relu(self.conv1(x))
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)
        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        h_V = self.conv7_V(h_V)
        return int(np.prod(h_V.size()))


    def pi_and_v(self, x):
        h = F.relu(self.conv1(x))
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)
        h_pi = self.diconv5_pi(h)
        h_pi = self.diconv6_pi(h_pi)
        pout = self.conv7_pi(h_pi)
        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        h_V = self.conv7_V(h_V)
        h_V = F.relu(h_V)
        h_V = h_V.view(h_V.size(0),-1)
        h_V = self.fc_V_1(h_V)
        h_V = self.relu_1(h_V)
        vout= self.fc_V_2(h_V) # 返回整张图像一个reward
        return pout,  vout