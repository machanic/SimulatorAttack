from torch import nn
import torch
from config import IMAGE_SIZE, IN_CHANNELS
from meta_simulator_bandits.learning.network.conv_lstm import ConvLSTM
from meta_simulator_bandits.learning.network.deconv_lstm import DeconvLSTM


class AutoEncoderConvLSTM(nn.Module):
    def __init__(self, dataset):
        super(AutoEncoderConvLSTM, self).__init__()
        self.input_size = IMAGE_SIZE[dataset]
        self.input_channels = IN_CHANNELS[dataset]
        # Encoder
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        reduced_input_size = self.input_size[0]//2, self.input_size[1]//2
        self.conv3 = ConvLSTM(reduced_input_size,32,32,kernel_size=3,stride=1,num_layers=2,padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.deconv1 = DeconvLSTM(reduced_input_size, 32, 32, kernel_size=3, stride=1, num_layers=2, padding=1)
        self.bn4 = nn.BatchNorm3d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, self.input_channels, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # 输出是0到1之间的矩阵
        self.rescale = torch.Tensor(1).cuda()
        torch.nn.init.constant(self.rescale, 0.1)

    def encode(self, x):
        batch_size, T,C,H,W = x.size()
        x = x.view(-1,C,H,W)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))  # (b, t, c, h, w)
        out = out.view(batch_size, T, 32, H//2, W//2)
        out = self.conv3(out)  # (b,t,c,h,w)
        out = out.permute(0,2,1,3,4)  # (b,c,t,h,w)
        out = self.relu(self.bn3(out))  # (b,c,t,h,w)
        out = out.permute(0,2,1,3,4)  # (b,t,c,h,w)
        return out

    def decode(self, z):
        out = self.deconv1(z)
        out = out.permute(0, 2, 1, 3, 4)  # (b,c,t,h,w)
        out = self.relu(self.bn4(out))
        out = out.permute(0, 2, 1, 3, 4).contiguous()  # (b,t,c,h,w)
        batch_size, T, C, H, W = out.size()
        out = out.view(batch_size * T, C, H, W)
        out = self.deconv2(out)
        out = self.relu(self.bn5(out))
        out = self.rescale * self.tanh(self.conv4(out))
        return out

    def forward(self, x):
        batch_size, T, C, H, W = x.size()
        z = self.encode(x)  # (b, t, c, h, w)
        out = self.decode(z)  # b*t, c, h, w
        out = out.view(batch_size, T, C, H, W)
        return out