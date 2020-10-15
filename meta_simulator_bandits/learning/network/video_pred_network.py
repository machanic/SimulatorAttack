from torch import nn
import torch
from config import IMAGE_SIZE, IN_CHANNELS
from meta_simulator_bandits.learning.network.conv_lstm import ConvLSTM

class VideoPredNetwork(nn.Module):
    def __init__(self, dataset):
        super(VideoPredNetwork, self).__init__()
        self.input_size = IMAGE_SIZE[dataset]
        self.input_channels = IN_CHANNELS[dataset]
        # Encoder
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv_lstm = ConvLSTM(self.input_size, 64, 64, 3, stride=1, num_layers=2)
        self.conv5 = nn.Conv2d(64, self.input_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # 输出是0到1之间的矩阵
        self.rescale = torch.Tensor(1).cuda()
        torch.nn.init.constant(self.rescale, 0.1)

    def encode(self, x):
        batch_size, T,C,H,W = x.size()
        x = x.view(-1,C,H,W)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))  # (b, t, c, h, w)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))  # (b*t, c, h, w)
        return out

    def decode(self, z, batch_size, T, H, W):
        out = self.relu(self.bn5(self.deconv1(z)))
        out = self.relu(self.bn6(self.deconv2(out)))
        out = self.relu(self.bn7(self.deconv3(out)))
        out = out.view(batch_size, T, 64, H, W)
        out = self.conv_lstm(out)  # (b,t,c,h,w)
        out = out.view(batch_size * T, 64, H, W)
        out = self.rescale * self.tanh(self.conv5(out))
        return out

    def forward(self, x):
        x = x.contiguous()
        batch_size, T, C, H, W = x.size()
        z = self.encode(x)  # (b, t, c, h, w)
        out = self.decode(z, batch_size, T, H, W)  # b*t, c, h, w
        out = out.view(batch_size, T, C, H, W)
        return out