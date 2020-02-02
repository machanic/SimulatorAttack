from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self, input_channels):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
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
        self.conv5 = nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(input_channels)
        # self.constant = torch.tensor(3.0).float().cuda()
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()  # 输出是0到1之间的矩阵


    def encode(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        return out

    def decode(self, z):
        out = self.relu(self.bn5(self.deconv1(z)))
        out = self.relu(self.bn6(self.deconv2(out)))
        out = self.relu(self.bn7(self.deconv3(out)))
        out = self.bn8(self.conv5(out))
        return out

    def forward(self, x):
        C,H,W = x.size(-3), x.size(-2), x.size(-1)
        x = x.view(-1, C, H, W)
        z = self.encode(x)
        return self.decode(z)

