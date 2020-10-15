from torch import nn
import torch

from config import IMAGE_SIZE
from meta_simulator_bandits.learning.network.conv_lstm import ConvLSTM
from meta_simulator_bandits.learning.network.deconv_lstm import DeconvLSTM


class VAEConvLSTM(nn.Module):
    def __init__(self, dataset, intermediate_size=128, hidden_size=20):
        super(VAEConvLSTM, self).__init__()
        self.input_size = IMAGE_SIZE[dataset]
        # Encoder
        self.conv1 = ConvLSTM(self.input_size, 3, 3, kernel_size=3, stride=1, num_layers=1)
        self.conv2 = ConvLSTM(self.input_size, 3, 32, kernel_size=2, stride=2, num_layers=1, padding=0)
        reduced_input_size = (self.input_size[0]//2, self.input_size[1]//2)
        self.conv3 = ConvLSTM(reduced_input_size, 32, 32, kernel_size=3, stride=1, num_layers=2)
        self.fc1 = nn.Linear(16 * 16 * 32, intermediate_size)
        # Latent space
        self.fc21 = nn.Linear(intermediate_size, hidden_size)
        self.fc22 = nn.Linear(intermediate_size, hidden_size)

        # Decoder
        self.fc3 = nn.Linear(hidden_size, intermediate_size)
        self.fc4 = nn.Linear(intermediate_size, 8192)
        self.deconv1 = DeconvLSTM(reduced_input_size,32,32,kernel_size=3,stride=1,num_layers=2,padding=1)
        self.deconv2 = DeconvLSTM(reduced_input_size,32,32,kernel_size=2,stride=2,num_layers=1,padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # 输出是0到1之间的矩阵
        self.rescale = torch.Tensor(1).cuda()
        torch.nn.init.constant(self.rescale, 0.1)

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 32, 16, 16)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.rescale * self.tanh(self.conv5(out))
        return out

    def forward(self, x):
        x = x.squeeze()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)