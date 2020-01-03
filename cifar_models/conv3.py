import math
from collections import OrderedDict

from torch import nn
import torch


class Conv3(nn.Module):
    '''
    The base model for few-shot learning on Omniglot
    '''

    def __init__(self, in_channels, img_size, num_classes):
        super(Conv3, self).__init__()
        self.img_size = img_size
        # Define the network
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 64, 3)),
            ('bn1', nn.BatchNorm2d(64, momentum=1, affine=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('bn2', nn.BatchNorm2d(64, momentum=1, affine=True)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(64, 64, 3)),
            ('bn3', nn.BatchNorm2d(64, momentum=1, affine=True)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2))
        ]))
        self.add_module('fc', nn.Linear(64 * (self.img_size[0] // 2 ** 4) *  (self.img_size[1] // 2 ** 4), num_classes))

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.channels=  in_channels
        # Initialize weights
        self._init_weights()

    def forward(self, x):
        ''' Define what happens to data in the net '''
        x = x.view(-1, self.channels, self.img_size[0], self.img_size[1])
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _init_weights(self):
        ''' Set weights to Gaussian, biases to zero '''
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_() + 1
                m.bias.data = torch.ones(m.bias.data.size())

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        # TODO: breaks if nets are not identical
        # TODO: won't copy buffers, e.g. for batch norm
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward_pass(self, in_, target, weights=None):
        ''' Run data through net, return loss and output '''
        input_var = in_.cuda()
        target_var = target.cuda()
        # Run the batch through the net, compute loss
        out = self.net_forward(input_var, weights)
        loss = self.loss_fn(out, target_var)
        return loss, out
