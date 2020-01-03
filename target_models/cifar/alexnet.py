'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F
from .utils import DropoutConv2d

__all__ = ['alexnet', 'alexnet_bn']


class AlexNet(nn.Module):

    def __init__(self, batch_norm=False, num_classes=10):
        super(AlexNet, self).__init__()
        layers = [
            DropoutConv2d(3, 64, kernel_size=11, stride=4, padding=5)
        ]
        if batch_norm:
            layers += [nn.BatchNorm2d(64)]
        layers += [
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DropoutConv2d(64, 192, kernel_size=5, padding=2),
        ]
        if batch_norm:
            layers += [nn.BatchNorm2d(192)]
        layers += [
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DropoutConv2d(192, 384, kernel_size=3, padding=1),
        ]
        if batch_norm:
            layers += [nn.BatchNorm2d(384)]
        layers += [
            nn.ReLU(inplace=True),
            DropoutConv2d(384, 256, kernel_size=3, padding=1),
        ]
        if batch_norm:
            layers += [nn.BatchNorm2d(256)]
        layers += [
            nn.ReLU(inplace=True),
            DropoutConv2d(256, 256, kernel_size=3, padding=1),
        ]
        if batch_norm:
            layers += [nn.BatchNorm2d(256)]
        layers += [
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(256, num_classes)

        # no dropout
        self.drop = 0

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, DropoutConv2d):
                m.drop = self.drop

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.drop, training=True)  # force dropout
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(batch_norm=False, **kwargs)
    return model


def alexnet_bn(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(batch_norm=True, **kwargs)
    return model
