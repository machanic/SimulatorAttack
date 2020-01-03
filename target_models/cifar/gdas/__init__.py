'''GDAS net.
Ported form
https://github.com/D-X-Y/GDAS
(c) Yuanyi Dong
'''
import os
import os.path as osp
import torch

from target_models.cifar.gdas.lib.scheduler import load_config
from target_models.cifar.gdas.lib.nas import model_types
from target_models.cifar.gdas.lib.nas import NetworkCIFAR as Network

__all__ = ['gdas']


def gdas(checkpoint_fname):
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    xargs = checkpoint['args']
    config = load_config(os.path.join(osp.dirname(__file__), xargs.model_config))
    genotype = model_types[xargs.arch]
    class_num = 10

    model = Network(xargs.init_channels, class_num, xargs.layers, config.auxiliary, genotype)
    model.load_state_dict(checkpoint['state_dict'])
    return model
