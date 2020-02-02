'''GDAS net.
Ported form
https://github.com/D-X-Y/GDAS
(c) Chen Ma
'''
import os
import os.path as osp
from types import SimpleNamespace
from cifar_models.gdas.lib.scheduler import load_config
from cifar_models.gdas.lib.nas import model_types
from cifar_models.gdas.lib.nas import NetworkCIFAR as Network
__all__ = ['gdas']

from config import PY_ROOT


def gdas(in_channels, num_classes):
    model_config = "{}/cifar_models/gdas/configs/nas-cifar-cos-cutW5.config".format(PY_ROOT)
    assert os.path.exists(model_config)
    xargs = {"arch":"GDAS_F1", "grad_clip":5.0, "init_channels":36, "layers":20, "manualSeed":6239,"model_config":model_config,
             "print_freq":100, "workers":0}
    xargs = SimpleNamespace(**xargs)
    config = load_config(os.path.join(osp.dirname(__file__), xargs.model_config))
    assert os.path.exists(os.path.join(osp.dirname(__file__), xargs.model_config)), os.path.join(osp.dirname(__file__), xargs.model_config)
    genotype = model_types[xargs.arch]
    model = Network(in_channels, xargs.init_channels, num_classes, xargs.layers, config.auxiliary, genotype)
    return model
