import glog as log
import os.path as osp
import h5py
import numpy as np
import torch
import target_models.cifar
import target_models.imagenet

from config import PY_ROOT


def load_weight_from_pth_checkpoint(model, fname):
    log.info('Load weights from {}'.format(fname))
    raw_state_dict = torch.load(fname, map_location='cpu')['state_dict']
    state_dict = dict()
    for key, val in raw_state_dict.items():
        new_key = key.replace('module.', '')
        state_dict[new_key] = val

    model.load_state_dict(state_dict)


def make_model(dataset, arch, **kwargs):
    """
    Make model, and load pre-trained weights.
    :param dataset: cifar10 or imagenet
    :param arch: arch name, e.g., alexnet_bn
    :return: model (in cpu and training mode)
    """
    assert dataset in ['CIFAR-10', 'imagenet']
    if dataset == 'CIFAR-10':
        if arch == 'gdas':
            assert kwargs['train_data'] == 'full'
            model = target_models.cifar.gdas('{}/subspace_attack/data/cifar10-target_models/gdas/seed-6293/checkpoint-cifar10-model.pth'.format(PY_ROOT))
            model.mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
            model.std = [63.0 / 255, 62.1 / 255, 66.7 / 255]
            model.input_space = 'RGB'
            model.input_range = [0, 1]
            model.input_size = [3, 32, 32]
        elif arch == 'pyramidnet272':
            assert kwargs['train_data'] == 'full'
            model = target_models.cifar.pyramidnet272(num_classes=10)
            load_weight_from_pth_checkpoint(model, '{}/subspace_attack/data/cifar10-target_models/pyramidnet272/checkpoint.pth'.format(PY_ROOT))
            model.mean = [0.49139968, 0.48215841, 0.44653091]
            model.std = [0.24703223, 0.24348513, 0.26158784]
            model.input_space = 'RGB'
            model.input_range = [0, 1]
            model.input_size = [3, 32, 32]
        else:
            # decide weight filename prefix, suffix
            if kwargs['train_data'] in ['cifar10.1']:
                # use cifar10.1 (2,000 images) to train_simulate_grad_mode target_models
                if kwargs['train_data'] == 'cifar10.1':
                    prefix = '{}/subspace_attack/data/cifar10.1-target_models'.format(PY_ROOT)
                else:
                    raise NotImplementedError('Unknown train_simulate_grad_mode data {}'.format(kwargs['train_data']))
                if kwargs['epoch'] == 'final':
                    suffix = 'final.pth'
                elif kwargs['epoch'] == 'best':
                    suffix = 'model_best.pth'
                else:
                    raise NotImplementedError('Unknown epoch {} for train_simulate_grad_mode data {}'.format(
                        kwargs['epoch'], kwargs['train_data']))
            elif kwargs['train_data'] == 'full':
                # use full training set to train_simulate_grad_mode target_models
                prefix = 'data/cifar10-target_models'
                if kwargs['epoch'] == 'final':
                    suffix = 'checkpoint.pth.tar'
                elif kwargs['epoch'] == 'best':
                    suffix = 'model_best.pth.tar'
                else:
                    raise NotImplementedError('Unknown epoch {} for train_simulate_grad_mode data {}'.format(
                        kwargs['epoch'], kwargs['train_data']))
            else:
                raise NotImplementedError('Unknown train_simulate_grad_mode data {}'.format(kwargs['train_data']))

            if arch == 'alexnet_bn':
                model = target_models.cifar.alexnet_bn(num_classes=10)
            elif arch == 'vgg11_bn':
                model = target_models.cifar.vgg11_bn(num_classes=10)
            elif arch == 'vgg13_bn':
                model = target_models.cifar.vgg13_bn(num_classes=10)
            elif arch == 'vgg16_bn':
                model = target_models.cifar.vgg16_bn(num_classes=10)
            elif arch == 'vgg19_bn':
                model = target_models.cifar.vgg19_bn(num_classes=10)
            elif arch == 'wrn-28-10-drop':
                model = target_models.cifar.wrn(depth=28, widen_factor=10, dropRate=0.3, num_classes=10)
            else:
                raise NotImplementedError('Unknown arch {}'.format(arch))

            # load weight
            load_weight_from_pth_checkpoint(model, osp.join(prefix, arch, suffix))

            # assign meta info
            model.mean = [0.4914, 0.4822, 0.4465]
            model.std = [0.2023, 0.1994, 0.2010]
            model.input_space = 'RGB'
            model.input_range = [0, 1]
            model.input_size = [3, 32, 32]

    elif dataset == 'imagenet':

        model = eval('target_models.imagenet.{}(num_classes=1000, pretrained=\'imagenet\')'.format(arch))

        if kwargs['train_data'] == 'full':
            # torchvision has load correct checkpoint automatically
            pass
        elif kwargs['train_data'] == 'imagenetv2-val':
            prefix = '{}/subspace_attack/data/imagenetv2-v1val45000-target_models'.format(PY_ROOT)
            if kwargs['epoch'] == 'final':
                suffix = 'checkpoint.pth.tar'
            elif kwargs['epoch'] == 'best':
                suffix = 'model_best.pth.tar'
            else:
                raise NotImplementedError('Unknown epoch {} for train_simulate_grad_mode data {}'.format(
                    kwargs['epoch'], kwargs['train_data']))

            # load weight
            load_weight_from_pth_checkpoint(model, osp.join(prefix, arch, suffix))
        else:
            raise NotImplementedError('Unknown train_simulate_grad_mode data {}'.format(kwargs['train_data']))
    else:
        raise NotImplementedError('Unknown dataset {}'.format(dataset))

    return model
