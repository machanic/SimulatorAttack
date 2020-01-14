import glob
import os.path as osp
import h5py
import target_models.cifar
import target_models.imagenet
from config import IN_CHANNELS, IMAGE_SIZE, PY_ROOT, CLASS_NUM
from tiny_imagenet_models.inception import inception_v3
from torchvision import models as torch_models
from target_models.cifar.carlinet import carlinet
import numpy as np
from cifar_models import *
from tiny_imagenet_models.densenet import densenet121, densenet161, densenet169, densenet201
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d


class StandardModel(nn.Module):
    """
    A StandardModel object wraps a cnn model.
    This model always accept standard image: in [0, 1] range, RGB order, un-normalized, NCHW format
    """
    def __init__(self, dataset, arch, no_grad=True, **kwargs):
        super(StandardModel, self).__init__()
        # init cnn model
        self.cnn = self.make_model(dataset, arch, **kwargs)
        self.cnn.cuda()

        # init cnn model meta-information
        self.mean = torch.FloatTensor(self.cnn.mean).view(1, 3, 1, 1).cuda()
        self.std = torch.FloatTensor(self.cnn.std).view(1, 3, 1, 1).cuda()
        self.input_space = self.cnn.input_space  # 'RGB' or 'GBR'
        self.input_range = self.cnn.input_range  # [0, 1] or [0, 255]
        self.input_size = self.cnn.input_size

        self.no_grad = no_grad

    def forward(self, x):
        # assign dropout probability
        if hasattr(self, 'drop'):
            self.cnn.drop = self.drop
        # channel order
        if self.input_space == 'BGR':
            x = x[:, [2, 1, 0], :, :]  # pytorch does not support negative stride index (::-1) yet

        # input range
        if max(self.input_range) == 255:
            x = x * 255

        # normalization
        x = (x - self.mean) / self.std

        if self.no_grad:
            with torch.no_grad():
                x = self.cnn(x)
        else:
            x = self.cnn(x)
        return x

    def load_weight_from_pth_checkpoint(self, model, fname):
        raw_state_dict = torch.load(fname, map_location='cpu')['state_dict']
        state_dict = dict()
        for key, val in raw_state_dict.items():
            new_key = key.replace('module.', '')
            state_dict[new_key] = val

        model.load_state_dict(state_dict)

    def load_weight_from_h5_checkpoint(self, model, fname):
        assert 'carlinet' in model.__class__.__name__.lower()
        with h5py.File(fname, 'r') as f:
            for key in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'dense_1', 'dense_2', 'dense_3']:
                m = model.__getattr__(key)
                # weight
                if 'conv' in key:
                    w = np.array(f['model_weights'][key][key]['kernel:0']).transpose(3, 2, 0, 1)
                if 'dense' in key:
                    w = np.array(f['model_weights'][key][key]['kernel:0']).transpose(1, 0)
                assert m.weight.shape == w.shape
                m.weight.data[:] = torch.FloatTensor(w)
                # bias
                b = np.array(f['model_weights'][key][key]['bias:0'])
                assert m.bias.shape == b.shape
                m.bias.data[:] = torch.FloatTensor(b)

    def make_model(self, dataset, arch, **kwargs):
        """
        Make model, and load pre-trained weights.
        :param dataset: cifar10 or imagenet
        :param arch: arch name, e.g., alexnet_bn
        :return: model (in cpu and training mode)
        """
        if dataset in ['CIFAR-10',"MNIST","FashionMNIST"]:
            if arch == 'gdas':
                assert kwargs['train_data'] == 'full'
                model = target_models.cifar.gdas('{}/subspace_attack/data/cifar10-models/gdas/seed-6293/checkpoint-cifar10-model.pth'.format(PY_ROOT))
                model.mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
                model.std = [63.0 / 255, 62.1 / 255, 66.7 / 255]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            elif arch == 'pyramidnet272':
                assert kwargs['train_data'] == 'full'
                model = target_models.cifar.pyramidnet272(num_classes=10)
                self.load_weight_from_pth_checkpoint(model, '{}/subspace_attack/data/cifar10-models/pyramidnet272/checkpoint.pth'.format(PY_ROOT))
                model.mean = [0.49139968, 0.48215841, 0.44653091]
                model.std = [0.24703223, 0.24348513, 0.26158784]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            elif arch == "carlinet":
                assert kwargs['train_data'] == 'full'
                model = carlinet()
                self.load_weight_from_h5_checkpoint(model, '{}/subspace_attack/data/cifar10-models/carlinet'.format(PY_ROOT))
                model.mean = [0.5, 0.5, 0.5]
                model.std = [1, 1, 1]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size =  [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]

            else:
                # decide weight filename prefix, suffix
                if kwargs['train_data'] in ['cifar10.1']:
                    # use cifar10.1 (2,000 images) to train_simulate_grad_mode target_models
                    if kwargs['train_data'] == 'cifar10.1':
                        prefix = '{}/subspace_attack/data/cifar10.1-models'.format(PY_ROOT)
                    else:
                        raise NotImplementedError('Unknown train data {}'.format(kwargs['train_data']))
                    if kwargs['epoch'] == 'final':
                        suffix = 'final.pth'
                    elif kwargs['epoch'] == 'best':
                        suffix = 'model_best.pth'
                    else:
                        raise NotImplementedError('Unknown epoch {} for train data {}'.format(
                            kwargs['epoch'], kwargs['train_data']))
                elif kwargs['train_data'] == 'full':
                    # use full training set to train_simulate_grad_mode target_models
                    prefix = '{}/subspace_attack/data/cifar10-models'.format(PY_ROOT)
                    if kwargs['epoch'] == 'final':
                        suffix = 'checkpoint.pth.tar'
                    elif kwargs['epoch'] == 'best':
                        suffix = 'model_best.pth.tar'
                    else:
                        raise NotImplementedError('Unknown epoch {} for train data {}'.format(
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
                self.load_weight_from_pth_checkpoint(model, osp.join(prefix, arch, suffix))

                # assign meta info
                model.mean = [0.4914, 0.4822, 0.4465]
                model.std = [0.2023, 0.1994, 0.2010]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [3, 32, 32]
        elif dataset == "TinyImageNet":
            class Identity(nn.Module):
                def __init__(self):
                    super(Identity, self).__init__()

                def forward(self, x):
                    return x
            if arch in target_models.__dict__:
                model = target_models.__dict__[arch](pretrained=True)
            num_classes = CLASS_NUM[dataset]
            if arch.startswith("resnet"):
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
            elif arch.startswith("densenet"):
                if arch == "densenet161":
                    model = densenet161(pretrained=True)
                elif arch == "densenet121":
                    model = densenet121(pretrained=True)
                elif arch == "densenet169":
                    model = densenet169(pretrained=True)
                elif arch == "densenet201":
                    model = densenet201(pretrained=True)
            elif arch == "resnext32_4":
                model = resnext101_32x4d(pretrained="imagenet")
            elif arch == "resnext64_4":
                model = resnext101_64x4d(pretrained="imagenet")
            elif arch.startswith("inception"):
                model = inception_v3(pretrained=True)
            elif arch.startswith("vgg"):
                model.avgpool = Identity()
                model.classifier[0] = nn.Linear(512 * 2 * 2, 4096)  # 64 /2**5 = 2
                model.classifier[-1] = nn.Linear(4096, num_classes)
            model_path = "{}/train_pytorch_model/real_image_model/{}@{}*.pth.tar".format(PY_ROOT, dataset, arch)
            model_path = glob.glob(model_path)
            model_path = model_path[0]
            model.load_state_dict(
                torch.load(model_path, map_location=lambda storage, location: storage)["state_dict"])
        elif dataset == 'ImageNet':

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
                self.load_weight_from_pth_checkpoint(model, osp.join(prefix, arch, suffix))
            else:
                raise NotImplementedError('Unknown train_simulate_grad_mode data {}'.format(kwargs['train_data']))
        else:
            raise NotImplementedError('Unknown dataset {}'.format(dataset))

        return model