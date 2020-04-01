import glob
import os

import pretrainedmodels
import torch
import torchvision.models as vision_models
from torch import nn
from torchvision import models as torch_models

import cifar_models as models
from adversarial_defense.com_defend.compression_network import ComDefend
from adversarial_defense.feature_distillation.jpeg import dnn_jpeg
from adversarial_defense.model.feature_defense_model import FeatureDefenseModel
from adversarial_defense.post_averaging.post_averaged_models import PostAveragedNetwork
from cifar_models_myself.efficient_densenet import EfficientDenseNet
from cifar_models_myself.ghostnet import ghost_net
from cifar_models_myself.miscellaneous import Identity
from config import pretrained_cifar_model_conf, IN_CHANNELS, IMAGE_SIZE, CLASS_NUM, PY_ROOT
from tiny_imagenet_models.densenet import densenet161, densenet121, densenet169, densenet201
from tiny_imagenet_models.inception import inception_v3
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d
from tiny_imagenet_models.wrn import tiny_imagenet_wrn


class DefensiveModel(nn.Module):
    """
    A DefensiveModel object wraps a cnn model.
    This model always accept standard image: in [0, 1] range, RGB order, un-normalized, NCHW format
    Two types of defensive models are available:
    1. com_defend, feature_distillation,post_averaging (based on preprocessing denoise)
    2. feature_scatter, pcl_loss, feature_denoise  (based on adversarial training)
    """
    def __init__(self, dataset, arch, no_grad, defense_type):
        super(DefensiveModel, self).__init__()
        # init cnn model
        self.in_channels = IN_CHANNELS[dataset]
        self.dataset = dataset
        self.arch = arch

        if defense_type != "feature_denoise" and defense_type != "pcl_loss":
            if dataset.startswith("CIFAR"):
                trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}-pretrained/{arch}/checkpoint.pth.tar".format(root=PY_ROOT, dataset=dataset, arch=arch)
                assert os.path.exists(trained_model_path), "{} does not exist!".format(trained_model_path)
            elif dataset == "TinyImageNet":
                trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}@*.pth.tar".format(root=PY_ROOT, dataset=dataset, arch=arch)
                trained_model_path_list = list(glob.glob(trained_model_path))
                assert len(trained_model_path_list)>0, "{} does not exist!".format(trained_model_path)
                trained_model_path = trained_model_path_list[0]
            else:
                trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}-pretrained/checkpoints/{arch}*.pth".format(
                    root=PY_ROOT, dataset=dataset, arch=arch)
                trained_model_path_ls = list(glob.glob(trained_model_path))
                assert trained_model_path_ls,  "{} does not exist!".format(trained_model_path)
                trained_model_path = trained_model_path_ls[0]
            self.model = self.make_model(dataset, arch, self.in_channels, CLASS_NUM[dataset], trained_model_path=trained_model_path)
            # init cnn model meta-information
            self.mean = torch.FloatTensor(self.model.mean).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad =True
            self.std = torch.FloatTensor(self.model.std).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True

        if defense_type == "com_defend":
            self.preprocessor = ComDefend(self.in_channels, 20)
        elif defense_type == "feature_distillation":
            self.preprocessor = dnn_jpeg
        elif defense_type == "post_averaging":
            R = 30 if arch in ["resnet-101", "resnet-110", "resnet-152"] else 6
            self.model = PostAveragedNetwork(self.cnn, K=15, R=R)
        elif defense_type == "feature_denoise" or defense_type == "pcl_loss":
            self.model = FeatureDefenseModel(self.dataset, self.arch, no_grad).cnn
            self.mean = self.model.mean
            self.std = self.model.std


        self.input_space = self.cnn.input_space  # 'RGB' or 'GBR'
        self.input_range = self.cnn.input_range  # [0, 1] or [0, 255]
        self.input_size = self.cnn.input_size
        self.no_grad = no_grad


    def forward(self, x):
        # assign dropout probability
        # if hasattr(self, 'drop'):
        #     self.cnn.drop = self.drop
        # channel order
        if self.input_space == 'BGR':
            x = x[:, [2, 1, 0], :, :]  # pytorch does not support negative stride index (::-1) yet
        # input range
        if max(self.input_range) == 255:
            x = x * 255
        if hasattr(self, "preprocessor"):
            x = self.preprocessor(x) # feature distillation
        # normalization
        x = (x - self.mean.type(x.dtype).to(x.device)) / self.std.type(x.dtype).to(x.device)
        if self.no_grad:
            with torch.no_grad():
                x = self.model(x)
        else:
            x = self.model(x)
        x = x.view(x.size(0), -1)
        return x

    def load_weight_from_pth_checkpoint(self, model, fname):
        raw_state_dict = torch.load(fname, map_location='cpu')['state_dict']
        state_dict = dict()
        for key, val in raw_state_dict.items():
            new_key = key.replace('module.', '')
            state_dict[new_key] = val
        model.load_state_dict(state_dict)

    def construct_cifar_model(self, arch, dataset, num_classes):
        conf = pretrained_cifar_model_conf[dataset][arch]
        arch = arch.split("-")[0].lower()
        if arch.startswith('resnext'):
            model = models.__dict__[arch](
                cardinality=conf["cardinality"],
                num_classes=num_classes,
                depth=conf["depth"],
                widen_factor=conf["widen_factor"],
                dropRate=conf["drop"],
            )
        elif arch.startswith('densenet'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                depth=conf["depth"],
                growthRate=conf["growthRate"],
                compressionRate=conf["compressionRate"],
                dropRate=conf["drop"],
            )
        elif arch.startswith('wrn'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                depth=conf["depth"],
                widen_factor=conf["widen_factor"],
                dropRate=conf["drop"],
            )
        elif arch.endswith('resnet'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                depth=conf["depth"],
                block_name=conf["block_name"],
            )
        else:
            model = models.__dict__[arch](num_classes=num_classes)
        return model

    def make_model(self, dataset, arch, in_channel, num_classes, trained_model_path=None):
        """
        Make model, and load pre-trained weights.
        :param dataset: cifar10 or imagenet
        :param arch: arch name, e.g., alexnet_bn
        :return: model (in cpu and training mode)
        """
        if dataset in ['CIFAR-10',"CIFAR-100", "MNIST","FashionMNIST"]:
            assert trained_model_path is not None and os.path.exists(trained_model_path), "Pretrained weight model file {} does not exist!".format(trained_model_path)
            if arch == 'gdas':
                model = models.gdas(in_channel, num_classes)
                model.mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
                model.std = [63.0 / 255, 62.1 / 255, 66.7 / 255]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [in_channel, IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            elif arch == 'pyramidnet272':
                model = models.pyramidnet272(in_channel, num_classes)
                model.mean = [0.49139968, 0.48215841, 0.44653091]
                model.std = [0.24703223, 0.24348513, 0.26158784]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [in_channel, IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            else:
                model = self.construct_cifar_model(arch, dataset, num_classes)
                model.mean = [0.4914, 0.4822, 0.4465]
                model.std = [0.2023, 0.1994, 0.2010]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [in_channel, IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            self.load_weight_from_pth_checkpoint(model, trained_model_path)
        elif dataset == "TinyImageNet":
            model = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
            model.input_space = 'RGB'
            model.input_range = [0, 1]
            model.mean = [0,0,0]
            model.std = [1,1,1]
            model.input_size = [in_channel,IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, location: storage)["state_dict"])
        elif dataset == 'ImageNet':
            os.environ["TORCH_HOME"] = "{}/train_pytorch_model/real_image_model/ImageNet-pretrained".format(PY_ROOT)
            model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained="imagenet")
        return model


class MetaLearnerModelBuilder(object):

    @staticmethod
    def construct_imagenet_model(arch, dataset):
        os.environ["TORCH_HOME"] = "{}/train_pytorch_model/real_image_model/ImageNet-pretrained".format(PY_ROOT)
        if arch == 'efficient_densenet':
            depth = 40
            block_config = [(depth - 4) // 6 for _ in range(3)]
            return EfficientDenseNet(IN_CHANNELS[dataset],block_config=block_config, num_classes=CLASS_NUM[dataset], small_inputs=False, efficient=False)
        elif arch ==  "ghost_net":
            network = ghost_net(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            return network
        model = vision_models.__dict__[arch](pretrained=False)
        return model

    @staticmethod
    def construct_tiny_imagenet_model(arch, dataset):
        if not arch.startswith("densenet") and not arch.startswith("resnext") and arch in torch_models.__dict__:
            network = torch_models.__dict__[arch](pretrained=False)
        num_classes = CLASS_NUM[dataset]
        if arch.startswith("resnet"):
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, num_classes)
        elif arch.startswith("densenet"):
            if arch == "densenet161":
                network = densenet161(pretrained=False)
            elif arch == "densenet121":
                network = densenet121(pretrained=False)
            elif arch == "densenet169":
                network = densenet169(pretrained=False)
            elif arch == "densenet201":
                network = densenet201(pretrained=False)
        elif arch == "resnext32_4":
            network = resnext101_32x4d(pretrained=None)
        elif arch == "resnext64_4":
            network = resnext101_64x4d(pretrained=None)
        elif arch == "ghost_net":
            network = ghost_net(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch.startswith("inception"):
            network = inception_v3(pretrained=False)
        elif arch == "WRN-28-10-drop":
            network = tiny_imagenet_wrn(in_channels=IN_CHANNELS[dataset],depth=28,num_classes=CLASS_NUM[dataset],widen_factor=10, dropRate=0.3)
        elif arch == "WRN-40-10-drop":
            network = tiny_imagenet_wrn(in_channels=IN_CHANNELS[dataset], depth=40, num_classes=CLASS_NUM[dataset],
                                        widen_factor=10, dropRate=0.3)
        elif arch.startswith("vgg"):
            network.avgpool = Identity()
            network.classifier[0] = nn.Linear(512 * 2 * 2, 4096)  # 64 /2**5 = 2
            network.classifier[-1] = nn.Linear(4096, num_classes)
        return network



