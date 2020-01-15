import pretrainedmodels
from torch import nn
from torchvision import models as torch_models
from cifar_models import Conv3, DenseNet121, DenseNet169, DenseNet201, GoogLeNet, MobileNet, MobileNetV2, ResNet18, \
    ResNet34, ResNet50, ResNet101, ResNet152, PNASNetA, PNASNetB, EfficientNetB0, DPN26, DPN92, ResNeXt29_2x64d, \
    ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d, SENet18, ShuffleNetG2, ShuffleNetG3, vgg11, vgg13, vgg16, vgg19, \
    PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152, wideresnet28, wideresnet34, \
    wideresnet40, gdas, pyramidnet272, carlinet, wideresnet28drop, wideresnet34drop, wideresnet40drop
from config import IN_CHANNELS, IMAGE_SIZE, CLASS_NUM, IMAGE_DATA_ROOT
from tiny_imagenet_models.densenet import densenet161, densenet121, densenet169, densenet201
from cifar_models.miscellaneous import Identity
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d


class ModelConstructor(object):
    @staticmethod
    def construct_cifar_model(arch, dataset):
        if arch == "conv3":
            network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], CLASS_NUM[dataset])
        elif arch == "densenet121":
            network = DenseNet121(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "densenet169":
            network = DenseNet169(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "densenet201":
            network = DenseNet201(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "googlenet":
            network = GoogLeNet(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "mobilenet":
            network = MobileNet(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "mobilenet_v2":
            network = MobileNetV2(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet18":
            network = ResNet18(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet34":
            network = ResNet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet50":
            network = ResNet50(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet101":
            network = ResNet101(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnet152":
            network = ResNet152(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "pnasnetA":
            network = PNASNetA(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "pnasnetB":
            network = PNASNetB(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "efficientnet":
            network = EfficientNetB0(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "dpn26":
            network = DPN26(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "dpn92":
            network = DPN92(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnext29_2":
            network = ResNeXt29_2x64d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnext29_4":
            network = ResNeXt29_4x64d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnext29_8":
            network = ResNeXt29_8x64d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "resnext29_32":
            network = ResNeXt29_32x4d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "senet18":
            network = SENet18(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "shufflenet_G2":
            network = ShuffleNetG2(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "shufflenet_G3":
            network = ShuffleNetG3(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "vgg11":
            network = vgg11(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "vgg13":
            network = vgg13(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "vgg16":
            network = vgg16(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "vgg19":
            network = vgg19(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet18":
            network = PreActResNet18(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet34":
            network = PreActResNet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet50":
            network = PreActResNet50(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet101":
            network = PreActResNet101(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "preactresnet152":
            network = PreActResNet152(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet28":
            network = wideresnet28(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet28drop":
            network = wideresnet28drop(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet34":
            network = wideresnet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet34drop":
            network = wideresnet34drop(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet40":
            network = wideresnet40(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "wideresnet40drop":
            network = wideresnet40drop(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "gdas":
            network = gdas(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "pyramidnet272":
            network = pyramidnet272(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif arch == "carlinet":
            network = carlinet(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        return network

    @staticmethod
    def construct_imagenet_model(arch):
        if arch in torch_models.__dict__:
            network = torch_models.__dict__[arch](pretrained=True)
        elif arch in pretrainedmodels.model_names:
            network = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained="imagenet")
        return network

    @staticmethod
    def construct_tiny_imagenet_model(arch, dataset):
        if arch in torch_models.__dict__:
            network = torch_models.__dict__[arch](pretrained=True)
        num_classes = CLASS_NUM[dataset]
        if arch.startswith("resnet"):
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, num_classes)
        elif arch.startswith("densenet"):
            if arch == "densenet161":
                network = densenet161(pretrained=True)
            elif arch == "densenet121":
                network = densenet121(pretrained=True)
            elif arch == "densenet169":
                network = densenet169(pretrained=True)
            elif arch == "densenet201":
                network = densenet201(pretrained=True)
        elif arch == "resnext32_4":
            network = resnext101_32x4d(pretrained="imagenet")
        elif arch == "resnext64_4":
            network = resnext101_64x4d(pretrained="imagenet")
        elif arch.startswith("vgg"):
            network.avgpool = Identity()
            network.classifier[0] = nn.Linear(512 * 2 * 2, 4096)  # 64 /2**5 = 2
            network.classifier[-1] = nn.Linear(4096, num_classes)
        return network



