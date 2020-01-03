import argparse
import glob
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import os

import re
import time

import torch
from torchvision import models as torch_models
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torchvision.transforms import transforms

from config import IN_CHANNELS, IMAGE_SIZE, CLASS_NUM, IMAGE_DATA_ROOT
from tiny_imagenet_models.densenet import densenet121, densenet161, densenet169, densenet201
from tiny_imagenet_models.inception import inception_v3
from tiny_imagenet_models.miscellaneous import Identity
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d
from cifar_models import *

def construct_model(arch, dataset):
    if dataset != "TinyImageNet":
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
            network = ShuffleNetG2(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "shufflenet_G3":
            network = ShuffleNetG3(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "vgg11":
            network = vgg11(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "vgg13":
            network = vgg13(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "vgg16":
            network = vgg16(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "vgg19":
            network = vgg19(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "preactresnet18":
            network = PreActResNet18(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "preactresnet34":
            network = PreActResNet34(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "preactresnet50":
            network = PreActResNet50(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "preactresnet101":
            network = PreActResNet101(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "preactresnet152":
            network = PreActResNet152(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "wideresnet28":
            network = wideresnet28(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "wideresnet34":
            network = wideresnet34(IN_CHANNELS[dataset],CLASS_NUM[dataset])
        elif arch == "wideresnet40":
            network = wideresnet40(IN_CHANNELS[dataset],CLASS_NUM[dataset])
    else:
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
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate_accuracy mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            # measure accuracy and record loss
            acc1,  = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0], input.size(0))
            # top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg

def get_preprocessor():
    preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return preprocess_transform

data_loader_cache = {}
def get_data_loader(dataset):
    if dataset in data_loader_cache:
        return data_loader_cache[dataset]
    preprocessor = get_preprocessor()
    if dataset == "CIFAR-10":
        val_dataset = CIFAR10(IMAGE_DATA_ROOT[dataset], train=False, transform=preprocessor)
    elif dataset == "MNIST":
        val_dataset = MNIST(IMAGE_DATA_ROOT[dataset], train=False, transform=preprocessor)
    elif dataset == "FashionMNIST":
        val_dataset = FashionMNIST(IMAGE_DATA_ROOT[dataset], train=False, transform=preprocessor)
    elif dataset=="SVHN":
        val_dataset = SVHN(IMAGE_DATA_ROOT[dataset],  transform=preprocessor)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=50, shuffle=False,
        num_workers=0, pin_memory=True)
    data_loader_cache[dataset] = val_loader
    return val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument("--dir_path",type=str,default="/home1/machen/meta_perturbations_black_box_attack/train_pytorch_model/real_image_model")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    print("using GPU {}".format(args.gpu))
    pattern = re.compile("(.*?)@(.*?)@.*tar")
    for abs_path in glob.glob(args.dir_path + "/*.tar"):
        f = os.path.basename(abs_path)
        ma = pattern.match(f)
        dataset = ma.group(1)
        if dataset not in ["CIFAR-10","MNIST","FashionMNIST"]:
            continue
        arch = ma.group(2)
        model = construct_model(arch, dataset)
        model.load_state_dict(torch.load(abs_path, map_location=lambda storage, location: storage)["state_dict"])
        model.cuda()
        model.eval()
        data_loader = get_data_loader(dataset)
        acc = validate(data_loader, model)
        print("val_acc:{:.3f}  dataset {}, model {}, path {}".format(acc, dataset,arch, f))