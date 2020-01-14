import glob
import random
import sys

import re

import argparse

from tiny_imagenet import TinyImageNet

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import os
from collections import defaultdict
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from benign_image_classifier.train import get_preprocessor
from config import IN_CHANNELS, CLASS_NUM, IMAGE_SIZE, IMAGE_DATA_ROOT, PY_ROOT
from cifar_models import *
import numpy as np
from torchvision import models as torch_models
from tiny_imagenet_models.densenet import densenet121, densenet161, densenet169, densenet201
from tiny_imagenet_models.inception import inception_v3
from tiny_imagenet_models.miscellaneous import Identity
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d

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
        num_classes = CLASS_NUM[args.dataset]
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



# 一个way是一个分类下以及一种arch下的数据
def generate_labeled_data(datasetname):
    preprocessor = get_preprocessor(IMAGE_SIZE[datasetname], use_flip=False)
    if datasetname == "CIFAR-10":
        train_dataset = CIFAR10(IMAGE_DATA_ROOT[datasetname], train=True, transform=preprocessor)
        # val_dataset = CIFAR10(IMAGE_DATA_ROOT[datasetname], train_simulate_grad_mode=False, transform=preprocessor)
    elif datasetname == "CIFAR-100":
        train_dataset = CIFAR100(IMAGE_DATA_ROOT[datasetname], train=True, transform=preprocessor)
        # val_dataset = CIFAR100(IMAGE_DATA_ROOT[datasetname], train_simulate_grad_mode=False, transform=preprocessor)
    elif datasetname == "MNIST":
        train_dataset = MNIST(IMAGE_DATA_ROOT[datasetname], train=True, transform=preprocessor)
    elif datasetname == "FashionMNIST":
        train_dataset = FashionMNIST(IMAGE_DATA_ROOT[datasetname], train=True, transform=preprocessor)
    elif datasetname == "TinyImageNet":
        train_dataset = TinyImageNet(IMAGE_DATA_ROOT[args.dataset], preprocessor, is_train=True)

    trn_data_dict = defaultdict(list)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=False,
        num_workers=0, pin_memory=True)

    for imgs, labels in data_loader:
        for idx, label in enumerate(labels):
            trn_data_dict[label.item()].append(imgs[idx])
    for label, data_list in trn_data_dict.items():
        if datasetname == "TinyImageNet":
            trn_data_dict[label] = torch.stack(random.sample(data_list, 100))
        else:
            trn_data_dict[label] = torch.stack(data_list)
    return trn_data_dict

def logits_loss(output, labels):
    labels = labels.view(-1, 1)
    first_term = torch.log(torch.gather(output, 1, labels)).squeeze().cuda()
    batch_size = labels.size(0)
    y_onehot = torch.zeros(size=(batch_size, output.size(1))).float().cuda()  # B, CLASS_NUM
    y_onehot.scatter_(dim=1, index=labels, value=1)
    mask = torch.ones_like(y_onehot) - y_onehot
    mask = mask.cuda()
    second_term = torch.log(torch.max(output * mask, dim=1)[0])
    loss = torch.max(first_term-second_term, torch.zeros_like(first_term).cuda()).mean()
    return loss


def pgd_attack(model, images, labels, loss_type="xent", eps=0.3, alpha=2 / 255, iters=40):
    gradients = []
    images_over_steps = []  # T, N, C, H, W
    images = images.cuda()
    labels = labels.cuda()
    if loss_type == "xent":
        loss = nn.CrossEntropyLoss()
    else:
        # output shape = (B, #Class), labels shape = (B, )
        # https://stackoverflow.com/questions/55139801/index-selection-in-case-of-conflict-in-pytorch-argmax/55146371#55146371
        loss = lambda input, target: logits_loss(input, target)
    ori_images = images.clone()
    for i in range(iters):
        images.requires_grad = True
        images_over_steps.append(images.detach().cpu().numpy())
        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs, labels).cuda()
        cost.backward()
        img_grad = images.grad
        adv_images = images + alpha * img_grad.sign()
        gradients.append(img_grad.clone().detach().cpu().numpy())
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0.0, max=1.0).detach_()
    return np.stack(images_over_steps), np.stack(gradients)

def generate_and_save_PGD_data(model, loss_type, arch_name, datasetname, data_dict, save_path, batch_size=100, PGD_iter_num=100):
    for label, img_list in data_dict.items():
        n = batch_size
        chunked_img_list = [img_list[i:i + n] for i in range(0, len(img_list), n)]
        all_images = []
        all_gradients = []
        for imgs in chunked_img_list:
            labels = torch.from_numpy(np.array([label for _ in range(len(imgs))]))
            # images_over_steps = T, B, C, H, W; gradients = T, B, C, H, W
            images_over_steps, gradients = pgd_attack(model, imgs, labels, loss_type, iters=PGD_iter_num)
            all_images.append(np.transpose(images_over_steps, axes=(1,0,2,3,4)))  # B, T, C, H, W
            all_gradients.append(np.transpose(gradients, axes=(1,0,2,3,4)))  # B, T, C, H ,W
        all_images = np.concatenate(all_images, axis=0)  # N, T, C, H, W
        all_gradients = np.concatenate(all_gradients, axis=0)  # N, T, C, H, W
        out_path ="{}/{}/{}/{}_model_{}_label_{}_images.npy".format(save_path, loss_type, datasetname, datasetname, arch_name, label)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fp = np.memmap(out_path, dtype='float32', mode='w+', shape=all_images.shape)
        fp[:, :, :, :, :] = all_images[:, :, :, :, :]
        del fp
        txt_path = "{}/{}/{}/{}_model_{}_label_{}_images.txt".format(save_path, loss_type, datasetname, datasetname, arch_name, label)
        with open(txt_path, "w") as file_obj:
            file_obj.write(str(len(all_images)))
            file_obj.flush()

        out_path = "{}/{}/{}/{}_model_{}_label_{}_grad.npy".format(save_path,  loss_type, datasetname, datasetname, arch_name, label)
        fp = np.memmap(out_path, dtype='float32', mode='w+', shape=all_gradients.shape)
        fp[:, :, :, :, :] = all_gradients[:, :, :, :, :]
        del fp
        print("save to {}".format(out_path))

def get_already_gen_models(loss_type, save_path, datasetname):
    dir_path = "{}/{}/{}/{}_model_*_images.npy".format(save_path, loss_type, datasetname, datasetname)
    pattern = re.compile(".*{}_model_(.*?)_label.*".format(datasetname))
    model_names = set()
    for model_path in glob.glob(dir_path):
        ma = pattern.match(os.path.basename(model_path))
        arch = ma.group(1)
        model_names.add(arch)
    return model_names



def generate_all_models(datasetname, loss_type, PGD_iter_num, batch_size):
    trn_data_dict = generate_labeled_data(datasetname)
    save_path = "{}/data_PGD_40/".format(PY_ROOT)
    os.makedirs(save_path, exist_ok=True)
    already_gen_models = get_already_gen_models(loss_type, save_path, datasetname)
    # model_names = ["conv3", "densenet121","densenet169","googlenet", "mobilenet","mobilenet_v2","resnet18","resnet34","resnet50","resnet101","pnasnetA","efficientnet","dpn26","resnext29_4","resnext29_32","senet18","shufflenet_G2","shufflenet_G3","vgg11","vgg13","vgg16","vgg19","preactresnet18","preactresnet34","preactresnet50","preactresnet101","wideresnet28","wideresnet34","wideresnet40"]
    model_dir_path = "{}/train_pytorch_model/real_image_model/{}*.pth.tar".format(PY_ROOT, datasetname)
    all_model_path_list = glob.glob(model_dir_path)
    model_names = set()
    pattern = re.compile(".*{}@(.*?)@.*".format(datasetname))
    for model_path in all_model_path_list:
        ma = pattern.match(os.path.basename(model_path))
        arch = ma.group(1)
        if arch not in already_gen_models:
            model_names.add(arch)
    for model_name in model_names:
        model = construct_model(model_name, datasetname)
        model = model.cuda()
        model_load_path = "{}/train_pytorch_model/real_image_model/{}@{}@epoch_*.pth.tar".format(PY_ROOT, datasetname, model_name)
        model_load_path = glob.glob(model_load_path)[0]
        assert os.path.exists(model_load_path),model_load_path
        model.load_state_dict(torch.load(model_load_path,map_location=lambda storage, location: storage)["state_dict"])
        model.eval()
        generate_and_save_PGD_data(model, loss_type, model_name, datasetname, trn_data_dict, save_path, batch_size, PGD_iter_num)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--PGD_iter', type=int, default=40)
    parser.add_argument('--loss_type', type=str, default="xent",
                        help='logits loss or xent')
    parser.add_argument("--dataset", type=str, default='CIFAR-10')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument("--batch_size",type=int,default=100)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    generate_all_models(args.dataset, args.loss_type, args.PGD_iter, args.batch_size)