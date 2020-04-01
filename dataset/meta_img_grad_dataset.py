import glob
import os
import random

import numpy as np
import torch
from torch.utils import data
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageFolder

from config import IMAGE_SIZE, IMAGE_DATA_ROOT, MODELS_TRAIN_STANDARD, PY_ROOT, IN_CHANNELS, \
    MODELS_TEST_STANDARD
from constant_enum import SPLIT_DATA_PROTOCOL
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel
from dataset.tiny_imagenet import TinyImageNet


class MetaImgOnlineGradTaskDataset(data.Dataset):
    """
    Support 和query数据用同样的PGD 40 sequence, support数据是指用0~20 PGD的前一半迭代，或指定几个监督信号， query数据是指用20~40的后一半
    """
    def __init__(self, tot_num_tasks, dataset, inner_batch_size, protocol):
        """
        Args:
            num_samples_per_class: num samples to generate "per class" in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.img_size = IMAGE_SIZE[dataset]
        self.dataset = dataset

        if protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II:
            self.model_names = MODELS_TRAIN_STANDARD[self.dataset]
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_II_TEST_I:
            self.model_names = MODELS_TEST_STANDARD[self.dataset]
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
            self.model_names = MODELS_TRAIN_STANDARD[self.dataset] + MODELS_TEST_STANDARD[self.dataset]

        self.model_dict = {}
        for arch in self.model_names:
            if StandardModel.check_arch(arch, dataset):
                model = StandardModel(dataset, arch, no_grad=False).eval()
                if dataset != "ImageNet":
                    model = model.cuda()
                self.model_dict[arch] = model
        is_train = True
        preprocessor = DataLoaderMaker.get_preprocessor(IMAGE_SIZE[dataset], is_train)
        if dataset == "CIFAR-10":
            train_dataset = CIFAR10(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
        elif dataset == "CIFAR-100":
            train_dataset = CIFAR100(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
        elif dataset == "MNIST":
            train_dataset = MNIST(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
        elif dataset == "FashionMNIST":
            train_dataset = FashionMNIST(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
        elif dataset == "TinyImageNet":
            train_dataset = TinyImageNet(IMAGE_DATA_ROOT[dataset], preprocessor, train=is_train)
        elif dataset == "ImageNet":
            preprocessor = DataLoaderMaker.get_preprocessor(IMAGE_SIZE[dataset], is_train, center_crop=True)
            sub_folder = "/train" if is_train else "/validation"  # Note that ImageNet uses pretrainedmodels.utils.TransformImage to apply transformation
            train_dataset = ImageFolder(IMAGE_DATA_ROOT[dataset] + sub_folder, transform=preprocessor)
        self.train_dataset = train_dataset
        self.total_num_images = len(train_dataset)
        self.all_tasks = dict()
        all_images_indexes = np.arange(self.total_num_images).tolist()
        for i in range(tot_num_tasks):
            self.all_tasks[i] = {"image": random.sample(all_images_indexes, inner_batch_size), "arch": random.choice(list(self.model_dict.keys()))}

    def cw_loss(self, logit, label):
        # untargeted cw loss: max_{i\neq y}logit_i - logit_y
        _, argsort = logit.sort(dim=1, descending=True)
        gt_is_max = argsort[:, 0].eq(label).long()
        second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
        gt_logit = logit[torch.arange(logit.shape[0]), label]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return second_max_logit - gt_logit

    def __getitem__(self, task_index):
        data_json = self.all_tasks[task_index]
        arch = data_json["arch"]
        # print("using {}".format(arch))
        image_indexes = data_json["image"]
        images, labels = [], []
        for image_index in image_indexes:
            image, label = self.train_dataset[image_index]
            images.append(image)
            labels.append(label)
        images, labels = torch.stack(images).cuda(), torch.from_numpy(np.array(labels)).cuda().long()
        images.requires_grad_()
        model = self.model_dict[arch].cuda()
        logits = model(images)
        loss = self.cw_loss(logits, labels).mean()
        model.zero_grad()
        loss.backward()
        grad_gt = images.grad
        if self.dataset == "ImageNet":
            self.model_dict[arch].cpu()
        return images.detach(), grad_gt.detach()

    def __len__(self):
        return len(self.all_tasks)


class MetaImgOfflineGradTaskDataset(data.Dataset):

    def __init__(self, tot_num_tasks, dataset, inner_batch_size, protocol):
        """
        Args:
            num_samples_per_class: num samples to generate "per class" in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.img_size = IMAGE_SIZE[dataset]
        self.dataset = dataset

        if protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II:
            self.model_names = MODELS_TRAIN_STANDARD[self.dataset]
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_II_TEST_I:
            self.model_names = MODELS_TEST_STANDARD[self.dataset]
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
            self.model_names = MODELS_TRAIN_STANDARD[self.dataset] + MODELS_TEST_STANDARD[self.dataset]
        data_dir_path = "{}/data_grad_regression/{}/*images.npy".format(PY_ROOT, dataset)
        data_len_dict = {}
        for file_path in glob.glob(data_dir_path):
            file_name = os.path.basename(file_path)
            arch = file_name.split("_")[0]
            if arch in self.model_names:
                shape_file_path = file_path.replace(".npy",".txt")
                with open(shape_file_path, "r") as file_obj:
                    img_shape = eval(file_obj.read().strip())
                    length = img_shape[0]
                data_len_dict[file_path] = length
        self.all_tasks = {}
        for i in range(tot_num_tasks):
            file_path = random.choice(list(data_len_dict.keys()))
            length = data_len_dict[file_path]
            image_indexes = random.sample(np.arange(length).tolist(), inner_batch_size)
            self.all_tasks[i] = {"image_indexes" : image_indexes, "image_file_path": file_path, "grad_file_path": file_path.replace("images.npy","gradients.npy")}

    def __getitem__(self, task_index):
        data_json = self.all_tasks[task_index]
        image_indexes = data_json["image_indexes"]
        image_file_path = data_json["image_file_path"]
        grad_file_path = data_json["grad_file_path"]

        images = []
        grads = []
        with open(image_file_path, "rb") as file_obj:
            for image_index in image_indexes:
                image = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                                  IMAGE_SIZE[self.dataset][0],
                                                                                  IMAGE_SIZE[self.dataset][1]),
                                      offset=image_index * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                             IMAGE_SIZE[self.dataset][1] * 32 // 8).copy()
                image = image.reshape(IN_CHANNELS[self.dataset], IMAGE_SIZE[self.dataset][0],
                                              IMAGE_SIZE[self.dataset][1])
                images.append(image)
        with open(grad_file_path, "rb") as file_obj:
            for image_index in image_indexes:
                grad = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                              IMAGE_SIZE[self.dataset][0],
                                                                              IMAGE_SIZE[self.dataset][1]),
                                  offset=image_index * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                         IMAGE_SIZE[self.dataset][1] * 32 // 8).copy()
                grad = grad.reshape(IN_CHANNELS[self.dataset], IMAGE_SIZE[self.dataset][0],
                                      IMAGE_SIZE[self.dataset][1])
                grads.append(grad)
        images = np.stack(images)
        grads = np.stack(grads)
        images = torch.from_numpy(images)
        grads = torch.from_numpy(grads)
        return images, grads

    def __len__(self):
        return len(self.all_tasks)