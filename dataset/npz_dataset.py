from torch.utils import data
import torch

from config import PY_ROOT
import numpy as np

class NpzDataset(data.Dataset):
    def __init__(self, dataset):
        file_path = "{}/attacked_images/{}/{}_images.npz".format(PY_ROOT, dataset, dataset)
        file_data = np.load(file_path)
        self.dataset = dataset
        if dataset == "ImageNet":
            self.images_big = file_data["images_299x299"]
            # self.images_small = file_data["images_224x224"]
            self.labels = file_data["labels"]
        else:
            self.images = file_data["images"]
            self.labels = file_data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.dataset == "ImageNet":
            image_big =self.images_big[index]
            # image_small = self.images_small[index]
            label = self.labels[index]
            return torch.from_numpy(image_big), label # 只返回299x299的图
        else:
            image = self.images[index]
            label = self.labels[index]
            return torch.from_numpy(image),label


class NpzImageIdDataset(data.Dataset):
    def __init__(self, dataset):
        file_path = "{}/attacked_images/{}/{}_images.npz".format(PY_ROOT, dataset, dataset)
        file_data = np.load(file_path)
        self.dataset = dataset
        if dataset == "ImageNet":
            self.images_big = file_data["images_299x299"]
            # self.images_small = file_data["images_224x224"]
            self.labels = file_data["labels"]
        else:
            self.images = file_data["images"]
            self.labels = file_data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.dataset == "ImageNet":
            image_big =self.images_big[index]
            # image_small = self.images_small[index]
            label = self.labels[index]
            return index, torch.from_numpy(image_big), label # 只返回299x299的图
        else:
            image = self.images[index]
            label = self.labels[index]
            return index, torch.from_numpy(image),label
