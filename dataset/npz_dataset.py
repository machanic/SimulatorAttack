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
            self.images_small = file_data["images_224x224"]
            self.labels = file_data["labels"]
        else:
            self.images = file_data["images"]
            self.labels = file_data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.dataset == "ImageNet":
            # image_big = self.images_big[index]
            image_big =self.images_big[index]
            image_small = self.images_small[index]
            label = self.labels[index]
            return torch.from_numpy(image_small),torch.from_numpy(image_big), label # 只返回224x224的图
        else:
            image = self.images[index]
            label = self.labels[index]
            return torch.from_numpy(image),label


class NpzAliasDataset(NpzDataset):
    def __init__(self, dataset):
        super(NpzAliasDataset, self).__init__(dataset)
        file_path = "{}/attacked_images/{}/{}_images_for_candidate.npz".format(PY_ROOT, dataset, dataset)
        file_data = np.load(file_path)
        self.dataset = dataset
        if dataset == "ImageNet":
            self.images_big = file_data["images_299x299"]
            self.images_small = file_data["images_224x224"]
            self.labels = file_data["labels"]
        else:
            self.images = file_data["images"]
            self.labels = file_data["labels"]

class NpzSubDataset(data.Dataset):
    def __init__(self, dataset, chunk_index, chunk_num):
        file_path = "{}/attacked_images/{}/{}_images.npz".format(PY_ROOT, dataset, dataset)
        file_data = np.load(file_path)
        self.dataset = dataset
        self.labels = file_data["labels"]
        self.orig_dataset_len = len(self.labels)
        each_chunk_size = self.orig_dataset_len // chunk_num
        if dataset == "ImageNet":
            self.images_big = file_data["images_299x299"]
            self.images_small = file_data["images_224x224"]
            self.images_big = self.chunks(self.images_big, each_chunk_size)[chunk_index]
            self.images_small = self.chunks(self.images_small, each_chunk_size)[chunk_index]
        else:
            self.images = file_data["images"]
            self.images = self.chunks(self.images, each_chunk_size)[chunk_index]
        self.labels = self.chunks(self.labels, each_chunk_size)[chunk_index]
        self.subset_pos = self.chunks(np.arange(self.orig_dataset_len), each_chunk_size)[chunk_index].tolist()

    def __len__(self):
        return len(self.labels)

    def chunks(self, l, each_slice_len):
        each_slice_len = max(1, each_slice_len)
        return list(l[i:i + each_slice_len] for i in range(0, len(l), each_slice_len))

    def __getitem__(self, index):
        if self.dataset == "ImageNet":
            # image_big = self.images_big[index]
            image_big =self.images_big[index]
            image_small = self.images_small[index]
            label = self.labels[index]
            return torch.from_numpy(image_small),torch.from_numpy(image_big), label # 只返回224x224的图
        else:
            image = self.images[index]
            label = self.labels[index]
            return torch.from_numpy(image),label
