import numpy as np
import torch
from torch.utils.data import Dataset

from config import IN_CHANNELS, IMAGE_SIZE, PY_ROOT


class AdvImagesDataset(Dataset):
    def __init__(self,  dataset):
        """
        :param root: root path of gradients file
        :param arch: the architecture name
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        """
        self.dataset = dataset
        self.num_channels = IN_CHANNELS[dataset]
        self.data_root_dir = "{}/data_adv_defense/guided_denoiser".format(PY_ROOT)
        self.adv_images_path = "{}/{}_adv_images.npy".format(self.data_root_dir, dataset)
        self.clean_images_path = "{}/{}_clean_images.npy".format(self.data_root_dir, dataset)
        self.labels_path = "{}/{}_labels.npy".format(self.data_root_dir, dataset)
        self.shape_path = "{}/{}_shape.txt".format(self.data_root_dir, dataset)
        self.labels = np.load(self.labels_path)
        with open(self.shape_path, "r") as file_obj:
            self.shape = eval(file_obj.read().strip())

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        with open(self.adv_images_path, "rb") as file_obj:
            adv_image = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                         IMAGE_SIZE[self.dataset][0],
                                                                         IMAGE_SIZE[self.dataset][1]),
                             offset=index * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                    IMAGE_SIZE[self.dataset][1] * 32 // 8)
        with open(self.clean_images_path, "rb") as file_obj:
            clean_image = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                         IMAGE_SIZE[self.dataset][0],
                                                                         IMAGE_SIZE[self.dataset][1]),
                             offset=index * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                    IMAGE_SIZE[self.dataset][1] * 32 // 8)
        label = self.labels[index]
        return torch.from_numpy(clean_image), torch.from_numpy(adv_image), label