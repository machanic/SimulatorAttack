import os
import os.path as osp
import pickle
import sys
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import download_url, check_integrity
from bidict import bidict
import torchvision.datasets as datasets

from dataset_loader_maker import pil_loader


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dirname, target_class_label, phase):
        super(ImageNetDataset, self).__init__()
        assert phase in ['train', 'validation']
        # get 1000 classes
        image_dirname = osp.join(root_dirname, phase)  # e.g., data/imagenet/val
        classes = [d for d in os.listdir(image_dirname) if os.path.isdir(os.path.join(image_dirname, d))]
        classes.sort()
        self.class_to_idx = bidict({classes[i]: i for i in range(len(classes))})
        target_class = self.class_to_idx.inv[target_class_label]

        # get all images
        self.images_fname = glob('{}/{}/*.JPEG'.format(image_dirname,target_class))
        self.images_fname.sort()
        # transform
        self.transform = transforms.Compose([transforms.Resize(size=(299, 299)),
                                             transforms.ToTensor()])

    def __getitem__(self, index):
        # we always emit data in [0, 1] range to keep things simpler (normalization is performed in the attack script).
        # image_id is the identifier of current image in the whole dataset
        # index is the index of current image in self.images_fname
        image_fname = self.images_fname[index]
        label = self.class_to_idx[image_fname.split('/')[-2]]
        with open(image_fname, 'rb') as f:
            with Image.open(f) as image:
                image = image.convert('RGB')
        image = self.transform(image)  # [3, 299, 299]
        return image, label

    def __len__(self):
        return len(self.images_fname)



class CIFAR10Dataset(torch.utils.data.Dataset):
    # Adopted from torchvision
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root_dirname, target_class_label, phase):
        self.root_dirname = os.path.expanduser(root_dirname)
        self.phase = phase  # training set or test set

        # we load CIFAR-10 dataset into standard format (without any data augmentation)
        self.transform = transforms.ToTensor()

        # download if not exists, and check integrity
        self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.phase == 'train':
            self.images = []
            self.labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root_dirname, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.images.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']
                fo.close()

            self.images = np.concatenate(self.images)
            self.images = self.images.reshape((50000, 3, 32, 32))
            self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root_dirname, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.images = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            self.images = self.images.reshape((10000, 3, 32, 32))
            self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC
        self.labels = np.array(self.labels)
        self.image_index_list = np.array([idx for idx,label in enumerate(self.labels) if label == target_class_label])
        self.images = self.images[self.image_index_list]
        self.labels = self.labels[self.image_index_list]


    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_index_list)

    def _check_integrity(self):
        root = self.root_dirname
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root_dirname = self.root_dirname
        download_url(self.url, root_dirname, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root_dirname, self.filename), "r:gz")
        os.chdir(root_dirname)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

class CIFAR100Dataset(CIFAR10Dataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, target_class_label, phase):
        self.is_train = phase == "train"
        self.category_names = sorted(os.listdir(root + "/train/"))
        self.transform = transforms.ToTensor()
        if phase == "train":
            self.dataset = datasets.ImageFolder(root + "/train/", transform=self.transform)
        else:
            self.file_name_to_category_id = {}
            with open(root + "/val/val_annotations.txt", "r") as file_obj:
                for line in file_obj:
                    image_file_name, category_name, *_ = line.split()
                    self.file_name_to_category_id[root + "/val/images/{}".format(image_file_name)] \
                        = self.category_names.index(category_name)
        self.del_image_index_list = [key for key, label in self.file_name_to_category_id.items() if label != target_class_label]
        for image_index in self.del_image_index_list:
            del self.file_name_to_category_id[image_index]

    def __len__(self):
        if self.is_train:
            return len(self.dataset)
        else:
            return len(self.file_name_to_category_id)

    def __getitem__(self, item):
        if self.is_train:
            img, label = self.dataset[item]
        else:
            file_path = list(self.file_name_to_category_id.keys())[item]
            img = pil_loader(file_path)
            img = self.transform(img)
            label = self.file_name_to_category_id[file_path]
        return img, label