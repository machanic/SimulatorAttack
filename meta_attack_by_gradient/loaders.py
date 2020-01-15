import os
import os.path as osp
import sys
import pickle
from PIL import Image
from glob import glob
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_url, check_integrity

from config import IMAGE_DATA_ROOT


class ImageNetIDDataset(torch.utils.data.Dataset):
    def __init__(self, root_dirname, phase, seed, size=224):
        super(ImageNetIDDataset, self).__init__()
        assert phase in ['train', 'val']

        # get 1000 classes
        image_dirname = osp.join(root_dirname, phase)  # e.g., data/imagenet/val
        classes = [d for d in os.listdir(image_dirname) if os.path.isdir(os.path.join(image_dirname, d))]
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        assert len(self.class_to_idx) == 1000

        # get all images
        self.images_fname = glob('{}/*/*.JPEG'.format(image_dirname))
        self.images_fname.sort()
        if phase == 'train_simulate_grad_mode':
            assert len(self.images_fname) == 1281167
        elif phase == 'val':
            assert len(self.images_fname) == 50000
        else:
            raise NotImplementedError('Unknown phase {} for imagenet support phases are: train_simulate_grad_mode/val'.format(phase))

        # record the index of each image in the whole dataset, since we may select a subset later
        self.images_id = np.arange(len(self.images_fname))

        # fix random seed
        # save previous RNG state
        state = np.random.get_state()
        # fix random seed, thus we have the same random status at each time
        np.random.seed(seed)
        perm = np.random.permutation(len(self.images_fname))
        self.images_fname = list(np.array(self.images_fname)[perm])
        self.images_id = list(np.array(self.images_id)[perm])
        # restore previous RNG state for training && test
        np.random.set_state(state)

        # transform
        self.transform = transforms.Compose([transforms.Resize(int(size / 0.875)),
                                             transforms.CenterCrop(size),
                                             transforms.ToTensor()])

    def __getitem__(self, index):
        # we always emit data in [0, 1] range to keep things simpler (normalization is performed in the attack script).
        # image_id is the identifier of current image in the whole dataset
        # index is the index of current image in self.images_fname
        image_id = self.images_id[index]
        image_fname = self.images_fname[index]
        label = self.class_to_idx[image_fname.split('/')[-2]]
        with open(image_fname, 'rb') as f:
            with Image.open(f) as image:
                image = image.convert('RGB')

        # get standard format: 224x224, 0-1 range, RGB channel order
        image = self.transform(image)  # [3, 224, 224]

        return image_id, image, label

    def __len__(self):
        return len(self.images_fname)


class CIFAR10IDDataset(torch.utils.data.Dataset):
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

    def __init__(self, root_dirname, phase, seed):
        self.root_dirname = os.path.expanduser(root_dirname)
        self.phase = phase  # training set or test set
        assert self.phase in ['train', 'test']

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

        # record the index of each image in the whole dataset, since we may select a subset later
        self.images_id = np.arange(len(self.images))

        # fix random seed
        # save previous RNG state
        state = np.random.get_state()
        # fix random seed, thus we have the same random status at each time
        np.random.seed(seed)
        perm = np.random.permutation(len(self.images))
        self.images_id = list(np.array(self.images_id)[perm])
        self.images = list(np.array(self.images)[perm])
        self.labels = list(np.array(self.labels)[perm])
        # restore previous RNG state for training && test
        np.random.set_state(state)

    def __getitem__(self, index):
        image_id, image, label = self.images_id[index], self.images[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        image = self.transform(image)

        return image_id, image, label

    def __len__(self):
        return len(self.images)

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


def make_loader(dataset, phase, batch_size=1, seed=0, **kwargs):
    """
    Make loader. To make sure we use the same data for evaluation,
    these loaders will return (image_id, image, label) tuple instead of vanilla (image, label) tuple.
    :param dataset: mnist, cifar10 or imagenet.
    :param phase: train, val or test.
    :param batch_size: batch size. For imagenet we usually set batch size to 1.
    :param seed: random seed in selecting images.
    :param kwargs: for imagenet, kwargs could contain size (e.g., 224 or 299)
    :return: pytorch DataLoader object, could be used as iterator.
    """

    if dataset == 'ImageNet':
        assert phase in ['train', 'val']
        loader = torch.utils.data.DataLoader(ImageNetIDDataset(IMAGE_DATA_ROOT[dataset], phase, seed, **kwargs),
                                             batch_size=batch_size, num_workers=0)
        loader.num_class = 1000

    elif dataset == 'CIFAR-10':
        assert phase in ['train', 'test']
        loader = torch.utils.data.DataLoader(CIFAR10IDDataset(IMAGE_DATA_ROOT[dataset], phase, seed),
                                             batch_size=batch_size, num_workers=0)
        loader.num_class = 10
    else:
        raise NotImplementedError('Unknown dataset {}, we only support cifar10 and imagenet now'.format(dataset))

    return loader
