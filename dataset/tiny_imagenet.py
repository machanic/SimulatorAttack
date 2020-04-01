import os

import torchvision.datasets as datasets
from PIL import Image
from torch.utils import data


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class TinyImageNet(data.Dataset):
    def __init__(self, root, transform, train=True, download=None):
        self.is_train = train
        self.category_names = sorted(os.listdir(root + "/train/"))
        self.transform = transform
        if train:
            self.dataset = datasets.ImageFolder(root + "/train/", transform=transform)
        else:
            self.file_name_to_category_id = {}
            with open(root + "/val/val_annotations.txt", "r") as file_obj:
                for line in file_obj:
                    image_file_name, category_name, *_ = line.split()
                    self.file_name_to_category_id[root + "/val/images/{}".format(image_file_name)] \
                        = self.category_names.index(category_name)
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
