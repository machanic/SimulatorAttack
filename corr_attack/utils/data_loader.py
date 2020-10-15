import torch
import torch.nn as nn
import torchvision.datasets as dset
import torch.utils.data
import torchvision.transforms as transforms
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import glob
from PIL import Image

def imagenet(state):
    if 'defense' in state and state['defense']:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(state['train_path'], transform=transform),
        batch_size=state['batch_size'], shuffle=True,
        num_workers=1)

    test_loader = torch.utils.data.DataLoader(
        dset.ImageFolder(state['data_path'], transform=transform),
        batch_size=state['test_bs'], shuffle=False,
        num_workers=1)

    nlabels = 1000

    return train_loader, test_loader, nlabels, mean, std
