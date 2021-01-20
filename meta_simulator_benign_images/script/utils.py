'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def save_image_logits_pairs(model, train_loader, dump_path, batch_size, max_items):
    model.eval()
    correct = 0
    loss_avg = 0

    process_images = []
    process_logits = []
    process_labels = []
    image_dump_path = dump_path
    label_dump_path = dump_path.replace("images.npy", "logits_labels")

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * batch_size >= max_items:
            break
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        logits = output.detach().cpu().numpy()
        data = data.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        process_images.append(data)
        process_logits.append(logits)
        process_labels.append(target)

    loss_avg /= float(len(train_loader) - 1)

    print('Average Loss: {:4f}%, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss_avg, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    process_images = np.concatenate(process_images, 0)
    assert process_images.ndim == 4
    process_logits = np.concatenate(process_logits, 0).astype(np.float64)
    process_labels = np.concatenate(process_labels, 0).astype(np.int32)
    fp = np.memmap(image_dump_path, dtype='float32', mode='w+', shape=process_images.shape)
    fp[:, :, :, :] = process_images[:, :, :, :]
    del fp
    del process_images
    np.savez(label_dump_path, logits=process_logits, labels=process_labels)