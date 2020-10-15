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


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean = mean / float(len(dataset))
    std = std / float(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)




TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def cw_loss(output, target):
    # pdb.set_trace()
    num_classes = output.size(1)
    target_onehot = torch.zeros(target.size() + (num_classes,))
    if torch.cuda.is_available():
        target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = target_onehot.detach()
    target_var.requires_grad = False
    real = (target_var * output).sum(1)
    other = ((torch.ones_like(target_var) - target_var) * output - target_var * 10000.).max(1)[0]
    loss = torch.clamp(torch.log(real + 1e-30) - torch.log(other + 1e-30), min=0.)
    loss = torch.sum(0.5 * loss) / len(target)
    return loss


def save_gradient(model, train_loader, dump_path, batch_size, max_items):
    model.eval()
    correct = 0
    loss_avg = 0

    process_images = []
    process_grads = []
    process_labels = []
    image_dump_path = dump_path
    grad_dump_path = dump_path.replace("images.npy","grads.npy")
    label_dump_path = dump_path.replace("images.npy", "labels.npy")

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * batch_size >= max_items:
            break
        data, target = data.cuda(), target.cuda()

        data.requires_grad_()
        model.zero_grad()

        output = model(data)

        loss = cw_loss(F.softmax(output, dim=1), target)
        grad = torch.autograd.grad(loss, data)[0]

        loss_avg += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        grad = grad.detach().cpu().numpy()
        data = data.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        process_images.append(data)
        process_grads.append(grad)
        process_labels.append(target)

    loss_avg /= float(len(train_loader) - 1)

    print('Average Loss: {:4f}%, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss_avg, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    process_images = np.concatenate(process_images, 0)
    assert process_images.ndim == 4
    process_grads = np.concatenate(process_grads, 0)
    process_labels = np.concatenate(process_labels, 0).astype(np.int32)
    fp = np.memmap(image_dump_path, dtype='float32', mode='w+', shape=process_images.shape)
    fp[:, :, :, :] = process_images[:, :, :, :]
    del fp
    del process_images
    fp = np.memmap(grad_dump_path, dtype='float32', mode='w+', shape=process_grads.shape)
    fp[:, :, :, :] = process_grads[:, :, :, :]
    del fp
    del process_grads
    np.save(label_dump_path, process_labels)