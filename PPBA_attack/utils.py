import torch
import numpy as np
import torchvision.transforms as trans
import math
from scipy.fftpack import dct, idct



# get most likely predictions and probabilities for a set of inputs
def get_preds(model, inputs, correct_class=None, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = inputs[(i * batch_size):upper]
        input_var = input.cuda()
        output = softmax.forward(model.forward(input_var))
        if correct_class is None:
            prob, pred = output.max(1)
        else:
            prob, pred = output[:, correct_class], torch.ones(output.size()) * correct_class
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


# get least likely predictions and probabilities for a set of inputs
def get_least_likely(model, inputs, dataset_name, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = inputs[(i * batch_size):upper]
        input_var = input.cuda()
        output = softmax.forward(model.forward(input_var))
        prob, pred = output.min(1)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


# defines a diagonal order
# order is fixed across diagonals but are randomized across channels and within the diagonal
# e.g.
# [1, 2, 5]
# [3, 4, 8]
# [6, 7, 9]
def diagonal_order(image_size, channels):
    x = torch.arange(0, image_size).cumsum(0)
    order = torch.zeros(image_size, image_size)
    for i in range(image_size):
        order[i, :(image_size - i)] = i + x[i:]
    for i in range(1, image_size):
        reverse = order[image_size - i - 1].index_select(0, torch.LongTensor([i for i in range(i-1, -1, -1)]))
        order[i, (image_size - i):] = image_size * image_size - 1 - reverse
    if channels > 1:
        order_2d = order
        order = torch.zeros(channels, image_size, image_size)
        for i in range(channels):
            order[i, :, :] = 3 * order_2d + i
    return order.view(1, -1).squeeze().long().sort()[1]


# defines a block order, starting with top-left (initial_size x initial_size) submatrix
# expanding by stride rows and columns whenever exhausted
# randomized within the block and across channels
# e.g. (initial_size=2, stride=1)
# [1, 3, 6]
# [2, 4, 9]
# [5, 7, 8]
def block_order(image_size, channels, initial_size=1, stride=1):
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(channels, initial_size, initial_size)
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, :(i+stride), i:(i+stride)] = perm[:num_first].view(channels, -1, stride)
        order[:, i:(i+stride), :i] = perm[num_first:].view(channels, stride, -1)
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]


# zeros all elements outside of the top-left (block_size * ratio) submatrix for every block
def block_zero(x, block_size=8, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = torch.zeros(x.size(0), x.size(1), block_size, block_size)
    mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] * mask
    return z


# applies DCT to each block of size block_size
def block_dct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            submat_dct = dct(dct(submat, axis=2, norm='ortho'), axis=3, norm='ortho')
            if masked:
                submat_dct = submat_dct * mask
            submat_dct = torch.from_numpy(submat_dct)
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = submat_dct
    return z


# applies IDCT to each block of size block_size
def block_idct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
    else:
        mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            if masked:
                submat = submat * mask
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho'))
    return z