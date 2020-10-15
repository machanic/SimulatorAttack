import os
import argparse
import random
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as tF
from torch.nn import init
import time

from NP_attack.models.neural_process_model import NP

def rgb2ycbcr(im):
    im = np.array(im)
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return tF.to_pil_image(np.uint8(ycbcr)[:, :, [0]])

class RGB2Y(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        return rgb2ycbcr(img)

def get_context_idx(data_dim, device):
    # generate the indeces of the N context points in a flattened image
    idx = np.array(range(0, data_dim))
    idx = torch.tensor(idx, device=device)
    return idx

def generate_grid(h, w, c, device):
    rows = torch.linspace(0, 1, h, device=device)
    cols = torch.linspace(0, 1, w, device=device)
    dim = torch.linspace(0, 1, c, device=device)
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    dim = dim.expand(1, h * w, -1).contiguous().view(1, -1, 1)
    grid = grid.repeat(1, 1, c).view(1, h * w * c, 2)
    grid = torch.cat([grid, dim], dim=-1)
    return grid

def idx_to_y(idx, data):
    # get the [0;1] pixel intensity at each index
    y = torch.index_select(data, dim=1, index=idx)
    return y


def idx_to_x(idx, x_grid, batch_size):
    # From flat idx to 2d coordinates of the 28x28 grid. E.g. 35 -> (1, 7)
    # Equivalent to np.unravel_index()
    x = torch.index_select(x_grid, dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x


def train(model, train_loader, epoch, device, x_grid, optimizer, np_loss, args):
    model.train()
    train_loss = 0
    for batch_idx, (y_all, _) in enumerate(train_loader):
        batch_size = y_all.shape[0]
        if args.dataset == 'mnist':
            y_all = y_all.to(device).view(batch_size, -1, 1)
        elif args.dataset == 'cifar10':
            y_all = y_all.permute(0, 2, 3, 1).to(device).view(batch_size, -1, 1)
        elif args.dataset == 'celeba':
            y_all = y_all.permute().to(device).view(batch_size, -1, c)
        elif args.dataset == 'streetview':
            y_all = y_all.permute().to(device).view(batch_size, -1, c)

        # N = random.randint(1, data_dim)  # number of context points
        context_idx = get_context_idx()

        x_context = x_grid.expand(batch_size, -1, -1)
        # y_context = idx_to_y(context_idx, y_all)
        y_context = y_all

        x_all = x_grid.expand(batch_size, -1, -1)

        optimizer.zero_grad()
        y_hat, z_all, z_context, _, _ = model(x_context, y_context, x_grid, x_all, y_all)
        # print(y_hat[0].shape)

        loss = np_loss(y_hat, y_all, z_all, z_context)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(y_all), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(y_all)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    if os.path.exists(args.log_dir) == False:
        os.makedirs(args.log_dir)
    filename = os.path.join(args.log_dir, 'train.txt')
    with open(filename, 'a') as f:
        f.write('====> Epoch: {} Average loss: {:.4f}\n'.format(
            epoch, train_loss / len(train_loader.dataset)))


def test(model, test_loader, epoch, x_grid, device, data_dim, channel_dim, np_loss, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (y_all, _) in enumerate(test_loader):

            batch_size = y_all.shape[0]
            if args.dataset == 'mnist':
                y_all = y_all.to(device).view(batch_size, -1, 1)
            elif args.dataset == 'cifar10':
                y_all = y_all.to(device).permute(0, 2, 3, 1).to(device).contiguous().view(batch_size, -1, 1)

            # context_idx = get_context_idx()
            x_context = x_grid.expand(batch_size, -1, -1)
            # x_context = idx_to_x(context_idx, batch_size)
            # y_context = idx_to_y(context_idx, y_all)
            y_context = y_all

            y_hat, z_all, z_context, _, _ = model(x_context, y_context, x_grid)
            test_loss += np_loss(y_hat, y_all, z_all, z_context).item()

            if i == 0:  # save PNG of reconstructed examples
                num_examples = min(batch_size, 16)

                recons = []
                context_idx = get_context_idx()
                # x_context = idx_to_x(context_idx, batch_size)
                # y_context = idx_to_y(context_idx, y_all)
                y_hat, _, _, _, _ = model(x_context, y_context, x_grid)
                y_hat = y_hat[0]
                recons = y_hat[:num_examples].view(-1, channel_dim, channel_dim, 3).permute(0, 3, 1, 2)

                background = torch.tensor([0., 0., 1.], device=device)
                background = background.view(1, -1, 1).expand(num_examples, 3, data_dim).contiguous()

                if args.dataset == 'mnist':
                    context_pixels = y_all[:num_examples].view(num_examples, 1, -1)[:, :, context_idx]
                    context_pixels = context_pixels.expand(num_examples, 3, -1)
                    background[:, :, context_idx] = context_pixels
                    comparison = torch.cat([background.view(-1, 3, channel_dim, channel_dim),
                                            recons]) + 0.5
                elif args.dataset == 'cifar10':
                    context_pixels = y_all[:num_examples].view(-1, channel_dim, channel_dim, 3).permute(0, 3, 1, 2)
                    background = context_pixels
                    comparison = torch.cat([background,
                                            recons]) + 0.5
                elif args.dataset == 'celeba':
                    context_pixels = y_all[:num_examples].permute(0, 2, 1)[:, :, context_idx]
                    context_pixels = context_pixels.expand(num_examples, 3, -1)
                    background[:, :, context_idx] = context_pixels
                    comparison = torch.cat([background.view(-1, 3, channel_dim, channel_dim),
                                            recons]) + 0.5
                elif args.dataset == 'streetview':
                    context_pixels = y_all[:num_examples].permute(0, 2, 1)[:, :, context_idx]
                    context_pixels = context_pixels.expand(num_examples, 3, -1)
                    background[:, :, context_idx] = context_pixels
                    comparison = torch.cat([background.view(-1, 3, channel_dim, channel_dim),
                                            recons]) + 0.5
                save_image(comparison.cpu(),
                           args.log_dir + '/%s_ep_' % (args.dataset + '_' + args.att_type) + str(epoch) + '.png',
                           nrow=num_examples)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    filename = os.path.join(args.log_dir, 'test.txt')
    with open(filename, 'a') as f:
        f.write('====> Test set loss: {:.4f}\n'.format(test_loss))


def kl_div_gaussians(mu_q, var_q, mu_p, var_p):
    # var_p = torch.exp(logvar_p)
    logvar_p, logvar_q = torch.log(var_p), torch.log(var_q)
    kl_div = (var_q + (mu_q - mu_p) ** 2) / var_p \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum() / kl_div.shape[0]
    return kl_div


def np_loss(y_hat, y, z_all, z_context):
    y_hat, y_dis = y_hat
    log_p = y_dis.log_prob(y).sum(dim=-1).sum(dim=-1)
    BCE = - log_p.sum() / log_p.shape[0]
    KLD = kl_div_gaussians(z_all[0], z_all[1], z_context[0], z_context[1])
    return BCE + KLD

def get_parse_args():
    parser = argparse.ArgumentParser(description='Neural Processes (NP) for MNIST image completion')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset for training')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--r_dim', type=int, default=128, metavar='N',
                        help='dimension of r, the hidden representation of the context points')
    parser.add_argument('--z_dim', type=int, default=128, metavar='N',
                        help='dimension of z, the global latent variable')
    parser.add_argument('--hidden_dim', type=int, default=400, metavar='N',
                        help='dimension of z, the global latent variable')
    parser.add_argument('--att_type', type=str, default='multihead',
                        help='attention type')
    parser.add_argument('--rep_type', type=str, default='identity',
                        help='representation type')
    parser.add_argument('--restore_model', type=str, default='np_pretrain/cifar_128',
                        help='restore resnet path')
    parser.add_argument('--measurement_type', type=str, default='gaussian',
                        help='the type of measurement matrix A')
    parser.add_argument('--log_dir', type=str, default='./log/cifar_np_multihead_128',
                        help='path to log')
    args = parser.parse_args()
    return args

def main():
    args = get_parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        transform_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5],
                                                                                     std=[1])])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True,
                           transform=transform_),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=False, transform=transform_),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        channel_dim, data_dim = 28, 784
        c = 1
    elif args.dataset == 'cifar10':
        transform_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                     std=[1, 1, 1])])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/home/zengyuyuan/data', train=True, download=True,
                             transform=transform_),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/home/zengyuyuan/data', train=False, transform=transform_),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        channel_dim = 32
        c = 3
        data_dim = channel_dim * channel_dim * 3
    model = NP(c, device, data_dim, args)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x_grid = generate_grid(channel_dim, channel_dim, c, device)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, epoch, device, x_grid, optimizer, np_loss, args)
        test(model, test_loader, epoch, x_grid, device, data_dim, channel_dim, np_loss, args)
        torch.save(model, os.path.join(args.restore_model, args.att_type + '_model.pkl'))
