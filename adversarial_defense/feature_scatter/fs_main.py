'''Train Adversarially Robust Models with Feature Scattering'''
from __future__ import print_function

import math
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from torch import nn
import argparse
import os
import time

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import torch
import utils
from adversarial_defense.feature_scatter.attack_methods import Attack_FeaScatter
from adversarial_defense.feature_scatter.utils import softCrossEntropy
from config import PY_ROOT, CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel

torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Feature Scatterring Training')

# add type keyword to registries
parser.register('type', 'bool', utils.str2bool)
parser.add_argument("--gpu",type=str,required=True)
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--adv_mode',
                    default='feature_scatter',
                    type=str,
                    help='adv_mode (feature_scatter)')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass (-1: from scratch; K: checkpoint-K)')
parser.add_argument('--max_epoch',
                    default=200,
                    type=int,
                    help='max number of epochs')
parser.add_argument('--save_epochs', default=100, type=int, help='save period')
parser.add_argument('--decay_epoch1',
                    default=60,
                    type=int,
                    help='learning rate decay epoch one')
parser.add_argument('--decay_epoch2',
                    default=90,
                    type=int,
                    help='learning rate decay point two')
parser.add_argument('--decay_rate',
                    default=0.1,
                    type=float,
                    help='learning rate decay rate')
parser.add_argument('--batch_size_train',
                    default=30,
                    type=int,
                    help='batch size for training')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='momentum (1-tf.momentum)')
parser.add_argument('--weight_decay',
                    default=2e-4,
                    type=float,
                    help='weight decay')
parser.add_argument('--log_step', default=10, type=int, help='log_step')

# number of classes and image size will be updated below based on the dataset
parser.add_argument('--num_classes', type=int, help='num classes')
parser.add_argument('--image_size', type=int, help='image size')
parser.add_argument('--dataset', required=True, type=str, help='dataset')  # concat cascade
parser.add_argument('-a', '--arch', type=str, required=True, help="The arch used to generate adversarial images for testing")
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


if args.dataset == 'CIFAR-10':
    print('------------cifar10---------')
    args.num_classes = CLASS_NUM[args.dataset]
    args.image_size = 32
elif args.dataset == 'CIFAR-100':
    print('----------cifar100---------')
    args.num_classes = CLASS_NUM[args.dataset]
    args.image_size = 32
if args.dataset == 'SVHN':
    print('------------svhn10---------')
    args.num_classes = 10
    args.image_size = 32
if args.dataset == 'TinyImageNet':
    print('------------Tiny ImageNet---------')
    args.num_classes = CLASS_NUM[args.dataset]
    args.image_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0
model_path = '{}/train_pytorch_model/adversarial_train/feature_scatter/{}@{}@epoch_{}@batch_{}.pth.tar'.format(
        PY_ROOT, args.dataset, args.arch, args.max_epoch, args.batch_size_train)
print("model will be saved to {}".format(model_path))
os.makedirs(os.path.dirname(model_path),exist_ok=True)
# Data
print('==> Preparing data..')
train_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size_train, True)


print('==> Building model..')
basic_net = StandardModel(args.dataset, args.arch, no_grad=False).train().cuda()
basic_net.apply(initialize_weights)
def print_para(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)
        break



# config for feature scatter
config_feature_scatter = {
    'train': True,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 1,
    'step_size': 8.0 / 255 * 2,
    'random_start': True,
    'ls_factor': 0.5,
}

if args.adv_mode.lower() == 'feature_scatter':
    print('-----Feature Scatter mode -----')
    net = Attack_FeaScatter(basic_net, config_feature_scatter)
else:
    print('-----OTHER_ALGO mode -----')
    raise NotImplementedError("Please implement this algorithm first!")

if device == 'cuda':
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

# if args.resume and args.init_model_pass != '-1':
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     f_path_latest = os.path.join(args.model_dir, 'latest')
#     f_path = os.path.join(args.model_dir,
#                           ('checkpoint-%s' % args.init_model_pass))
#     if not os.path.isdir(args.model_dir):
#         print('train from scratch: no checkpoint directory or file found')
#     elif args.init_model_pass == 'latest' and os.path.isfile(f_path_latest):
#         checkpoint = torch.load(f_path_latest)
#         net.load_state_dict(checkpoint['net'])
#         start_epoch = checkpoint['epoch'] + 1
#         print('resuming from epoch %s in latest' % start_epoch)
#     elif os.path.isfile(f_path):
#         checkpoint = torch.load(f_path)
#         net.load_state_dict(checkpoint['net'])
#         start_epoch = checkpoint['epoch'] + 1
#         print('resuming from epoch %s' % (start_epoch - 1))
#     elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
#         print('train from scratch: no checkpoint directory or file found')

soft_xent_loss = softCrossEntropy()


def train_fun(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # update learning rate
    if epoch < args.decay_epoch1:
        lr = args.lr
    elif epoch < args.decay_epoch2:
        lr = args.lr * args.decay_rate
    else:
        lr = args.lr * args.decay_rate * args.decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    def get_acc(outputs, targets):
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        acc = 1.0 * correct / total
        return acc

    iterator = tqdm(train_loader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        adv_acc = 0

        optimizer.zero_grad()

        # forward
        outputs, loss_fs = net(inputs.detach(), targets)

        optimizer.zero_grad()
        loss = loss_fs.mean()
        loss.backward()

        optimizer.step()

        train_loss = loss.item()

        duration = time.time() - start_time
        if batch_idx % args.log_step == 0:
            if adv_acc == 0:
                adv_acc = get_acc(outputs, targets)
            iterator.set_description(str(adv_acc))

            nat_outputs, _ = net(inputs, targets, attack=False)
            nat_acc = get_acc(nat_outputs, targets)

            print(
                "epoch %d, step %d, lr %.4f, duration %.2f, training nat acc %.2f, training adv acc %.2f, training adv loss %.4f"
                % (epoch, batch_idx, lr, duration, 100 * nat_acc,
                   100 * adv_acc, train_loss))
    if epoch >= 0:
        print('Saving latest @ epoch %s..' % (epoch))
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }

        torch.save(state, model_path)


for epoch in range(start_epoch, args.max_epoch):
    train_fun(epoch, net)
