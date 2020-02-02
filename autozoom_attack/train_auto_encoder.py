import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from config import PY_ROOT, IMAGE_SIZE
import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from config import IN_CHANNELS
from optimizer.radam import RAdam
from dataset.dataset_loader_maker import DataLoaderMaker
from autozoom_attack.codec import Codec


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=200, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--dataset", type=str, choices=list(IN_CHANNELS.keys()), default="CIFAR-10",help="CIFAR-10")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate_accuracy', dest='evaluate_accuracy', action='store_true',
                    help='evaluate_accuracy model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument("--total_images", type=int, default=50000)

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_train_worker(args)


def main_train_worker(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.dataset.startswith("CIFAR"):
        compress_mode = 2
        use_tanh=False
        resize = None
        img_size = 32
    if args.dataset == "ImageNet":
        compress_mode = 3
        use_tanh = True
        resize = 128
        img_size = 299
    elif args.dataset in ["MNIST", "FashionMNIST"]:
        compress_mode = 1
        use_tanh = False
        resize = None
        img_size = 28
    network = Codec(img_size, IN_CHANNELS[args.dataset], compress_mode, resize=resize, use_tanh=use_tanh)
    model_path = '{}/train_pytorch_model/AutoZOOM/AutoEncoder_{}@compress_{}@use_tanh_{}@epoch_{}@lr_{}@batch_{}.pth.tar'.format(
       PY_ROOT, args.dataset, compress_mode, use_tanh, args.epochs, args.lr, args.batch_size)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("Model will be saved to {}".format(model_path))
    network.cuda()
    mse_loss_fn = nn.MSELoss().cuda()
    optimizer = RAdam(network.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    train_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, True, (img_size, img_size))
    # val_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, False)

    for epoch in range(0, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        # train_simulate_grad_mode for one epoch
        train(train_loader, network, mse_loss_fn, optimizer, epoch, args, use_tanh)
        # evaluate_accuracy on validation set
        save_checkpoint({
            'epoch': epoch + 1,
            'encoder': network.encoder.state_dict(),
            'decoder': network.decoder.state_dict(),
            "compress_mode": compress_mode,
            "use_tanh": use_tanh,
            'optimizer': optimizer.state_dict(),
        }, filename=model_path)



def train(train_loader, model, criterion, optimizer, epoch, args, use_tanh):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train_simulate_grad_mode mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(train_loader):
        if batch_idx * args.batch_size >= args.total_images:
            break
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        if use_tanh:
            input = atanh((input - 0.5) * 1.99999)
        # compute output
        output = model(input)
        loss = criterion(output, input)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, batch_idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def save_checkpoint(state, filename='traditional_dl.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    main()
