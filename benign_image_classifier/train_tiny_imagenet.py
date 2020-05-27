import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d

from config import PY_ROOT, CLASS_NUM
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
from config import IMAGE_SIZE
from config import IN_CHANNELS
from config import IMAGE_DATA_ROOT
from torchvision import transforms
import torchvision.models as models
from dataset.tiny_imagenet import TinyImageNet
from tiny_imagenet_models.densenet import densenet121,densenet161,densenet169,densenet201
from tiny_imagenet_models.inception import inception_v3

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def get_preprocessor(input_size=None, use_flip=True):
    if input_size is not None:
        if use_flip:
            preprocess_transform = transforms.Compose([
                transforms.Resize(size=input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            preprocess_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    else:
        if use_flip:
            preprocess_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            preprocess_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    return preprocess_transform


model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=200, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--dataset", type=str, choices=list(IN_CHANNELS.keys()), default="TinyImageNet")
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
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')


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
    print("=> creating model '{}'".format(args.arch))
    if args.arch in models.__dict__:
        network = models.__dict__[args.arch](pretrained=True)
    num_classes = CLASS_NUM[args.dataset]
    if args.arch.startswith("resnet"):
        num_ftrs = network.fc.in_features
        network.fc = nn.Linear(num_ftrs, num_classes)
    elif args.arch.startswith("densenet"):
        if args.arch == "densenet161":
            network = densenet161(pretrained=True)
        elif args.arch == "densenet121":
            network = densenet121(pretrained=True)
        elif args.arch == "densenet169":
            network = densenet169(pretrained=True)
        elif args.arch == "densenet201":
            network = densenet201(pretrained=True)
    elif args.arch == "resnext32_4":
        network = resnext101_32x4d(pretrained=None)
    elif args.arch == "resnext64_4":
        network = resnext101_64x4d(pretrained=None)
    elif args.arch == "resnext32_4":
        network = resnext101_32x4d(pretrained="imagenet")
    elif args.arch == "resnext64_4":
        network = resnext101_64x4d(pretrained="imagenet")
    elif args.arch.startswith("squeezenet"):
        network.classifier[-1] = nn.AdaptiveAvgPool2d(1)
        network.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif args.arch.startswith("inception"):
        network = inception_v3(pretrained=True)
    elif args.arch.startswith("vgg"):
        network.avgpool = Identity()
        network.classifier[0] = nn.Linear(512 * 2 * 2, 4096)  # 64 /2**5 = 2
        network.classifier[-1] = nn.Linear(4096, num_classes)

   # densenet和inception必须自己改一份新代码，因为forward用了F.avg_pool2d
    model_path = '{}/train_pytorch_model/real_image_model/{}@{}@epoch_{}@lr_{}@batch_{}.pth.tar'.format(
       PY_ROOT, args.dataset, args.arch, args.epochs, args.lr, args.batch_size)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("after train_simulate_grad_mode, model will be saved to {}".format(model_path))
    preprocessor = get_preprocessor(IMAGE_SIZE[args.dataset], use_flip=True)
    network.cuda()
    image_classifier_loss = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    train_dataset = TinyImageNet(IMAGE_DATA_ROOT[args.dataset], preprocessor, train=True)
    test_dataset = TinyImageNet(IMAGE_DATA_ROOT[args.dataset], preprocessor, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train_simulate_grad_mode for one epoch
        train(train_loader, network, image_classifier_loss, optimizer, epoch, args)
        # evaluate_accuracy on validation set
        val_acc = validate(val_loader, network, image_classifier_loss, args)
        # remember best acc@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            "val_acc": val_acc,
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=model_path)



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to train_simulate_grad_mode mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1,  = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate_accuracy mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1,  = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            # top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
        print('Validate Set Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
    return top1.avg

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
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
