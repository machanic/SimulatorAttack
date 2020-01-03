import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from config import PY_ROOT
import argparse
import os
import random
import time
from cifar_models import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from config import IMAGE_SIZE
from config import IN_CHANNELS, CLASS_NUM
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from config import IMAGE_DATA_ROOT
from torchvision import transforms

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def get_preprocessor(input_channels=3, input_size=None, use_flip=True):
    # if input_channels == 3:
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    # else:
    #     mean = [0.456]
    #     std = [0.224]
    # normalizer = transforms.Normalize(mean=mean, std=std)

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
        preprocess_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return preprocess_transform



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', type=str, default="conv3")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
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
    if args.arch == "conv3":
        network = Conv3(IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "densenet121":
        network = DenseNet121(IN_CHANNELS[args.dataset], CLASS_NUM[args.dataset])
    elif args.arch == "densenet169":
        network = DenseNet169(IN_CHANNELS[args.dataset], CLASS_NUM[args.dataset])
    elif args.arch == "densenet201":
        network = DenseNet201(IN_CHANNELS[args.dataset], CLASS_NUM[args.dataset])
    elif args.arch == "googlenet":
        network = GoogLeNet(IN_CHANNELS[args.dataset], CLASS_NUM[args.dataset])
    elif args.arch == "mobilenet":
        network = MobileNet(IN_CHANNELS[args.dataset], CLASS_NUM[args.dataset])
    elif args.arch == "mobilenet_v2":
        network = MobileNetV2(IN_CHANNELS[args.dataset], CLASS_NUM[args.dataset])
    elif args.arch == "resnet18":
        network = ResNet18(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "resnet34":
        network = ResNet34(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "resnet50":
        network = ResNet50(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "resnet101":
        network = ResNet101(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "resnet152":
        network = ResNet152(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "pnasnetA":
        network = PNASNetA(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "pnasnetB":
        network = PNASNetB(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "efficientnet":
        network = EfficientNetB0(IN_CHANNELS[args.dataset], CLASS_NUM[args.dataset])
    elif args.arch == "dpn26":
        network = DPN26(IN_CHANNELS[args.dataset], CLASS_NUM[args.dataset])
    elif args.arch == "dpn92":
        network = DPN92(IN_CHANNELS[args.dataset], CLASS_NUM[args.dataset])
    elif args.arch == "resnext29_2":
        network = ResNeXt29_2x64d(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "resnext29_4":
        network = ResNeXt29_4x64d(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "resnext29_8":
        network = ResNeXt29_8x64d(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "resnext29_32":
        network = ResNeXt29_32x4d(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "senet18":
        network = SENet18(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "shufflenet_G2":
        network = ShuffleNetG2(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "shufflenet_G3":
        network = ShuffleNetG3(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "vgg11":
        network = vgg11(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "vgg13":
        network = vgg13(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "vgg16":
        network = vgg16(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "vgg19":
        network = vgg19(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "preactresnet18":
        network = PreActResNet18(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "preactresnet34":
        network = PreActResNet34(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "preactresnet50":
        network = PreActResNet50(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "preactresnet101":
        network = PreActResNet101(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "preactresnet152":
        network = PreActResNet152(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "wideresnet28":
        network = wideresnet28(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "wideresnet34":
        network = wideresnet34(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])
    elif args.arch == "wideresnet40":
        network = wideresnet40(IN_CHANNELS[args.dataset],CLASS_NUM[args.dataset])

    model_path = '{}/train_pytorch_model/real_image_model/{}@{}@epoch_{}@lr_{}@batch_{}.pth.tar'.format(
       PY_ROOT, args.dataset, args.arch, args.epochs, args.lr, args.batch_size)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("after train, model will be saved to {}".format(model_path))
    preprocessor = get_preprocessor(IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset])
    network.cuda()
    image_classifier_loss = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    if args.dataset == "CIFAR-10":
        train_dataset = CIFAR10(IMAGE_DATA_ROOT[args.dataset], train=True, transform=preprocessor)
        val_dataset = CIFAR10(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor)
    elif args.dataset == "MNIST":
        train_dataset = MNIST(IMAGE_DATA_ROOT[args.dataset], train=True, transform=preprocessor)
        val_dataset = MNIST(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor)
    elif args.dataset == "FashionMNIST":
        train_dataset = FashionMNIST(IMAGE_DATA_ROOT[args.dataset], train=True,transform=preprocessor)
        val_dataset = FashionMNIST(IMAGE_DATA_ROOT[args.dataset], train=False, transform=preprocessor)
    elif args.dataset=="SVHN":
        train_dataset = SVHN(IMAGE_DATA_ROOT[args.dataset],  transform=preprocessor)
        val_dataset = SVHN(IMAGE_DATA_ROOT[args.dataset],  transform=preprocessor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train_simulate_grad_mode for one epoch
        train(train_loader, network, image_classifier_loss, optimizer, epoch, args)
        # evaluate_accuracy on validation set
        validate(val_loader, network, image_classifier_loss, args)
        # remember best acc@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
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
    model.train_simulate_grad_mode()

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
    lr = args.lr * (0.1 ** (epoch // 50))
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
