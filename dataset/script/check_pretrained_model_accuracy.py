import random
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms

from config import pretrained_cifar_model_conf, IMAGE_DATA_ROOT, IMAGE_SIZE
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel


import os
import re
import time
import cifar_models as models
import argparse
import pretrainedmodels
import glog as log
import torch
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

def validate(val_loader, model):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate_accuracy mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            output = output.view(output.size(0),-1)
            # measure accuracy and record loss
            acc1, acc5  = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, top5.avg

def set_log_file(fname):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    import subprocess
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def construct_cifar_model(arch, dataset, num_classes):
    conf = pretrained_cifar_model_conf[dataset][arch]
    arch = arch.split("-")[0].lower()
    if arch.startswith('resnext'):
        model = models.__dict__[arch](
            cardinality=conf["cardinality"],
            num_classes=num_classes,
            depth=conf["depth"],
            widen_factor=conf["widen_factor"],
            dropRate=conf["drop"],
        )
    elif arch.startswith('densenet'):
        model = models.__dict__[arch](
            num_classes=num_classes,
            depth=conf["depth"],
            growthRate=conf["growthRate"],
            compressionRate=conf["compressionRate"],
            dropRate=conf["drop"],
        )
    elif arch.startswith('wrn'):
        model = models.__dict__[arch](
            num_classes=num_classes,
            depth=conf["depth"],
            widen_factor=conf["widen_factor"],
            dropRate=conf["drop"],
        )
    elif arch.endswith('resnet'):
        print(arch, conf["depth"],conf["block_name"] )
        model = models.__dict__[arch](
            num_classes=num_classes,
            depth=conf["depth"],
            block_name=conf["block_name"],
        )
    else:
        model = models.__dict__[arch](num_classes=num_classes)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument("--dir_path",type=str,default="/home1/machen/meta_perturbations_black_box_attack/train_pytorch_model/real_image_model")
    parser.add_argument("--dataset",type=str, required=True)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"

    pattern = re.compile("(.*?)@(.*?)@.*tar")
    args.dir_path = args.dir_path + "/{}-pretrained".format(args.dataset)
    set_log_file(args.dir_path + "/check_{}.log".format(args.dataset))
    log.info("Using GPU {}".format(args.gpu))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    best_acc = 0  # best test accuracy
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    dataset = args.dataset
    if dataset != "ImageNet":
        for arch in os.listdir(args.dir_path ):
            if not os.path.isdir(args.dir_path + "/" +arch):
                continue
            model = StandardModel(dataset, arch, True)
            model.cuda()
            model.eval()
            transform_test = transforms.Compose([
                transforms.Resize(size=(IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1])),
                transforms.ToTensor(),
            ])
            if dataset == "CIFAR-10":
                test_dataset = CIFAR10(IMAGE_DATA_ROOT[dataset], train=False, transform=transform_test)
            elif dataset == "CIFAR-100":
                test_dataset = CIFAR100(IMAGE_DATA_ROOT[dataset], train=False, transform=transform_test)
            data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True,
                                                      num_workers=0)
            acc_top1, acc_top5 = validate(data_loader, model)
            log.info("val_acc_top1:{:.4f}  val_acc_top5:{:.4f} dataset {} in model {}".format(acc_top1, acc_top5,
                                                                                                       dataset, arch))
    else:
        for file_name in os.listdir(args.dir_path + "/checkpoints"):
            arch = file_name.split("-")[0]
            assert arch in pretrainedmodels.__dict__["model_names"], file_name
            log.info("Begin evaluate validation accuracy on {} and {}".format(dataset, arch))
            model = StandardModel(dataset, arch, True)
            data_loader = DataLoaderMaker.get_imagenet_img_label_data_loader(model.cnn, dataset, 200, False)
            acc_top1, acc_top5 = validate(data_loader, model)
            log.info(
                "val_acc_top1:{:.4f}  val_acc_top5:{:.4f} dataset {} in model {}".format(acc_top1, acc_top5,
                                                                                                  dataset, arch))
            model.cpu()