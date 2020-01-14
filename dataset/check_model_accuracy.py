import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import os
import re
import time
from dataset_loader_maker import DataLoaderMaker
from model_constructor import ModelConstructor
import argparse
import glob
from cifar_models import *

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
            # measure accuracy and record loss
            acc1,  = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0], input.size(0))
            # top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg



data_loader_cache = {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument("--dir_path",type=str,default="/home1/machen/meta_perturbations_black_box_attack/train_pytorch_model/real_image_model")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    print("using GPU {}".format(args.gpu))
    pattern = re.compile("(.*?)@(.*?)@.*tar")
    for abs_path in glob.glob(args.dir_path + "/*.tar"):
        f = os.path.basename(abs_path)
        ma = pattern.match(f)
        dataset = ma.group(1)
        if dataset not in ["CIFAR-10","MNIST","FashionMNIST"]:
            continue
        arch = ma.group(2)
        if dataset in ["CIFAR-10","MNIST","FashionMNIST"]:
            model = ModelConstructor.construct_cifar_model(arch, dataset)
        elif dataset == "TinyImageNet":
            model = ModelConstructor.construct_tiny_imagenet_model(arch, dataset)
        elif dataset == "ImageNet":
            model = ModelConstructor.construct_imagenet_model(arch)
        model.load_state_dict(torch.load(abs_path, map_location=lambda storage, location: storage)["state_dict"])
        model.cuda()
        model.eval()
        data_loader = DataLoaderMaker.get_img_label_data_loader(dataset, 50, False)
        acc = validate(data_loader, model)
        print("val_acc:{:.3f}  dataset {}, model {}, path {}".format(acc, dataset,arch, f))