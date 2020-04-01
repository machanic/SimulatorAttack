import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import os
import re
import time
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import MetaLearnerModelBuilder
import argparse
import glob
from cifar_models_myself import *
import pretrainedmodels
import glog as log

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument("--dir_path",type=str,default="/home1/machen/meta_perturbations_black_box_attack/train_pytorch_model/real_image_model")
    parser.add_argument("--dataset",type=str, required=True)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/meta_perturbations_black_box_attack/train_pytorch_model/real_image_model/ImageNet-pretrained"

    pattern = re.compile("(.*?)@(.*?)@.*tar")
    set_log_file(args.dir_path + "/check_{}.log".format(args.dataset))
    log.info("using GPU {}".format(args.gpu))
    log.info(args.dir_path + "/{}*.tar".format(args.dataset))

    for abs_path in glob.glob(args.dir_path + "/{}*.tar".format(args.dataset)):
        f = os.path.basename(abs_path)
        ma = pattern.match(f)
        dataset = args.dataset
        arch = ma.group(2)
        if dataset in ["CIFAR-10","CIFAR-100","MNIST","FashionMNIST"]:
            model = MetaLearnerModelBuilder.construct_cifar_model(arch, dataset)
        elif dataset == "TinyImageNet":
            model = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
        elif dataset == "ImageNet":
            if arch not in pretrainedmodels.__dict__:
                print("arch {} not in pretrained models".format(arch))
                continue
            model = MetaLearnerModelBuilder.construct_imagenet_model(arch, dataset)
        if dataset != "ImageNet":
            model.load_state_dict(torch.load(abs_path, map_location=lambda storage, location: storage)["state_dict"])
        model.cuda()
        model.eval()
        data_loader = DataLoaderMaker.get_img_label_data_loader(dataset, 100, False)
        acc = validate(data_loader, model)
        log.info("val_acc:{:.4f}  dataset {}, model {}, path {}".format(acc, dataset, arch, f))