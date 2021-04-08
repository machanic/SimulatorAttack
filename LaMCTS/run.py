# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys

import json
import os
sys.path.append(os.getcwd())
from LaMCTS.MCTS import MCTS
import os
import argparse
from config import IN_CHANNELS, IMAGE_SIZE, MODELS_TEST_STANDARD, PY_ROOT, CLASS_NUM
import sys
import glog as log
import glob
import os.path as osp
from torch.nn import functional as F
from dataset.defensive_model import DefensiveModel
from dataset.standard_model import StandardModel
import torch
import math
from dataset.dataset_loader_maker import DataLoaderMaker
import numpy as np

class Tracker:
    def __init__(self, foldername):
        self.counter = 0
        self.results = []
        self.curt_best = float("inf")
        self.foldername = foldername
        try:
            os.makedirs(foldername, exist_ok=True)
        except OSError:
            print("Creation of the directory %s failed" % foldername)
        else:
            print("Successfully created the directory %s " % foldername)

    def dump_trace(self):
        trace_path = self.foldername + '/result' + str(len(self.results))
        final_results_str = json.dumps(self.results)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def track(self, result):
        if result < self.curt_best:
            self.curt_best = result
        self.results.append(self.curt_best)
        if len(self.results) % 100 == 0:
            self.dump_trace()

class Attack:
    def __init__(self, image, epsilon, true_labels, target_labels, model, image_height, image_width, in_channels):
        self.model = model
        self.dims  = image_height * image_width * in_channels                   #problem dimensions
        self.channels = in_channels
        self.image_height = image_height
        self.image_width = image_width
        self.lb      =  np.clip(image.reshape(-1) - epsilon, 0, 1.0)         #lower bound for each dimensions
        self.ub      =  np.clip(image.reshape(-1) + epsilon, 0, 1.0)         #upper bound for each dimensions
        self.tracker = Tracker('Attack')      #defined in functions.py
        self.true_labels = true_labels.cuda()
        self.target_labels = target_labels
        if self.target_labels is not None:
            self.target_labels = target_labels.cuda()

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 50
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "linear"
        self.gamma_type = "scale"

    def xent_loss(self, logit, label, target=None):
        if target is not None:
            return -F.cross_entropy(logit, target)
        else:
            return F.cross_entropy(logit, label)

    def cw_loss(self, logit, label, target=None):
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), target]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        logits = self.model(torch.from_numpy(x.reshape(self.channels, self.image_height, self.image_width)).cuda().float())
        loss = self.cw_loss(logits, self.true_labels, self.target_labels).mean()
        self.tracker.track(loss.item())
        return -loss.item()  # 目标函数必须最小化，在MCTS里面最大化


def get_exp_dir_name(dataset,  loss, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'LaMCTS_on_defensive_model-{}-{}-loss-{}-{}'.format(dataset, loss,  norm, target_str)
    else:
        dirname = 'LaMCTS-{}-{}-loss-{}-{}'.format(dataset, loss,  norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

parser = argparse.ArgumentParser(description='Process inputs')
parser.add_argument('--iterations', type=int, default=100, help='specify the iterations to collect in the search')
parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
parser.add_argument('--arch', default=None, type=str, help='network architecture')
parser.add_argument('--test_archs', action="store_true")
parser.add_argument('--targeted', action="store_true")
parser.add_argument('--target_type',type=str, default='increment', choices=['random', 'least_likely',"increment"])
parser.add_argument('--epsilon', type=float, default=8/255.0, help='the linf perturbation bound')
parser.add_argument("--gpu",type=int, required=True)
parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
parser.add_argument('--attack_defense',action="store_true")
parser.add_argument('--defense_model',type=str, default=None)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if __name__ == "__main__":
    if args.test_archs:
        archs = []
        if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(PY_ROOT,
                                                                                        args.dataset,  arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
                else:
                    log.info(test_model_path + " does not exists!")
        elif args.dataset == "TinyImageNet":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(
                    root=PY_ROOT, dataset=args.dataset, arch=arch)
                test_model_path = list(glob.glob(test_model_list_path))
                if test_model_path and os.path.exists(test_model_path[0]):
                    archs.append(arch)
        else:
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
                    PY_ROOT,
                    args.dataset, arch)
                test_model_list_path = list(glob.glob(test_model_list_path))
                if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
                    continue
                archs.append(arch)
    else:
        assert args.arch is not None
        archs = [args.arch]
    args.arch = ", ".join(archs)
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset,  "xent", "linf",
                                                           args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.test_archs:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)

    for arch in archs:
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))

        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, 1)
        for batch_idx, data_tuple in enumerate(dataset_loader):
            if args.dataset == "ImageNet":
                if model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            if args.targeted:
                target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
            else:
                target_labels = None
            if images.size(-1) != model.input_size[-1]:
                images = F.interpolate(images, size=model.input_size[-1], mode='bilinear',align_corners=True)
            images = images.detach().cpu().numpy().squeeze()
            f = Attack(images, args.epsilon, true_labels, target_labels,  model, image_height=IMAGE_SIZE[args.dataset][0],
                       image_width=IMAGE_SIZE[args.dataset][1], in_channels=IN_CHANNELS[args.dataset])
            agent = MCTS(
                lb=f.lb,  # the lower bound of each problem dimensions
                ub=f.ub,  # the upper bound of each problem dimensions
                dims=f.dims,  # the problem dimensions
                ninits=f.ninits,  # the number of random samples used in initializations
                func=f,  # function object to be optimized
                Cp=f.Cp,  # Cp for MCTS
                leaf_size=f.leaf_size,  # tree leaf size
                kernel_type=f.kernel_type,  # SVM configruation
                gamma_type=f.gamma_type  # SVM configruation
             )

            agent.search(iterations = args.iterations)

    """
    FAQ:
    
    1. How to retrieve every f(x) during the search?
    
    During the optimization, the function will create a folder to store the f(x) trace; and
    the name of the folder is in the format of function name + function dimensions, e.g. Ackley10.
    
    Every 100 samples, the function will write a row to a file named results + total samples, e.g. result100 
    mean f(x) trace in the first 100 samples.
    
    Each last row of result file contains the f(x) trace starting from 1th sample -> the current sample.
    Results of previous rows are from previous experiments, as we always append the results from a new experiment
    to the last row.
    
    Here is an example to interpret a row of f(x) trace.
    [5, 3.2, 2.1, ..., 1.1]
    The first sampled f(x) is 5, the second sampled f(x) is 3.2, and the last sampled f(x) is 1.1 
    
    2. How to improve the performance?
    Tune Cp, leaf_size, and improve BO sampler with others.
    
    """
