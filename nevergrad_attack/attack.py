# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import os

import json

sys.path.append(os.getcwd())
import nevergrad as ng
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
from dataset.dataset_loader_maker import DataLoaderMaker
import numpy as np

class NeverGradAttack:
    def __init__(self, image, epsilon, true_labels, target_labels, model, image_height, image_width, in_channels, args):
        self.loss_fn = args.loss
        self.epsilon = epsilon
        self.targeted = args.targeted
        self.discrete = args.discrete
        self.model = model
        self.image = torch.from_numpy(image).cuda().float()
        self.dims    = image_height * image_width * in_channels                   #problem dimensions
        self.channels = in_channels
        self.image_height = image_height
        self.image_width = image_width
        self.true_labels = true_labels.cuda()
        self.target_labels = target_labels
        if self.target_labels is not None:
            self.target_labels = target_labels.cuda()
        if not self.discrete:
            self.lb      =  np.clip(image.reshape(self.image.size(0), -1) - epsilon, 0, 1.0).ravel()        #lower bound for each dimensions
            self.ub      =  np.clip(image.reshape(self.image.size(0), -1) + epsilon, 0, 1.0).ravel()     #upper bound for each dimensions

            init = self.from_unit_cube(np.random.rand(self.image.size(0), self.dims).ravel(), self.lb, self.ub).astype(np.float)
            self.param = ng.p.Array(init=init).set_bounds(self.lb.ravel(), self.ub.ravel(), method='clipping')
            self.optimizer = ng.optimizers.NGOpt(parametrization=self.param, budget=args.iterations)
        else:
            # Discrete, ordered
            variables = ng.p.TransitionChoice([-1,1],repetitions=self.dims)
            self.optimizer = ng.optimizers.NGOpt(parametrization=variables, budget=args.iterations)
            # instrum = ng.p.Instrumentation(*variables)
            # self.optimizer = ng.optimizers.DiscreteOnePlusOne(parametrization=instrum, budget=args.iterations, num_workers=1)
        self.query_all = torch.zeros(self.image.size(0)).float()
        self.not_done_all = None

    def from_unit_cube(self, x, lb, ub):
        """Project from [0, 1]^d to hypercube with bounds lb and ub"""
        assert np.all(lb < ub)
        xx = x * (ub - lb) + lb
        return xx

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
        if not self.discrete:
            assert np.all(x <= self.ub) and np.all(x >= self.lb)
            logits = self.model(torch.from_numpy(x.reshape(-1, self.channels, self.image_height, self.image_width)).cuda().float())
        else:
            x = torch.from_numpy(np.array([xx for xx in x])).cuda().float() * self.epsilon
            image = self.image + x.view_as(self.image)
            image = image.view(-1, self.channels, self.image_height, self.image_width)
            logits = self.model(image).cuda().float()
        adv_pred = logits.argmax(dim=1)
        self.query_all = self.query_all + self.not_done_all.detach().cpu().float()
        if self.targeted:
            self.not_done_all = self.not_done_all * (1 - adv_pred.eq(self.target_labels).float()).float()
        else:
            self.not_done_all = self.not_done_all * adv_pred.eq(self.true_labels).float()
        criterion = self.cw_loss if self.loss_fn == "cw" else self.xent_loss
        loss = criterion(logits, self.true_labels, self.target_labels).mean()
        return -loss.item()

    def attack(self):
        logits = self.model(self.image)
        pred = logits.argmax(dim=1)
        correct = pred.eq(self.true_labels).float()
        self.not_done_all = correct.clone()
        adv_images = self.optimizer.minimize(self).value
        if self.discrete:
            pert = torch.from_numpy(np.array([xx for xx in adv_images])).cuda().float() * self.epsilon
            adv_images = self.image + pert.view_as(self.image)
        else:
            assert np.all(adv_images <= self.ub) and np.all(adv_images >= self.lb)
            adv_images = torch.from_numpy(adv_images.reshape(-1, self.channels, self.image_height, self.image_width)).cuda().float()
        with torch.no_grad():
            adv_logit = self.model(adv_images)
        adv_pred = adv_logit.argmax(dim=1)
        if args.targeted:
            not_done =  1 - adv_pred.eq(self.target_labels).float() # not_done初始化为 correct, shape = (batch_size,)
        else:
            not_done = adv_pred.eq(self.true_labels).float()
        success = (1 - not_done) * correct
        return success.detach().cpu(), correct.detach().cpu()


def get_exp_dir_name(dataset,  loss, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'nevergrad_on_defensive_model-{}-{}-loss-{}-{}'.format(dataset, loss,  norm, target_str)
    else:
        dirname = 'nevergrad-{}-{}-loss-{}-{}'.format(dataset, loss,  norm, target_str)
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
parser.add_argument('--iterations', type=int, default=1000, help='specify the iterations to collect in the search')
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
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument("--loss", type=str, required=True, choices=["xent", "cw"])
parser.add_argument("--discrete", action="store_true")
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
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset,  args.loss, "linf",
                                                           args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.discrete:
        if args.test_archs:
            if args.attack_defense:
                log_file_path = osp.join(args.exp_dir, 'run_discrete_defense_{}.log'.format(args.defense_model))
            else:
                log_file_path = osp.join(args.exp_dir, 'run_discrete.log')
        elif args.arch is not None:
            if args.attack_defense:
                log_file_path = osp.join(args.exp_dir, 'run_discrete_defense_{}_{}.log'.format(args.arch, args.defense_model))
            else:
                log_file_path = osp.join(args.exp_dir, 'run_discrete_{}.log'.format(args.arch))
    else:
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
        if args.discrete:
            if args.attack_defense:
                save_result_path = args.exp_dir + "/{}_{}_discrete_result.json".format(arch, args.defense_model)
            else:
                save_result_path = args.exp_dir + "/{}_discrete_result.json".format(arch)
        else:
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
        dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size)
        success_all = torch.zeros(dataset_loader.dataset.__len__()).float()
        correct_all = torch.zeros(dataset_loader.dataset.__len__()).float()
        query_all = torch.zeros(dataset_loader.dataset.__len__()).float()
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
            images = images.detach().cpu().numpy()
            attacker = NeverGradAttack(images, args.epsilon, true_labels, target_labels,  model,
                                       image_height=IMAGE_SIZE[args.dataset][0], image_width=IMAGE_SIZE[args.dataset][1],
                                       in_channels=IN_CHANNELS[args.dataset],args=args)
            success, correct = attacker.attack()
            selected = torch.arange(batch_idx * args.batch_size, min((batch_idx + 1) * args.batch_size, success_all.size(0)))
            success_all[selected] = success
            correct_all[selected] = correct
            query_all[selected] = attacker.query_all
            log.info("Attack {}-th image {}".format(batch_idx, "success" if success[0].item() ==1 else "fail"))
        avg_query = query_all[(success_all * correct_all).byte()].mean().item()
        median_query = query_all[ (success_all * correct_all).byte()].median().item()
        meta_info_dict = {"avg_correct": correct_all.mean().item(),
                          "avg_not_done": 1 - success_all[correct_all.byte()].mean().item(),
                          "mean_query": query_all[(success_all * correct_all).byte()].mean().item(),
                          "median_query": query_all[(success_all * correct_all).byte()].median().item(),
                          "max_query": query_all[(success_all * correct_all).byte()].max().item(),
                          "correct_all": correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": (np.ones_like(success_all.detach().cpu().numpy().astype(np.int32)) - success_all.detach().cpu().numpy().astype(np.int32)).tolist(),
                          "query_all": query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "args": vars(args)}
        with open(save_result_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(save_result_path))
        log.info("All over! success rate is {}, avg_query is {}, median query is {}".format(success_all[correct_all.byte()].mean(), avg_query, median_query))