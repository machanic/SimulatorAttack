"""
Implements SimBA attack
"""
import json
import os
import sys

import random

sys.path.append(os.getcwd())

import argparse
from types import SimpleNamespace

import glob
import numpy as np
import torch

from config import IMAGE_SIZE, IN_CHANNELS, CLASS_NUM, MODELS_TEST_STANDARD, PY_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.defensive_model import DefensiveModel
from simba_attack.utils import *
from torch.nn import functional as F
import glog as log

from dataset.standard_model import StandardModel


class SimBA(object):
    def __init__(self, dataset, batch_size, pixel_attack, freq_dims, stride, order,
                 max_iters, targeted, target_type, norm, pixel_epsilon, l2_bound, linf_bound, lower_bound=0.0, upper_bound=1.0):
        """
            :param pixel_epsilon: perturbation limit according to lp-ball
            :param norm: norm for the lp-ball constraint
            :param lower_bound: minimum value data point can take in any coordinate
            :param upper_bound: maximum value data point can take in any coordinate
            :param max_crit_queries: max number of calls to early stopping criterion  per data poinr
        """
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        self.pixel_epsilon = pixel_epsilon
        self.dataset = dataset
        self.norm = norm
        self.pixel_attack = pixel_attack
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.freq_dims = freq_dims
        self.stride = stride
        self.order = order
        self.linf_bound = linf_bound
        self.l2_bound = l2_bound
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)
        self.max_iters = max_iters
        self.targeted = targeted
        self.target_type = target_type

        self.data_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.total_images = len(self.data_loader.dataset)
        self.image_height = IMAGE_SIZE[dataset][0]
        self.image_width = IMAGE_SIZE[dataset][1]
        self.in_channels = IN_CHANNELS[dataset]

        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)

    def get_probs(self, model, x, y):
        output = model(x)
        probs = torch.index_select(torch.nn.Softmax(dim=1)(output), 1, y)
        return torch.diag(probs)

    def expand_vector(self, x, size, image_size):
        batch_size = x.size(0)
        x = x.view(-1, self.in_channels, size, size)
        z = torch.zeros(batch_size, self.in_channels, image_size, image_size)
        z[:, :, :size, :size] = x
        return z

    def trans(self, z, image_size):
        if self.pixel_attack:
            perturbation =  z.cuda()
        else:
            perturbation =  block_idct(z, self.norm, block_size=image_size, linf_bound=self.linf_bound).cuda()
        return perturbation

    # The SimBA_DCT attack, argument labels is the target labels or true labels
    def attack_batch_images(self, model, images, labels):
        batch_size = images.size(0)
        image_size = images.size(2)
        max_iters = self.max_iters
        if self.order == 'rand':
            indices = torch.randperm(self.in_channels * self.freq_dims * self.freq_dims)[:max_iters]
        elif self.order == 'diag':
            indices = diagonal_order(image_size, self.in_channels)[:max_iters]
        elif self.order == 'strided':
            indices = block_order(image_size, self.in_channels, initial_size=self.freq_dims, stride=self.stride)[:max_iters]
        else:
            indices = block_order(image_size, self.in_channels)[:max_iters]
        if self.order == 'rand':
            expand_dims = self.freq_dims
        else:
            expand_dims = image_size
        n_dims = self.in_channels * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims).cuda()
        # logging tensors
        probs = torch.zeros(batch_size, max_iters).cuda()
        success = torch.zeros(batch_size, max_iters)
        queries = torch.zeros(batch_size, max_iters)
        # l2_norms = torch.zeros(batch_size, max_iters)
        # linf_norms = torch.zeros(batch_size, max_iters)
        with torch.no_grad():
            logit = model(images)
        preds = logit.argmax(dim=1)
        prev_probs = self.get_probs(model, images, labels)
        remaining_count = batch_size
        remaining_indices = torch.arange(0, batch_size).long()
        for k in range(max_iters):
            dim = indices[k]
            expanded = (images[remaining_indices] + self.trans(self.expand_vector(x[remaining_indices], expand_dims, image_size),
                                                               image_size)).clamp(0, 1)
            queries_k = torch.zeros(batch_size)
            with torch.no_grad():
                logit = model(expanded)
            # queries_k[remaining_indices] += 1 FIXME see https://github.com/cg563/simple-blackbox-attack/issues/7
            preds_next = logit.argmax(dim=1)
            preds[remaining_indices] = preds_next
            if self.targeted:
                remaining = preds.ne(labels)
            else:
                remaining = preds.eq(labels)
            # check if all images are misclassified and stop early
            if remaining.sum().item() < remaining_count:
                log.info("remaining:{}".format(remaining.sum().item()))
                remaining_count = remaining.sum().item()
            if remaining.sum().item() == 0:
                adv = (images + self.trans(self.expand_vector(x, expand_dims, image_size), image_size)).clamp(0, 1)
                probs_k = self.get_probs(model, adv, labels)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                success[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)  # queries shape = (batch_size, max_iters)
                break
            remaining_indices = torch.arange(0, batch_size)[remaining].long()
            if k > 0:
                success[:, k - 1] = 1 - remaining.detach().cpu().long()
            diff = torch.zeros(remaining.sum().item(), n_dims).cuda()
            diff[:, dim] = self.pixel_epsilon
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            adv = (images[remaining_indices] + self.trans(self.expand_vector(left_vec, expand_dims, image_size), image_size)).clamp(0, 1)
            left_probs = self.get_probs(model, adv, labels[remaining_indices])

            # increase query count for all images
            queries_k[remaining_indices] += 1
            if self.targeted:
                improved = left_probs.gt(prev_probs[remaining_indices]).byte()
            else:
                improved = left_probs.lt(prev_probs[remaining_indices]).byte()
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum().item() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (images[remaining_indices] + self.trans(self.expand_vector(right_vec, expand_dims, image_size),
                                                          image_size)).clamp(0, 1)
            right_probs = self.get_probs(model, adv, labels[remaining_indices])
            if self.targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum().item() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum().item() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
        expanded = (images + self.trans(self.expand_vector(x, expand_dims, image_size), image_size)).clamp(0, 1)
        with torch.no_grad():
            logit = model(expanded)
        preds = logit.argmax(dim=1)
        if self.targeted:
            remaining = preds.ne(labels)
        else:
            remaining = preds.eq(labels)
        success[:, max_iters - 1] = 1 - remaining.long()

        success = success.detach().cpu().numpy().astype(np.uint8)
        queries = queries.sum(1)
        success = np.bitwise_or.reduce(success, axis=1)
        success = torch.from_numpy(success).float()

        adv_images = expanded
        return adv_images, success, queries

    def normalize(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def attack_all_images(self, args, model, result_dump_path):

        for batch_idx, data_tuple in enumerate(self.data_loader):
            if args.dataset == "ImageNet":
                if model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            if model.input_size[-1] == 299:
                self.freq_dims = 33
                self.stride = 7
            elif model.input_size[-1] == 331:
                self.freq_dims = 30
                self.stride = 7
            if images.size(-1) != model.input_size[-1]:
                self.image_width = model.input_size[-1]
                self.image_height = model.input_size[-1]
                images = F.interpolate(images, size=model.input_size[-1], mode='bilinear',align_corners=True)
            if self.targeted:
                if self.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[self.dataset],
                                                               size=target_labels[invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif self.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[self.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(self.target_type))
            else:
                target_labels = None
            selected = torch.arange(batch_idx * args.batch_size,
                                    min((batch_idx + 1) * args.batch_size, self.total_images))
            images = images.cuda()
            true_labels = true_labels.cuda()
            if self.targeted:
                target_labels = target_labels.cuda()
            with torch.no_grad():
                logit = model(images)
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels).float().detach().cpu()
            if self.targeted:
                adv_images, success, query = self.attack_batch_images(model, images.cuda(), target_labels.cuda())
            else:
                adv_images, success, query = self.attack_batch_images(model, images.cuda(), true_labels.cuda())
            delta = adv_images.view_as(images) - images
            # if self.norm == "l2":
            #     l2_out_bounds_mask = (self.normalize(delta) > self.l2_bound).long().view(-1).detach().cpu().numpy()  # epsilon of L2 norm attack = 4.6
            #     l2_out_bounds_indexes = np.where(l2_out_bounds_mask == 1)[0]
            #     if len(l2_out_bounds_indexes) > 0:
            #         success[l2_out_bounds_indexes] = 0

            out_of_bound_indexes = np.where(query.detach().cpu().numpy() > args.max_queries)[0]
            if len(out_of_bound_indexes) > 0:
                success[out_of_bound_indexes] = 0
            log.info("{}-th batch attack over, avg. query:{}".format(batch_idx, query.mean().item()))
            success = success * correct
            success_query = success * query
            for key in ['query', 'correct',
                        'success', 'success_query']:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()

        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "avg_not_done": 1.0 - self.success_all[self.correct_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "not_done_all": (1 - self.success_all.detach().cpu().numpy().astype(np.int32)).tolist(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write experimental result information to {}".format(result_dump_path))



def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def get_exp_dir_name(dataset, pixel_attack, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    attack_str = "SimBA_DCT_attack"
    if pixel_attack:
        attack_str = "SimBA_pixel_attack"
    if args.attack_defense:
        dirname = '{}_on_defensive_model-{}-{}-{}'.format(attack_str, dataset,  norm, target_str)
    else:
        dirname = '{}-{}-{}-{}'.format(attack_str, dataset, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def get_parse_args():
    parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for parallel runs')
    parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')
    parser.add_argument('--max_queries',type=int,default=10000)
    parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
    parser.add_argument('--pixel_epsilon', type=float, default=0.2,  help='step size per pixel')
    parser.add_argument('--linf_bound', type=float,  help='L_inf epsilon bound for L2 norm attack, this option cannot be used with --pixel_attack together')
    parser.add_argument('--l2_bound', type=float, help='L_2 epsilon bound for L2 norm attack')
    parser.add_argument('--freq_dims', type=int, help='dimensionality of 2D frequency space')
    parser.add_argument('--order', type=str, default='strided', help='(random) order of coordinate selection')
    parser.add_argument('--stride', type=int, help='stride for block order')
    parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
    parser.add_argument('--json-config', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/configures/SimBA_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args

if __name__ == "__main__":
    args = get_parse_args()
    args_dict = None
    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
    if args.targeted:
        args.num_iters = 100000
        if args.dataset == "ImageNet":
            args.max_queries = 50000
    if args.norm == "linf":
        assert not args.pixel_attack, "L_inf norm attack cannot be used with --pixel_attack together"
    args.exp_dir = os.path.join(args.exp_dir,
                            get_exp_dir_name(args.dataset,args.pixel_attack, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.test_archs:
        if args.attack_defense:
            log_file_path = os.path.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = os.path.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            log_file_path = os.path.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = os.path.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    if args.order == 'rand':
        n_dims = IN_CHANNELS[args.dataset] * args.freq_dims * args.freq_dims
    else:
        n_dims = IN_CHANNELS[args.dataset] * IMAGE_SIZE[args.dataset][0] * IMAGE_SIZE[args.dataset][1]
    if args.num_iters > 0:
        max_iters = int(min(n_dims, args.num_iters))
    else:
        max_iters = int(n_dims)
    attacker = SimBA(args.dataset, args.batch_size, args.pixel_attack, args.freq_dims, args.stride, args.order,max_iters,
                     args.targeted,args.target_type, args.norm, args.pixel_epsilon, args.l2_bound, args.linf_bound, 0.0, 1.0)
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
        attacker.attack_all_images(args, model, save_result_path)
        model.cpu()