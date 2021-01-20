import sys
import os
import os
sys.path.append(os.getcwd())
import argparse
import glob
import os

import os.path as osp
import random

import json
import numpy as np
import torch
from config import PY_ROOT, MODELS_TEST_STANDARD, CLASS_NUM, IMAGE_SIZE
import glog as log
from torch.nn import functional as F
from dataset.defensive_model import DefensiveModel
from dataset.dataset_loader_maker import DataLoaderMaker
from parsimonious_attack.attack import ParsimoniousAttack
from dataset.standard_model import StandardModel


def attack_all_images(dataset_loader, attacker, target_model, args, result_dump_path):
    queries = []
    not_done = []
    correct_all = []
    for batch_idx, data_tuple in enumerate(dataset_loader):
        if args.dataset == "ImageNet":
            if target_model.input_size[-1] >= 299:
                images, true_labels = data_tuple[1], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[2]
        else:
            images, true_labels = data_tuple[0], data_tuple[1]

        if images.size(-1) != target_model.input_size[-1]:
            images = F.interpolate(images, size=target_model.input_size[-1], mode='bilinear', align_corners=True)
        images = images.cuda()
        true_labels = true_labels.cuda()
        if args.targeted:
            if args.target_type == 'random':
                target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                              size=true_labels.size()).long().cuda()
                invalid_target_index = target_labels.eq(true_labels)
                while invalid_target_index.sum().item() > 0:
                    target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                                        size=target_labels[
                                                                            invalid_target_index].shape).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
            elif args.target_type == 'least_likely':
                with torch.no_grad():
                    logits = target_model(images)
                target_labels = logits.argmin(dim=1)
            elif args.target_type == "increment":
                target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
            else:
                raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
        else:
            target_labels = None
        with torch.no_grad():
            logits = target_model(images)
            pred = logits.argmax(dim=1)
            correct = pred.eq(true_labels).detach().cpu().numpy().astype(np.int32)
            correct_all.append(correct)
            if correct[0].item() == 0:
                queries.append(0)
                not_done.append(1)
                log.info("The {}-th image is already classified incorrectly.".format(batch_idx))
                continue

        if args.targeted:
            num_queries, is_success = attacker.perturb(images, target_labels)
        else:
            num_queries, is_success = attacker.perturb(images, true_labels)
        if is_success:
            not_done.append(0)
            queries.append(num_queries)
        else:
            not_done.append(1)
            queries.append(args.max_queries)
        log.info("Attack {}-th image done, attack: {}, query: {}".format(batch_idx, "success" if is_success else "fail", num_queries))

    correct_all = np.concatenate(correct_all, axis=0).astype(np.int32)
    query_all = np.array(queries).astype(np.int32)
    not_done_all = np.array(not_done).astype(np.int32)
    success = (1 - not_done_all) * correct_all
    success_query = success * query_all
    meta_info_dict = {"query_all": query_all.tolist(), "not_done_all": not_done_all.tolist(),
                      "correct_all": correct_all.tolist(),
                      "mean_query": np.mean(success_query[np.nonzero(success)[0]]).item(),
                      "max_query": np.max(success_query[np.nonzero(success)[0]]).item(),
                      "median_query": np.median(success_query[np.nonzero(success)[0]]).item(),
                      "avg_not_done": np.mean(not_done_all[np.nonzero(correct_all)[0]].astype(np.float32)).item(),
                      "args": vars(args)}
    with open(result_dump_path, "w") as result_file_obj:
        json.dump(meta_info_dict, result_file_obj, sort_keys=True)
    log.info("done, write stats info to {}".format(result_dump_path))

def get_exp_dir_name(dataset,  targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'parsimonious_attack-linf_on_defensive_model-{}-{}'.format(dataset,  target_str)
    else:
        dirname = 'parsimonious_attack-linf-{}-{}'.format(dataset, target_str)
    return dirname

def set_log_file(fname):
    import subprocess
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def get_parse_args():
    parser = argparse.ArgumentParser()
    # Experiment Setting
    parser.add_argument('--sample_size', default=100, type=int)
    parser.add_argument("--gpu", type=int, required=True)
    # Attack setting
    parser.add_argument('--loss_func', default='cw', type=str, help='The type of loss function')
    parser.add_argument('--epsilon', default=0.031372, type=float, help='The maximum perturbation')
    parser.add_argument('--max_queries', default=10000, type=int, help='The query limit')
    # Local Search setting
    parser.add_argument('--max_iters', default=1, type=int, help='The number of iterations in local search')
    parser.add_argument('--block_size', default=4, type=int, help='Initial block size')
    parser.add_argument('--batch_size', default=64, type=int, help='The size of batch. No batch if negative')
    parser.add_argument('--no_hier', action='store_true', help='No hierarchical evaluation if true')
    parser.add_argument('--dataset',type=str, required=True)
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args

if __name__ == "__main__":
    args = get_parse_args()
    args.block_size = int(IMAGE_SIZE[args.dataset][0] // 8)
    if args.dataset == "ImageNet":
        args.epsilon = 0.05
        if args.targeted:
            args.max_queries = 50000
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.exp_dir = osp.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.targeted, args.target_type,
                                             args))  # 随机产生一个目录用于实验
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
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT,
                    args.dataset, arch)
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
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    data_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, 1)
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
        attacker = ParsimoniousAttack(model, args)
        attack_all_images(data_loader, attacker, model, args, save_result_path)
        log.info("Attack {} in {} dataset done!".format(arch, args.dataset))
