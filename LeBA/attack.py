import argparse

import os

import glob
import sys
import os.path as osp
import glog as log
import json
from types import SimpleNamespace

import torch

from LeBA.learnable_black_box_attack import LeBA
from config import MODELS_TEST_STANDARD, PY_ROOT
from dataset.standard_model import StandardModel
from dataset.defensive_model import DefensiveModel

def get_exp_dir_name(dataset, loss, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = '{}_on_defensive_model-{}-{}_loss-{}-{}'.format(args.mode, dataset, loss, norm, target_str)
    else:
        dirname = '{}-{}-{}_loss-{}-{}'.format(args.mode, dataset, loss, norm, target_str)
    return dirname

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def parse_args():
    # SimBA++: python LeBA10.py --mode=simba++ --model1=inception_v3 --model2=resnet152 --input_dir=images --label=labels --out_dir="your output dir"
    parser = argparse.ArgumentParser(description='BA&SA L3 Query Attack')
    parser.add_argument('--arch',required=True, help="Name of victim Model", type=str)
    parser.add_argument('--surrogate_arch',default='resnet152', help="Name of substitute Model",type=str)
    parser.add_argument('--pixel_epsilon', default=0.1, help="Epsilon in Simba Attack part", type=float)
    parser.add_argument('--seed', default=1, help="Random number generate seed", type=int)
    parser.add_argument('--lr', default=0.005, help="Learning rate for train s_model.", type=float)
    parser.add_argument('--FL_rate', default=0.01, help="rate for forward loss", type=float)
    parser.add_argument('--pretrain_weight', help="pretrained weight path for surrogate model", type=str)
    parser.add_argument('--mode',  choices=["SimBA++", "SimBA+", "LeBA","SimBA"], help="train(LeBA) / test(LeBA test mode(SimBA++)) / SimBA++ / SimBA+ / SimBA", type=str)
    parser.add_argument('--batch_size', default=0, help="batch_size, if = 0, compute batch_size with gpu number", type=int)
    parser.add_argument('--ba_num', default=10, help="iterations for TIMI attack", type=int)
    parser.add_argument('--ba_interval', default=20, help="interval for TIMI attack", type=int)
    parser.add_argument('--epsilon', default=16.37, help="max perturbation (L2 norm)", type=float)
    parser.add_argument('--out_dir', default='out', help="output dir", type=str)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataaset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument("--gpu", type=int, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"

    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
        args_dict = defaults
    if args.targeted and args.dataset == "ImageNet":
        args.max_queries = 50000
    args.exp_dir = os.path.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.loss, args.norm, args.targeted, args.target_type,
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
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    surrogate_model = StandardModel(args.dataset, args.surrogate_arch,no_grad=False,load_pretrained=True)

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
        if args.pretrain_weight is not None:
            surrogate_model.load_state_dict(torch.load(args.pretrain_weight))
        attacker = LeBA(args.dataset, model, surrogate_model, args.surrogate_arch, args.mode, args.pretrain_weight,
                        args.epsilon, args.ba_num, args.ba_interval,
                        args.batch_size, args.targeted, args.target_type, args.pixel_epsilon, args.norm, args.lr,
                        args.FL_rate, 0, 1, args.max_queries)
        attacker.attack()
        attacker.save_result(save_result_path, args)

    log.info("All done!")

