
import argparse
import random

import json
from types import SimpleNamespace

import numpy as np
import torch

#Descrption:
from config import IMAGE_SIZE


def str2bool(s):
    assert s in ['True', 'False']
    if s == 'True':
        return True
    else:
        return False

def get_parse_args():
    parser = argparse.ArgumentParser()
    #important parameters
    parser.add_argument("--maxiter", type = int, default=1000, help = "set 0 to use default value")
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--max_queries", type=int,default=10000)
    parser.add_argument("--finetune_interval", type = int,  help = "iteration interval for finetuneing")

    parser.add_argument('--learning_rate', default = 1e-2, type = float, help = 'learning rate')
    parser.add_argument('--update_pixels', default = 125, type = int, help = 'updated pixels every iteration')
    parser.add_argument('--simba_update_pixels', default = 125, type = int, help = 'updated pixels every iteration')
    parser.add_argument("--resize", action="store_true")
    parser.add_argument('--total_number', default = 1000, type = int, help = 'maximum attack numbers')
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument("--istransfer", type = str, default = 'False')
    parser.add_argument('--json-config', type=str, default='./configures/meta_attack_conf.json', help='a configures file to be passed in instead of arguments')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--test_batch_size', type = int, default = 1)
    parser.add_argument('--no_cuda', action = 'store_true')
    parser.add_argument("--norm", required=True, type=str, choices=["l2","linf"])
    parser.add_argument("--dataset", choices=["CIFAR-10", "CIFAR-100", "TinyImageNet", "ImageNet"], required=True)
    parser.add_argument("-b", "--binary_steps", type = int, default = 0)
    parser.add_argument("-z", "--use_zvalue", action = 'store_true')
    # parser.add_argument("-r", "--reset_adam", action = 'store_true', help = "reset adam after an initial solution is found")
    parser.add_argument("--use_resize", action = 'store_true', help = "resize image (only works on imagenet!)")
    parser.add_argument("--seed", type = int, default = 1216)
    parser.add_argument("--solver", choices = ["adam", "newton", "adam_newton", "fake_zero"], default = "adam")
    parser.add_argument("--start_iter", default = 0, type = int, help = "iteration number for start, useful when loading a checkpoint")
    parser.add_argument("--init_size", default = 32, type = int, help = "starting with this size when --use_resize")

    parser.add_argument('--inception', action = 'store_true', default = False)
    parser.add_argument('--use_tanh', action = 'store_true', help="must be set")
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument("--epsilon",type=float,default=None)
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)

    args = parser.parse_args()
    if args.norm == "linf":
        args.use_tanh = False
    elif args.norm == "l2":
        args.use_tanh = True
    # True False process
    vars(args)['istransfer'] = str2bool(args.istransfer)
    if args.dataset == "ImageNet" and args.targeted:
        args.max_queries = 50000
    args.init_size = IMAGE_SIZE[args.dataset][0]
    #max iteration process
    if args.targeted:
        args.finetune_interval = 3

    if args.binary_steps != 0:
        args.init_const = 0.01
    else:
        args.init_const = 0.5

    if args.binary_steps == 0:
        args.binary_steps = 1

    if args.json_config:
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args



