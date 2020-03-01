
import argparse
import random 
import numpy as np
import torch

#Descrption:
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
    parser.add_argument("--max_fintune_iter", type = int, default = 63, help = "maximum finetune iterations")
    parser.add_argument("--max_queries", type=int,default=10000)
    parser.add_argument("--finetune_interval", type = int, default = 3, help = "iteration interval for finetuneing")

    parser.add_argument('--learning_rate', default = 1e-2, type = float, help = 'learning rate')
    parser.add_argument('--update_pixels', default = 125, type = int, help = 'updated pixels every iteration')
    parser.add_argument('--simba_update_pixels', default = 125, type = int, help = 'updated pixels every iteration')
    parser.add_argument("--resize", action="store_true")
    parser.add_argument('--total_number', default = 1000, type = int, help = 'maximum attack numbers')
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument("--istransfer", type = str, default = 'False')

    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--test_batch_size', type = int, default = 1)
    parser.add_argument('--no_cuda', action = 'store_true')

    parser.add_argument("--dataset", choices=["CIFAR-10", "CIFAR-100", "TinyImageNet", "ImageNet"], required=True)
    parser.add_argument("-b", "--binary_steps", type = int, default = 0)
    parser.add_argument("-z", "--use_zvalue", action = 'store_true')
    # parser.add_argument("-r", "--reset_adam", action = 'store_true', help = "reset adam after an initial solution is found")
    parser.add_argument("--use_resize", action = 'store_true', help = "resize image (only works on imagenet!)")
    parser.add_argument("--seed", type = int, default = 1216)
    parser.add_argument("--solver", choices = ["adam", "newton", "adam_newton", "fake_zero"], default = "adam")
    parser.add_argument("--save_ckpts", default = "", help = "path to save checkpoint file")
    parser.add_argument("--start_iter", default = 0, type = int, help = "iteration number for start, useful when loading a checkpoint")
    parser.add_argument("--init_size", default = 32, type = int, help = "starting with this size when --use_resize")

    parser.add_argument('--lr', default = 1e-2, type = int, help = 'learning rate')
    parser.add_argument('--inception', action = 'store_true', default = False)
    parser.add_argument('--use_tanh', action = 'store_true', help="must be set")
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument("--epsilone",type=float,default=4.6)

    args = parser.parse_args()
    assert args.use_tanh is True
    # True False process
    vars(args)['istransfer'] = str2bool(args.istransfer)
    if args.dataset == "ImageNet" and args.targeted:
        args.max_queries = 50000
    #max iteration process

    if args.binary_steps != 0:
        args.init_const = 0.01
    else:
        args.init_const = 0.5

    if args.binary_steps == 0:
        args.binary_steps = 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args



