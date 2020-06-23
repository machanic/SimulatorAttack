import glob
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
import json
import os
import os.path as osp
import random
from types import SimpleNamespace
from dataset.standard_model import StandardModel
from dataset.defensive_model import DefensiveModel
import glog as log
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.modules import Upsample
from config import IN_CHANNELS, CLASS_NUM, PY_ROOT, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from utils.statistics_toolkit import success_rate_and_query_coorelation, success_rate_avg_query
from advertorch.attacks import FGSM
import glog as log

# It will also count the queries in white-box attacks
class UAPWhiteboxTargetedAttack(object):
    def __init__(self, args):
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)
        self.delta = args.delta
        self.max_iter = args.max_iter
        self.eps = args.epsilon
        self.norm =args.norm

    def projection(self, values, eps, norm_p):
        """
        Project `values` on the L_p norm ball of size `eps`.
        :param values: Array of perturbations to clip.
        :type values: `np.ndarray`
        :param eps: Maximum norm allowed.
        :type eps: `float`
        :param norm_p: L_p norm to use for clipping. Only 1, 2 and `np.Inf` supported for now.
        :type norm_p: `int`
        :return: Values of `values` after projection.
        :rtype: `np.ndarray`
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        values_tmp = values.reshape((values.shape[0], -1))

        if norm_p == 2:
            values_tmp = values_tmp * np.expand_dims(np.minimum(1., eps / (np.linalg.norm(values_tmp, axis=1) + tol)),
                                                     axis=1)
        elif norm_p == 1:
            values_tmp = values_tmp * np.expand_dims(
                np.minimum(1., eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1)
        elif norm_p == np.inf:
            values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), eps)
        else:
            raise NotImplementedError(
                'Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported.')

        values = values_tmp.reshape(values.shape)
        return values

    ###
    # Different optimization steps
    # All take the form of func(x, g, lr)
    # eg: exponentiated gradients
    # l2/linf: projected gradient descent
    ###
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

    def make_universal_perturbations(self, trn_images, target_labels, model):
        noise = 0
        fooling_rate = 0.0
        targeted_success_rate = 0.0
        nb_instances = len(trn_images)
        attacker = FGSM(model, eps=self.eps, targeted=True)
        pred_y = model(trn_images)
        pred_y_max = pred_y.max(1)[1]
        # Start to generate the adversarial examples
        nb_iter = 0
        while targeted_success_rate < 1. - self.delta and nb_iter < self.max_iter:
            # Go through the data set and compute the perturbation increments sequentially
            rnd_idx = random.sample(range(nb_instances), nb_instances)
            for j,(x_i, y_i) in enumerate(zip(trn_images[rnd_idx], target_labels[rnd_idx])):
                x_i = x_i.unsqueeze(0)
                y_i = y_i.unsqueeze(0)
                current_label =model(x_i + noise).max(1)[1][0].item()
                target_label = y_i[0][j].item()
                if current_label != target_label:
                    # Compute adversarial perturbation
                    adv_xi = attacker.perturb(x_i + noise, y=y_i)
                    new_label = model(adv_xi).max(1)[1]
                    if new_label == target_label:  # If the class has changed, update v
                        noise = adv_xi - x_i
                        noise = self.projection(noise, self.eps, self.norm)
            nb_iter += 1
            # Apply attack and clip
            x_adv = trn_images + noise  # shape is different, broastcast!
            x_adv = torch.clamp(x_adv, 0, 1)
            y_adv = model(x_adv).max(1)[1]
            fooling_rate = 1.0 - torch.sum(pred_y_max.eq(y_adv).float()).item() / nb_instances
            targeted_success_rate = torch.sum(y_adv.eq(target_labels).float()).item() / nb_instances

        self.fooling_rate = fooling_rate
        self.converged = nb_iter < self.max_iter
        self.noise = noise
        log.info('Success rate of universal perturbation attack: {:.2f}'.format(fooling_rate))
        log.info('Targeted success rate of universal perturbation attack: {:.3f}'.format(targeted_success_rate))
        return x_adv, noise




def get_exp_dir_name(dataset, loss, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'UAP_attack_on_defensive_model-{}-{}_loss-{}-{}'.format(dataset, loss, norm, target_str)
    else:
        dirname = 'UAP_attack-{}-{}_loss-{}-{}'.format(dataset, loss, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

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
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--online-lr', type=float, help='Learning rate for the prior')
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument("--loss", type=str, required=True, choices=["xent", "cw"])
    parser.add_argument('--exploration', type=float,
                        help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile-size', type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--json-config', type=str, default='/home1/machen/meta_perturbations_black_box_attack/configures/bandits_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--max_iter', type=int, help="the maximum iterations")
    parser.add_argument('--batch-size', type=int, help='batch size for bandits attack.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random', 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument('--defense_model',type=str, default=None)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"
    print("using GPU {}".format(args.gpu))

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
    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 50000
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.loss, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
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
    attacker = UAPWhiteboxTargetedAttack(args)
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
        attacker.attack_all_images(args, arch, model, save_result_path)
        model.cpu()
