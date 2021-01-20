"""
Implements SimBA ODS attack
"""
import json
import os
import sys

import random

import numpy as np
sys.path.append(os.getcwd())
import argparse
from types import SimpleNamespace
import glob
from config import IMAGE_SIZE, IN_CHANNELS, CLASS_NUM, MODELS_TEST_STANDARD, PY_ROOT, MODELS_TRAIN_STANDARD, \
    MODELS_TRAIN_WITHOUT_RESNET
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.defensive_model import DefensiveModel
from torch.nn import functional as F
import glog as log
import torch
from dataset.standard_model import StandardModel


class SimBAODS(object):
    def __init__(self, dataset, batch_size, ODS, surrogate_models, freq_dims, stride, order,
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
        self.surrogate_models = surrogate_models
        self.dataset = dataset
        self.norm = norm
        self.ODS = ODS
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.freq_dims = freq_dims
        self.stride = stride
        self.order = order
        self.linf_bound = linf_bound
        self.l2_bound = l2_bound
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
        self.distortion_all = torch.zeros_like(self.query_all)

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

    def cw_loss(self, logits, label, target=None):
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logits.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logits[torch.arange(logits.shape[0]), target]
            second_max_logit = logits[torch.arange(logits.shape[0]), second_max_index]
            return torch.sum(target_logit - second_max_logit)
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logits.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logits[torch.arange(logits.shape[0]), label]
            second_max_logit = logits[torch.arange(logits.shape[0]), second_max_index]
            return torch.sum(second_max_logit - gt_logit)

    def xent_loss(self, logits, label, target=None):
        if target is not None:
            return -F.cross_entropy(logits, target)
        else:
            return F.cross_entropy(logits, label)

    def loss(self, logits, true_labels, target_labels):
        if self.targeted:
            return self.xent_loss(logits, true_labels, target_labels).item()
        else:
            return self.cw_loss(logits, true_labels, target_labels).item()

    def l2_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            delta = new_x - orig
            out_of_bounds_mask = (self.normalize(delta) > eps).float()
            x = (orig + eps * delta / self.normalize(delta)) * out_of_bounds_mask
            x += new_x * (1 - out_of_bounds_mask)
            return x
        return proj

    def linf_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            return orig + torch.clamp(new_x - orig, -eps, eps)
        return proj

    def get_perturbation(self, x, image_size):
        if self.ODS:
            x_grad = x.clone()
            x_grad.requires_grad_()
            random_direction = torch.rand((1, CLASS_NUM[self.dataset])).cuda() * 2 - 1
            ind = np.random.randint(len(self.surrogate_models))
            surrogate_model = self.surrogate_models[ind]
            surrogate_model.cuda()
            with torch.enable_grad():
                loss = (surrogate_model(x_grad) * random_direction).sum()
                loss.backward()
            surrogate_model.cpu()
            perturbation = x_grad.grad.data / x_grad.grad.norm()
        else:
            ind1 = np.random.randint(x.size(1))
            ind2 = np.random.randint(image_size)
            ind3 = np.random.randint(image_size)
            perturbation = torch.zeros(x.size()).cuda()
            perturbation[:,ind1,ind2,ind3] = 1
        return perturbation

    # The SimBA_DCT attack, argument labels is the target labels or true labels
    def attack_batch_images(self, model, images, true_labels, target_labels):
        proj_maker = self.l2_proj if args.norm == 'l2' else self.linf_proj  # 调用proj_maker返回的是一个函数
        if self.norm == "l2":
            proj_step = proj_maker(images, self.l2_bound)
        else:
            proj_step = proj_maker(images, self.linf_bound)
        batch_size = images.size(0)
        assert batch_size == 1
        max_iters = self.max_iters
        x_best = images.clone()
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct = pred.eq(true_labels).float()
        loss_best = self.loss(logits, true_labels, target_labels)
        queries = 1 # https://github.com/ermongroup/ODS/issues/4
        for m in range(max_iters):
            delta = self.get_perturbation(x_best, images.size(-1))
            for sign in [1,-1]:
                x_new = x_best + self.pixel_epsilon * sign * delta
                if self.norm == "linf":
                    x_new = proj_step(x_new)
                x_new = torch.clamp(x_new, 0, 1)
                logits = model(x_new)
                queries += 1
                loss_new = self.loss(logits, true_labels, target_labels)
                if loss_best < loss_new:
                    x_best = x_new
                    loss_best = loss_new
                    break
            adv_pred = logits.argmax(dim=1)
            if args.targeted:
                not_done =  (1 - adv_pred.eq(target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done =  adv_pred.eq(true_labels).float()
            success = (1 - not_done) * correct
            if success[0].item() == 1:
                break
        dist = (x_best - images).norm().item()  # L2 norm distance
        if self.norm == "l2" and dist > self.l2_bound:
            x_best = proj_step(x_best)
            logits = model(x_best)
            adv_pred = logits.argmax(dim=1)
            if args.targeted:
                not_done =  (1 - adv_pred.eq(target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done =  adv_pred.eq(true_labels).float()
            success = (1 - not_done) * correct
        query = torch.zeros(1).float()
        query.fill_(queries)
        distortion = torch.zeros(1).float()
        distortion.fill_(dist)
        return x_best, success.detach().cpu().float(), correct.detach().cpu().float(), query, distortion

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
            adv_images, success, correct, query, distortion = self.attack_batch_images(model, images.cuda(), true_labels, target_labels)
            if query[0].item() > args.max_queries:
                success[0] = 0
            log.info("{}-th image attack over, query:{}, success:{}".format(batch_idx, query, bool(success)))
            success = success * correct
            success_query = success * query
            for key in ['query', 'correct',
                        'success', 'success_query', 'distortion']:
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
                          "avg_distortion": self.distortion_all[self.success_all.byte()].mean().item(),
                          "distortion_all": self.distortion_all.detach().cpu().numpy().astype(np.float32).tolist(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write experimental result information to {}".format(result_dump_path))



def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def get_exp_dir_name(dataset, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    attack_str = "SimBA_ODS_attack"
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
    parser = argparse.ArgumentParser(description='Runs SimBA_ODS on a set of images')
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for parallel runs')
    parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')
    parser.add_argument('--max_queries',type=int,default=10000)
    parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
    parser.add_argument('--linf_bound', type=float,  help='L_inf epsilon bound for L2 norm attack')
    parser.add_argument('--l2_bound', type=float, help='L_2 epsilon bound for L2 norm attack')
    parser.add_argument('--pixel_epsilon', type=float, default=0.2, help='step size per pixel')
    parser.add_argument('--ODS', action='store_true', help='attack in pixel space')
    parser.add_argument('--freq_dims', type=int, help='dimensionality of 2D frequency space')
    parser.add_argument('--order', type=str, default='strided', help='(random) order of coordinate selection')
    parser.add_argument('--stride', type=int, help='stride for block order')
    parser.add_argument('--json-config', type=str,
                        default=os.getcwd()+'/configures/SimBA_attack_conf.json',
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
    args.ODS = True
    args.exp_dir = os.path.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
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
    surrogate_models = []
    train_model_names = MODELS_TRAIN_STANDARD[args.dataset] if not args.attack_defense else MODELS_TRAIN_WITHOUT_RESNET[
        args.dataset]
    for surr_arch in train_model_names:
        if surr_arch in archs:
            continue
        surrogate_model = StandardModel(args.dataset, surr_arch, no_grad=False)
        surrogate_model.eval()
        surrogate_models.append(surrogate_model)
    attacker = SimBAODS(args.dataset, args.batch_size, args.ODS, surrogate_models, args.freq_dims, args.stride, args.order,max_iters,
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