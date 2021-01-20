import argparse
import os
import random
import sys
sys.path.append(os.getcwd())
from types import SimpleNamespace

import glob
import glog as log
import json
import numpy as np
import torch
from scipy.fftpack import dct, idct
from torchvision import models
from torchvision.utils import save_image
from PPBA_attack.utils import block_order,block_idct
from config import IMAGE_SIZE, IN_CHANNELS, CLASS_NUM, MODELS_TEST_STANDARD, PY_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from torch.nn import functional as F

from dataset.defensive_model import DefensiveModel
from dataset.standard_model import StandardModel


class PPBA(object):
    def __init__(self, dataset, order, r, rho, mom,n_samples,
                 targeted, target_type,  norm, epsilon, low_dim, lower_bound=0.0, upper_bound=1.0,
                 max_queries=10000):
        """
            :param epsilon: perturbation limit according to lp-ball
            :param norm: norm for the lp-ball constraint
            :param lower_bound: minimum value data point can take in any coordinate
            :param upper_bound: maximum value data point can take in any coordinate
            :param max_queries: max number of calls to model per data point
            :param max_crit_queries: max number of calls to early stopping criterion  per data poinr
        """
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        self.epsilon = epsilon
        self.norm = norm
        self.max_queries = max_queries
        self.order = order
        self.r = r

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)

        self.targeted = targeted
        self.target_type = target_type
        self.dataset = dataset
        self.data_loader = DataLoaderMaker.get_test_attacked_data(dataset, 1)
        self.total_images = len(self.data_loader.dataset)
        self.image_height = IMAGE_SIZE[dataset][0]
        self.image_width = IMAGE_SIZE[dataset][1]
        self.in_channels = IN_CHANNELS[dataset]
        self.low_dim = low_dim
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)
        if dataset.startswith("CIFAR"):  # TODO 实验不同的参数效果
            self.freq_dim = 11 # 28
            self.stride = 7
        elif dataset=="TinyImageNet":
            self.freq_dim = 15
            self.stride = 7
        elif dataset == "ImageNet":
            self.freq_dim = 28
            self.stride = 7
        self.mom =mom  # default 1 not add
        self.n_samples = n_samples # number of samples per iteration (1 by default), not the number of images to be evaluated.
        self.rho = rho
        self.construct_random_matrix()

    def construct_random_matrix(self):
        if self.order == "strided":
            self.random_matrix = np.zeros((self.low_dim, self.in_channels * self.image_height * self.image_width))
            indices = block_order(self.image_height, self.in_channels, initial_size=self.freq_dim, stride=self.stride)
        else:
            self.random_matrix = np.zeros((self.low_dim, self.in_channels * self.freq_dim * self.freq_dim))
            indices = random.sample(range(self.r), self.low_dim)
        for i in range(self.low_dim):
            self.random_matrix[i][indices[i]] = 1

        if self.order == "strided":
            self.random_matrix = torch.from_numpy(self.random_matrix).view(-1, self.in_channels, self.image_height, self.image_width)
        else:
            self.random_matrix = torch.from_numpy(self.random_matrix).view(-1, self.in_channels, self.freq_dim, self.freq_dim)
        self.random_matrix = block_idct(self.expand_vector(
            self.random_matrix, size=(self.image_height if self.order == "strided" else self.freq_dim),),
            block_size=self.image_height).view(self.low_dim, -1).cuda()


    def cw_loss(self, model, images, label, target=None):
        logits = model(images)
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logits.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logits[torch.arange(logits.shape[0]), target]
            second_max_logit = logits[torch.arange(logits.shape[0]), second_max_index]
            return torch.clamp(second_max_logit - target_logit, min=0)
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logits.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logits[torch.arange(logits.shape[0]), label]
            second_max_logit = logits[torch.arange(logits.shape[0]), second_max_index]
            return torch.clamp(gt_logit - second_max_logit, min=0)

    # def cw_loss(self, model, images, true_label, target_label, targeted=False):
    #     outputs = model(images)
    #     one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()
    #     # 设置targeted attack的类别从这里开始改
    #     i, _ = torch.max((torch.ones_like(one_hot_labels).cuda() - one_hot_labels) * outputs, dim=1)
    #     j, _ = torch.max((one_hot_labels) * outputs, dim=1)
    #     if targeted:
    #         return torch.clamp(i - j, min=0)
    #     else:
    #         return torch.clamp(j - i, min=0)

    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, self.in_channels, size, size)
        z = torch.zeros(batch_size, self.in_channels, self.image_height, self.image_width)
        z[:, :, :size, :size] = x
        return z

    def get_probability(self, success_probability):
        probability = [v / sum(success_probability) for v in success_probability]
        return probability

    def func(self, model, orig_images, xs, true_labels, target_labels):
        z = torch.from_numpy(xs).float().cuda().view(-1, self.low_dim)
        perturbation = (z @ self.random_matrix).view(z.shape[0], self.in_channels, self.image_height, self.image_width)  # matrix multiplication
        if self.norm == "linf":
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        new_image = (orig_images + perturbation).clamp(0, 1)
        loss = self.cw_loss(model, new_image, true_labels, target_labels)
        return loss

    # only supports 1 image to be passed in
    def attack_batch_images(self, model, batch_index, images, true_labels):
        batch_size = images.size(0)
        query = torch.zeros(batch_size).cuda()
        with torch.no_grad():
            logit = model(images)
        pred = logit.argmax(dim=1)
        correct = pred.eq(true_labels).float()
        not_done = correct.clone()
        selected = torch.arange(batch_index * batch_size, min((batch_index + 1) * batch_size, self.total_images))

        if self.targeted:
            if self.target_type == 'random':
                target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset], size=true_labels.size()).long().cuda()
                invalid_target_index = target_labels.eq(true_labels)
                while invalid_target_index.sum().item() > 0:
                    target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                 size=target_labels[invalid_target_index].shape).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
            elif self.target_type == 'least_likely':
                target_labels = logit.argmin(dim=1)
            elif self.target_type == "increment":
                target_labels = torch.fmod(true_labels + 1, CLASS_NUM[self.dataset])
            else:
                raise NotImplementedError('Unknown target_type: {}'.format(self.target_type))
        else:
            target_labels = None

        z = np.zeros((1,self.low_dim))
        prev_f = self.func(model, images, z, true_labels, target_labels)[0].item()
        query += not_done
        is_success = 0 if prev_f > 0 else 1
        effective_number = [
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
        ]
        ineffective_number = [
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
        ]
        for k in range(self.max_queries-1):
            u = np.zeros((self.n_samples, self.low_dim))
            r = np.random.uniform(size=(self.n_samples, self.low_dim))
            # 这里 u 会为最终采样出来的方向，由于 $u \in [-\rho, 0, \rho]^n$, 在计算每个值出现的概率之后，通过 r 均匀采样，来确定 u 最终的值.
            # (比如初始三者都是等概率,即分成0-0.33,0.33-0.66,0.66-1三个区间，r随机得到0.6落在第二个区间，则该位取 0)
            effective_probability = [
                effective_number[i] / (effective_number[i] + ineffective_number[i])
                for i in range(len(effective_number))
            ]
            probability = self.get_probability(effective_probability)
            u[r < probability[0]] = -1
            u[r >= probability[0] + probability[1]] = 1
            uz = z + self.rho * u
            if self.norm == "l2":
                uz_l2 = np.linalg.norm(uz, axis=1)
                uz = uz * np.minimum(1, self.epsilon / uz_l2).reshape(-1, 1)
            fu = self.func(model, images, uz, true_labels, target_labels).detach().cpu().numpy()
            query += not_done
            if fu.min().item() < prev_f:
                worked_u = u[fu < prev_f]
                effective_probability[0] = effective_probability[0] * self.mom + (worked_u == -1).sum(0)
                effective_probability[1] = effective_probability[1] * self.mom + (worked_u == 0).sum(0)
                effective_probability[2] = effective_probability[2] * self.mom + (worked_u == 1).sum(0)
                not_worked_u = u[fu >= prev_f]
                ineffective_number[0] = ineffective_number[0] * self.mom + (not_worked_u == -1).sum(0)
                ineffective_number[1] = ineffective_number[1] * self.mom + (not_worked_u == 0).sum(0)
                ineffective_number[2] = ineffective_number[2] * self.mom + (not_worked_u == 1).sum(0)
                z = uz[np.argmin(fu).item()]
                prev_f = fu.min().item()
            else:
                ineffective_number[0] += (u == -1).sum(0)
                ineffective_number[1] += (u == 0).sum(0)
                ineffective_number[2] += (u == 1).sum(0)
            if prev_f <= 0:
                is_success = 1
                break

        z = torch.from_numpy(z).float().cuda()
        perturbation = (z @ self.random_matrix).view(1, self.in_channels, self.image_height, self.image_width)
        if self.norm == "linf":
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        adv_images = (images + perturbation).clamp(0, 1)
        with torch.no_grad():
            adv_logit = model(adv_images)
        adv_pred = adv_logit.argmax(dim=1)
        adv_prob = F.softmax(adv_logit, dim=1)
        if self.targeted:
            not_done = not_done * (1 - adv_pred.eq(target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
        else:
            not_done = not_done * adv_pred.eq(true_labels).float()
        success = (1 - not_done) * correct
        success = success * (query <= self.max_queries).float()
        success_query = success * query
        not_done_prob = adv_prob[torch.arange(batch_size), true_labels] * not_done
        is_success = int(success[0].item()) & is_success
        log.info("{}-th image attack success: {} query: {}".format(batch_index, bool(is_success), int(query[0].item())))
        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()

    def attack_all_images(self, args, model, tmp_dump_path, result_dump_path):

        for batch_idx, data_tuple in enumerate(self.data_loader):
            if os.path.exists(tmp_dump_path):
                with open(tmp_dump_path, "r") as file_obj:
                    json_content = json.load(file_obj)
                    resume_batch_idx = int(json_content["batch_idx"])  # resume
                    for key in ['query_all', 'correct_all', 'not_done_all',
                                'success_all', 'success_query_all','not_done_prob_all']:
                        if key in json_content:
                            setattr(self, key, torch.from_numpy(np.asarray(json_content[key])).float())
                    if batch_idx < resume_batch_idx:  # resume
                        continue
            if args.dataset == "ImageNet":
                if model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            if model.input_size[-1] == 299:
                self.freq_dim = 33
                self.stride = 7
            elif model.input_size[-1] == 331:
                self.freq_dim = 30
                self.stride = 7
            if args.dataset == "ImageNet" and model.input_size[-1] >= 299:
                self.image_width = model.input_size[-1]
                self.image_height = model.input_size[-1]
                if images.size(-1) != model.input_size[-1]:
                    images = F.interpolate(images, size=model.input_size[-1], mode='bilinear',align_corners=True)
                if self.random_matrix.size(1) != self.in_channels * self.image_height * self.image_width:
                    self.construct_random_matrix()

            self.attack_batch_images(model, batch_idx, images.cuda(), true_labels.cuda())
            tmp_info_dict = {"batch_idx": batch_idx + 1}
            for key in ['query_all', 'correct_all', 'not_done_all',
                        'success_all', 'success_query_all','not_done_prob_all']:
                value_all = getattr(self, key).detach().cpu().numpy().tolist()
                tmp_info_dict[key] = value_all
            with open(tmp_dump_path, "w") as result_file_obj:
                json.dump(tmp_info_dict, result_file_obj, sort_keys=True)
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write experimental result information info to {}".format(result_dump_path))


def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--low-dim", type=int, default=1500)
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--mom", type=float, default=1)
    parser.add_argument("--order", type=str, default="strided")
    parser.add_argument("--r", type=int, default=2352)
    parser.add_argument("--max_queries", type=int, default=10000)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--rho", type=float, default=0.001, help="modify from original 0.01 to 0.001 due to epsilon = 1.0 in CIFAR-10")
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--json-config', type=str,
                        default='./configures/PPBA_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"
    print("using GPU {}".format(args.gpu))
    return args

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def get_exp_dir_name(dataset, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'PPBA_attack_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'PPBA_attack-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

if __name__ == "__main__":
    args = get_parse_args()
    args_dict = None
    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
    if args.dataset == "ImageNet" and args.norm =="linf":
        args.epsilon = 0.05
    if args.dataset == "ImageNet" and args.targeted:
        args.max_queries = 50000
    args.exp_dir = os.path.join(args.exp_dir,
                            get_exp_dir_name(args.dataset,  args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
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
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    attacker = PPBA(args.dataset, args.order, args.r, args.rho, args.mom,args.n_samples,args.targeted,args.target_type,
                    args.norm, args.epsilon, args.low_dim, 0.0, 1.0, args.max_queries)
    for arch in archs:
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
            tmp_result_path = args.exp_dir + "/tmp_{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
            tmp_result_path = args.exp_dir + "/tmp_{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker.attack_all_images(args, model, tmp_result_path, save_result_path)
        model.cpu()
        os.unlink(tmp_result_path)