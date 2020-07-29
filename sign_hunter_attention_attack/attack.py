"""
Implements SignHunter
"""
import json
import os
import sys
from collections import defaultdict

from sign_hunter_attention_attack.grad_cam import GradCAM

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")

import argparse
from types import SimpleNamespace

import glob
import numpy as np
import torch

from config import IMAGE_SIZE, IN_CHANNELS, CLASS_NUM, MODELS_TEST_STANDARD, PY_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.defensive_model import DefensiveModel
from sign_hunter_attention_attack.utils import sign, lp_step, convert_sparse_gradients_to_mask
from torch.nn import functional as F
import glog as log
import bidict
from dataset.standard_model import StandardModel


class SignHunterAttentionAttack(object):
    def __init__(self, dataset, targeted, target_type, epsilon, norm, batch_size,
                 lower_bound=0.0, upper_bound=1.0,
                 max_queries=10000, max_crit_queries=np.inf):
        """
            :param epsilon: perturbation limit according to lp-ball
            :param norm: norm for the lp-ball constraint
            :param lower_bound: minimum value data point can take in any coordinate
            :param upper_bound: maximum value data point can take in any coordinate
            :param max_queries: max number of calls to model per data point
            :param max_crit_queries: max number of calls to early stopping criterion  per data poinr
        """
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        assert not (np.isinf(max_queries) and np.isinf(max_crit_queries)), "one of the budgets has to be finite!"
        self.epsilon = epsilon
        self.norm = norm
        self.max_queries = max_queries
        self.max_crit_queries = max_crit_queries

        self.best_est_deriv = None
        self.xo_t = None
        self.sgn_t = None
        self.h = np.zeros(batch_size).astype(np.int32)
        self.i = np.zeros(batch_size).astype(np.int32)
        self.exhausted = [False for _ in range(batch_size)]

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._proj = None
        self.is_new_batch = False
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)
        self.targeted = targeted
        self.target_type = target_type

        self.data_loader = DataLoaderMaker.get_test_attacked_data(dataset, args.batch_size)
        self.total_images = len(self.data_loader.dataset)
        self.image_height = IMAGE_SIZE[dataset][0]
        self.image_width = IMAGE_SIZE[dataset][1]
        self.in_channels = IN_CHANNELS[dataset]

        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)



    def normalize(self, t):
        """
        Return the norm of a tensor (or numpy) along all the dimensions except the first one
        :param t:
        :return:
        """
        _shape = t.shape
        batch_size = _shape[0]
        num_dims = len(_shape[1:])
        if torch.is_tensor(t):
            norm_t = torch.sqrt(t.pow(2).sum(dim=[_ for _ in range(1, len(_shape))])).view([batch_size] + [1] * num_dims)
            norm_t += (norm_t == 0).float() * np.finfo(np.float64).eps
            return norm_t
        else:
            _norm = np.linalg.norm(
                t.reshape([batch_size, -1]), axis=1
            ).reshape([batch_size] + [1] * num_dims)
            return _norm + (_norm == 0) * np.finfo(np.float64).eps



    def l2_proj_maker(self, xs, eps):
        orig = xs.clone()

        def proj(new_x):
            delta = new_x - orig
            out_of_bounds_mask = (self.normalize(delta) > eps).float()
            x = (orig + eps * delta / self.normalize(delta)) * out_of_bounds_mask
            x += new_x * (1 - out_of_bounds_mask)
            return x

        return proj


    def linf_proj_maker(self, xs, eps):
        """
        makes an linf projection function such that new points
        are projected within the eps linf-balls centered around xs
        :param xs:
        :param eps:
        :return:
        """
        if torch.is_tensor(xs):
            orig_xs = xs.clone()

            def proj(new_xs):
                return orig_xs + torch.clamp(new_xs - orig_xs, - eps, eps)
        else:
            orig_xs = xs.copy()

            def proj(new_xs):
                return np.clip(new_xs, orig_xs - eps, orig_xs + eps)
        return proj

    def proj_replace(self, xs_t, sugg_xs_t, dones_mask_t):
        sugg_xs_t = self._proj(sugg_xs_t) # _proj函数在run函数开始处定义
        # replace xs only if not done
        xs_t = sugg_xs_t * (1. - dones_mask_t) + xs_t * dones_mask_t
        return xs_t


    def attack_all_images(self, arch, model, args, tmp_dump_path, result_dump_path):
        grad_cam = GradCAM({"model_type": arch, "arch": model, "layer_name":"relu_final"})
        for batch_idx, data_tuple in enumerate(self.data_loader):
            if os.path.exists(tmp_dump_path):
                with open(tmp_dump_path, "r") as file_obj:
                    json_content = json.load(file_obj)
                    resume_batch_idx = int(json_content["batch_idx"])  # resume
                    for key in ['query_all', 'correct_all', 'not_done_all',
                                'success_all', 'success_query_all']:
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
            if images.size(-1) != model.input_size[-1]:
                images = F.interpolate(images, size=model.input_size[-1], mode='bilinear', align_corners=True)
            images = images.cuda()
            true_labels = true_labels.cuda()

            bin_masks = []  # length= batch_size
            for image, true_label in zip(images, true_labels):
                saliency_map = grad_cam.forward(image, true_label)
                cam_mask = saliency_map.ge(torch.max(saliency_map) * 0.15)
                cam_mask = cam_mask.long()
                bin_masks.append(cam_mask)
            bin_masks = torch.stack(bin_masks)  # B,H,W

            if self.targeted:
                if self.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                  size=target_labels[invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    logits = model(images)
                    target_labels = logits.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None

            self.attack_batch_images(batch_idx, images, true_labels, target_labels, model, bin_masks,
                                      args)
            tmp_info_dict = {"batch_idx": batch_idx + 1, "batch_size": args.batch_size}
            for key in ['query_all', 'correct_all', 'not_done_all',
                        'success_all', 'success_query_all']:
                value_all = getattr(self, key).detach().cpu().numpy().tolist()
                tmp_info_dict[key] = value_all
            with open(tmp_dump_path, "w") as result_file_obj:
                json.dump(tmp_info_dict, result_file_obj, sort_keys=True)

        over_query_limit_samples_indexes = np.where(self.success_query_all.cpu().detach().numpy() > args.max_queries)[0].tolist()
        if over_query_limit_samples_indexes:
            self.success_query_all[over_query_limit_samples_indexes] = args.max_queries
            self.not_done_all[over_query_limit_samples_indexes] = 1
            self.success_all[over_query_limit_samples_indexes] = 0
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

    def attack_batch_images(self, batch_index, xs, true_labels, target_labels, model, bin_masks, args):
        """
        attack with `xs` as data points using the oracle `l` and the early stopping criterion `early_stop_crit_fct`
        :param xs: data points to be perturbed adversarially (numpy array)
        :param early_stop_crit_fct: early stop function (m data pts -> {0,1}^m)
                ith entry is 1 if the ith data point is misclassified
        :return: a dict of logs whose length is the number of iterations
        """
        # convert to tensor
        batch_size = xs.size(0)
        num_axes = len(xs.shape[1:])
        with torch.no_grad():
            logit = model(xs)
        pred = logit.argmax(dim=1)
        query = torch.zeros(batch_size).float().cuda()
        correct = pred.eq(true_labels).float()
        not_done = correct.clone()
        selected = torch.arange(batch_index * batch_size,
                                min((batch_index + 1) * batch_size, self.total_images))  # 选择这个batch的所有图片的index

        # make a projector into xs lp-ball and within valid pixel range
        if self.norm == 'l2':
            _proj = self.l2_proj_maker(xs, self.epsilon)
            self._proj = lambda xx: torch.clamp(_proj(xx), self.lower_bound, self.upper_bound)
        elif self.norm == 'linf':
            _proj = self.linf_proj_maker(xs, self.epsilon)
            self._proj = lambda xx: torch.clamp(_proj(xx), self.lower_bound, self.upper_bound)
        else:
            raise Exception('Undefined l-p!')

        # iterate till model evasion or budget exhaustion to inform self._suggest this is  a new batch
        self.is_new_batch = True
        step_index = 0
        
        while int(query.max().item()) < args.max_queries:
            # propose new perturbations
            sugg_xs_t, num_loss_queries_per_step = self._suggest(model, xs, true_labels, target_labels,
                                                                 bin_masks)
            # project around xs and within pixel range and replace xs only if not done
            xs = self.proj_replace(xs, sugg_xs_t, 1.0 - not_done.view(-1, *[1] * num_axes))
            with torch.no_grad():
                adv_logit = model(xs)
            adv_pred = adv_logit.argmax(dim=1)
            adv_prob = F.softmax(adv_logit, dim=1)
            query += num_loss_queries_per_step * not_done
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels).float()  # 只要是跟原始label相等的，就还需要query，还没有成功
            success = (1 - not_done) * correct
            success_query = success * query
            not_done_prob = adv_prob[torch.arange(args.batch_size), true_labels] * not_done
            self.is_new_batch = False
            step_index += 1
            log.info('Attacking image {} - {} / {}, step {}, max query {}'.format(
                batch_index * batch_size, (batch_index + 1) * batch_size,
                self.total_images, step_index, int(query.max().item())
            ))
            log.info('        correct: {:.4f}'.format(correct.mean().item()))
            log.info('       not_done: {:.4f}'.format(not_done[correct.byte()].mean().item()))
            if success.sum().item() > 0:
                log.info('     mean_query: {:.4f}'.format(success_query[success.byte()].mean().item()))
                log.info('   median_query: {:.4f}'.format(success_query[success.byte()].median().item()))
            if not_done.sum().item() > 0:
                log.info('  not_done_prob: {:.4f}'.format(not_done_prob[not_done.byte()].mean().item()))
            if not not_done.byte().any():  # all success
                break

        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来
        # set self._proj to None to ensure it is intended use
        self._proj = None

    def loss_fct(self,model, x, label, target=None):
        logit = model(x)
        if target is not None:
            return -F.cross_entropy(logit, target, reduction='none')
        else:
            return F.cross_entropy(logit, label, reduction='none')

    def get_coordinate_bin_masks(self, bin_masks):
        bin_masks = bin_masks.detach().cpu().numpy()
        x_coordinates, y_coordinates = np.nonzero(bin_masks)
        coordinates = defaultdict(list)
        for x, y in zip(x_coordinates, y_coordinates):
            coordinates[x].append(y)
        return coordinates

    def flip(self, sgn_t, nonzero_coordinates):
        """
        :param sgn_t: shape = (batch, dim)
        :param nonzero_coordinates: dict = batch_id, id_list
        :return:
        """
        iends_eq_dims = []
        for batch_id, id_list in nonzero_coordinates.items():
            real_dim = len(id_list)
            chunk_len = np.ceil(real_dim / (2 ** self.h[batch_id])).astype(int)
            istart = self.i[batch_id] * chunk_len
            iend = min(real_dim, (self.i[batch_id] + 1) * chunk_len)
            iends_eq_dims.append(int(iend == real_dim))
            flip_id_list = id_list[istart:iend]
            sgn_t[batch_id, flip_id_list] *= -1
        return np.array(iends_eq_dims, dtype=np.uint8)

    def flip_certain_images(self, sgn_t, nonzero_coordinates, certain_img_ids):
        """
        :param sgn_t: shape = (batch, dim)
        :param nonzero_coordinates: dict = batch_id, id_list
        :return:
        """
        for batch_id, id_list in nonzero_coordinates.items():
            if batch_id in certain_img_ids:
                real_dim = len(id_list)
                chunk_len = np.ceil(real_dim / (2 ** self.h[batch_id])).astype(int)
                istart = self.i[batch_id] * chunk_len
                iend = min(real_dim, (self.i[batch_id] + 1) * chunk_len)
                flip_id_list = id_list[istart:iend]
                sgn_t[batch_id, flip_id_list] *= -1


    def _suggest(self, model, xs_t, label, target, bin_masks):
        """
        :param xs_t: data points to be perturbed adversarially (numpy array)
        :param shrink_coordinate_maps: original coordinate -> shrink coordinate
        :return:
        """
        _shape = list(xs_t.shape)

        bin_masks = bin_masks.view(_shape[0], -1)
        # expansive operation
        non_zero_coordinates = self.get_coordinate_bin_masks(bin_masks)
        real_dim_list = []
        for batch_id, id_list in non_zero_coordinates.items():
            real_dim = len(id_list)
            real_dim_list.append(real_dim)

        # additional queries at the start
        query_count = torch.zeros(xs_t.shape[0]).float().cuda()

        if self.is_new_batch:
            self.xo_t = xs_t.clone()  # 新的原始数据作为 xs_t
            self.h.fill(0)
            self.i.fill(0)
            self.exhausted = [False for _ in range(_shape[0])]
        if (not self.i.any()) and (not self.h.any()):
            self.sgn_t = torch.zeros(_shape[0], np.prod(_shape[1:]).item()).cuda()
            self.sgn_t[bin_masks==1] = 1  #
            fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.norm)
            bxs_t = self.xo_t
            est_deriv = (self.loss_fct(model, fxs_t, label, target) - self.loss_fct(model, bxs_t, label, target)) / self.epsilon
            self.best_est_deriv = est_deriv
            #add_queries = 3  # because of bxs_t and the 2 evaluations in the i=0, h=0, case.
            query_count += 2
        self.sgn_t[bin_masks==0] = 0
        self.sgn_t[(bin_masks==1).byte() & (self.sgn_t==0).byte()] = 1  # binary mask has changed

        iends_eq_dims = self.flip(self.sgn_t, non_zero_coordinates)

        fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.norm)
        bxs_t = self.xo_t
        est_deriv = (self.loss_fct(model, fxs_t, label, target) - self.loss_fct(model, bxs_t, label, target)) / self.epsilon
        query_count += 1
        for ie, exhausted in enumerate(self.exhausted):
            if exhausted:
                query_count[ie] += 1
                self.exhausted[ie] = False

        revert_images = [i for i, val in enumerate(est_deriv < self.best_est_deriv) if val]
        self.flip_certain_images(self.sgn_t, non_zero_coordinates, revert_images)  # loss decrease, revert the sign
        self.best_est_deriv = (est_deriv >= self.best_est_deriv).float() * est_deriv \
                              + (est_deriv < self.best_est_deriv).float() * self.best_est_deriv  # loss increase , 保留大的作为best
        # compute the cosine similarity
        # cos_sims, ham_sims = metric_fct(self.xo_t.cpu().numpy(), self.sgn_t.cpu().numpy())
        # perform the step
        new_xs = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.norm)   # 第三个变量函数定义明明是lr，因为是sign gradient，所以lr == epsilon
        # update i and h for next iteration
        self.i += 1
        # 每张图的iend不同,因此需改成独立的统计
        end_idx = (self.i == np.array([2 ** hh.item() for hh in self.h])).astype(np.uint8)
        condition = end_idx | iends_eq_dims
        select_condition_index = np.nonzero(condition)[0]
        if len(select_condition_index) > 0:
            for index in select_condition_index:  # it is time to shrink the block and search again.
                self.h[index] += 1
                self.i[index] = 0
                # if h is exhausted, set xo_t to be xs_t
                if self.h[index] == np.ceil(np.log2(real_dim_list[index])).astype(int) + 1:
                    self.xo_t[index] = xs_t[index].clone()
                    self.h[index] = 0
                    self.exhausted[index] = True
        return new_xs, query_count


def get_exp_dir_name(attacker, dataset, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = '{}_attack_on_defensive_model-{}-{}-{}'.format(attacker, dataset, norm, target_str)
    else:
        dirname = '{}_attack-{}-{}-{}'.format(attacker, dataset, norm, target_str)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dataset",type=str, required=True)
    parser.add_argument("--attacker",type=str, choices=["rand_sign_attention","sign_hunter_attention"], default="sign_hunter")
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--max_queries',type=int, default=10000)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--batch_size',type=int,default=100)
    parser.add_argument('--json-config', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/configures/sign_hunter_attack.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
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
    args.exp_dir = os.path.join(args.exp_dir,
                            get_exp_dir_name(args.attacker, args.dataset, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
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
    if args.attacker == 'sign_hunter':
        attacker = SignHunterAttentionAttack(args.dataset, args.targeted, args.target_type, args.epsilon, args.norm,
                                             args.batch_size, lower_bound=0.0, upper_bound=1.0, max_queries=args.max_queries)
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
        attacker.attack_all_images(model, args, tmp_result_path, save_result_path)
        model.cpu()
        os.unlink(tmp_result_path)