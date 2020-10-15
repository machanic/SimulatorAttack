import os
import sys
import time

import random

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
from types import SimpleNamespace
from collections import defaultdict
import glog as log
import json
import numpy as np
import torch
from torch.nn import functional as F
import os.path as osp
from config import PY_ROOT, CLASS_NUM, MODELS_TRAIN_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel

np.set_printoptions(precision=5, suppress=True)

class SquareAttack(object):
    def __init__(self, dataset, targeted, target_type, epsilon, norm, lower_bound=0.0, upper_bound=1.0,
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
        self.dataset = dataset
        self.epsilon = epsilon
        self.norm = norm
        self.max_queries = max_queries

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._proj = None
        self.is_new_batch = False
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)
        self.targeted = targeted
        self.target_type = target_type


    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        # it = int(it / n_iters * 10000)
        if 50 < it <= 100:
            p = p_init / 2
        elif 100 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 300:
            p = p_init / 8
        elif 500 < it <= 400:
            p = p_init / 16
        elif 1000 < it <= 500:
            p = p_init / 32
        elif 2000 < it <= 600:
            p = p_init / 64
        elif 4000 < it <= 700:
            p = p_init / 128
        elif 6000 < it <= 800:
            p = p_init / 256
        elif 8000 < it <= 900:
            p = p_init / 512
        else:
            p = p_init
        return p

    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = np.zeros([x, y])
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
            max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

            counter2[0] -= 1
            counter2[1] -= 1

        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

        return delta

    def meta_pseudo_gaussian_pert(self, s):
        delta = np.zeros([s, s])
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
            if np.random.rand(1) > 0.5:
                delta = np.transpose(delta)

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, :s // 2] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
            delta[:s // 2, s // 2:] = self.pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice(
                [-1, 1])
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        return delta

    def cw_loss(self, logits, label, target=None):
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logits.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logits[torch.arange(logits.shape[0]), target]
            second_max_logit = logits[torch.arange(logits.shape[0]), second_max_index]
            return second_max_logit - target_logit
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logits.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logits[torch.arange(logits.shape[0]), label]
            second_max_logit = logits[torch.arange(logits.shape[0]), second_max_index]
            return gt_logit - second_max_logit

    def xent_loss(self, logits, label, target=None):
        if target is not None:
            return F.cross_entropy(logits, target, reduction='none')
        else:
            return -F.cross_entropy(logits, label, reduction='none')


    def loss(self, logits, label, loss_type):
        if loss_type == "xent_loss":
            if self.targeted:
                return self.xent_loss(logits,None,label)
            else:
                return self.xent_loss(logits,label,None)
        elif loss_type == "cw_loss":
            if self.targeted:
                return self.cw_loss(logits, None, label)
            else:
                return self.cw_loss(logits,label, None)

    def square_attack_l2(self, model, x, y, eps, max_queries, p_init, loss_type):
        """ The L2 square attack """
        np.random.seed(0)
        c, h, w = x.shape[1:]
        n_features = c * h * w
        n_ex_total = x.shape[0]
        # x, y = x[corr_classified], y[corr_classified]
        ### initialization
        delta_init = np.zeros(x.shape)
        s = h // 5
        sp_init = (h - s * 5) // 2
        center_h = sp_init + 0
        for counter in range(h // s):
            center_w = sp_init + 0
            for counter2 in range(w // s):
                delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += self.meta_pseudo_gaussian_pert(s).reshape(
                    [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
                center_w += s
            center_h += s
        saved_images = []
        saved_logits = []
        x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, self.lower_bound, self.upper_bound)

        logits = model(torch.from_numpy(x_best).cuda().float())
        loss_min = self.loss(logits, torch.from_numpy(y).long().cuda(), loss_type=loss_type).detach().cpu().numpy()
        margin_min = self.loss(logits, torch.from_numpy(y).long().cuda(), loss_type='cw_loss').detach().cpu().numpy()
        n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

        time_start = time.time()
        n_iters = max_queries - 1
        metrics = np.zeros([n_iters, 7])
        slice_len = 100 if "ImageNet" not in self.dataset else 20
        assert n_iters >= slice_len
        slice_iteration_end = random.randint(slice_len, n_iters)
        for i_iter in range(slice_iteration_end):
            # idx_to_fool = (margin_min > 0.0).astype(np.bool)
            x_curr, x_best_curr = x, x_best
            y_curr, margin_min_curr = y, margin_min
            loss_min_curr = loss_min
            delta_curr = x_best_curr - x_curr
            p = self.p_selection(p_init, i_iter, n_iters)
            s = max(int(round(np.sqrt(p * n_features / c))), 3)
            if s % 2 == 0:
                s += 1
            s2 = s + 0
            ### window_1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)
            new_deltas_mask = np.zeros(x_curr.shape)
            new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0
            ### window_2
            center_h_2 = np.random.randint(0, h - s2)
            center_w_2 = np.random.randint(0, w - s2)
            new_deltas_mask_2 = np.zeros(x_curr.shape)
            new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
            ### compute total norm available
            curr_norms_window = np.sqrt(
                np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
            curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
            mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
            norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))

            ### create the updates
            new_deltas = np.ones([x_curr.shape[0], c, s, s])
            new_deltas = new_deltas * self.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
            new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
            old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
            new_deltas += old_deltas
            new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
                    np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
            delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
            delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

            x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
            x_new = np.clip(x_new, self.lower_bound, self.upper_bound)

            logits = model(torch.from_numpy(x_new).cuda().float())
            loss = self.loss(logits, torch.from_numpy(y_curr).long().cuda(), loss_type=loss_type).detach().cpu().numpy()

            margin = self.loss(logits, torch.from_numpy(y_curr).long().cuda(), loss_type='cw_loss').detach().cpu().numpy()

            idx_improved = (loss < loss_min_curr).astype(np.bool)
            loss_min = idx_improved * loss + ~idx_improved * loss_min_curr
            margin_min = idx_improved * margin + ~idx_improved * margin_min_curr

            idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_best = idx_improved * x_new + ~idx_improved * x_best_curr
            n_queries += 1

            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_corr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq, median_nq_ae = np.mean(n_queries), np.mean(
                n_queries[margin_min <= 0]), np.median(n_queries), np.median(n_queries[margin_min <= 0])

            time_total = time.time() - time_start
            metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq, margin_min.mean(), time_total]
            if i_iter >= slice_iteration_end - slice_len:
                saved_images.append(x_new)
                saved_logits.append(logits.detach().cpu().numpy())

        curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
        log.info('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))

        saved_images = np.transpose(np.stack(saved_images), (1, 0, 2, 3, 4))  # B,T,C,H,W
        saved_logits = np.transpose(np.stack(saved_logits), (1, 0, 2))  # B,T, #classes
        return saved_images, saved_logits

    def square_attack_linf(self, model, x, y, eps, max_queries, p_init, loss_type):
        """ The Linf square attack """
        np.random.seed(0)  # important to leave it here as well
        c, h, w = x.shape[1:]
        n_features = c * h * w
        n_ex_total = x.shape[0]
        # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
        init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
        x_best = np.clip(x + init_delta, self.lower_bound, self.upper_bound)
        saved_images = []
        saved_logits = []
        logits = model(torch.from_numpy(x_best).cuda().float())
        loss_min = self.loss(logits, torch.from_numpy(y).long().cuda(), loss_type=loss_type).detach().cpu().numpy()

        margin_min = self.loss(logits, torch.from_numpy(y).long().cuda(), loss_type='cw_loss').detach().cpu().numpy()
        n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

        time_start = time.time()
        n_iters = max_queries - 1
        metrics = np.zeros([n_iters, 7])

        slice_len = 100 if "ImageNet" not in self.dataset else 20
        slice_iteration_end = random.randint(slice_len, n_iters)
        for i_iter in range(slice_iteration_end):
            # idx_to_fool = (margin_min > 0).astype(np.bool)  # 攻击失败的index
            x_curr, x_best_curr, y_curr = x, x_best, y
            loss_min_curr, margin_min_curr = loss_min, margin_min
            deltas = x_best_curr - x_curr

            p = self.p_selection(p_init, i_iter, n_iters)
            for i_img in range(x_best_curr.shape[0]):
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)

                x_curr_window = x_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
                x_best_curr_window = x_best_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
                # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h + s, center_w:center_w + s], self.lower_bound,
                                self.upper_bound) - x_best_curr_window) < 10 ** -7) == c * s * s:
                    value = np.random.choice([-eps, eps], size=[c, 1, 1])
                    deltas[i_img, :, center_h:center_h + s, center_w:center_w + s] = value
                    value_ = np.squeeze(value)

            x_new = np.clip(x_curr + deltas, self.lower_bound, self.upper_bound)
            logits = model(torch.from_numpy(x_new).cuda().float())
            loss = self.loss(logits, torch.from_numpy(y_curr).long().cuda(), loss_type=loss_type).detach().cpu().numpy()
            margin = self.loss(logits,torch.from_numpy(y_curr).long().cuda(), loss_type='cw_loss').detach().cpu().numpy()

            idx_improved = loss < loss_min_curr
            loss_min = idx_improved * loss + ~idx_improved * loss_min_curr
            margin_min = idx_improved * margin + ~idx_improved * margin_min_curr
            idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_best = idx_improved * x_new + ~idx_improved * x_best_curr
            n_queries += 1
            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_corr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(
                n_queries[margin_min <= 0])
            # avg_margin_min = np.mean(margin_min)
            time_total = time.time() - time_start
            metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]
            if i_iter >= slice_iteration_end - slice_len:
                saved_images.append(x_new)  # idx_to_fool 的存在，每次的batch不同，所以应该去掉idx_to_fool
                saved_logits.append(logits.detach().cpu().numpy())

        saved_images = np.transpose(np.stack(saved_images),(1,0,2,3,4)) # B,T,C,H,W
        saved_logits = np.transpose(np.stack(saved_logits),(1,0,2))  # B,T,#classes

        return saved_images, saved_logits

    def attack_all_images(self, args, model_data_dict, save_dir):

        for (arch_name, target_model), image_label_list in model_data_dict.items():
            all_image_list = []
            all_logits_list = []
            target_model.cuda()
            targeted_str = "untargeted" if not args.targeted else "targeted"
            save_path_prefix = "{}/dataset_{}@arch_{}@norm_{}@loss_{}@{}".format(save_dir, args.dataset,
                                                                                 arch_name, args.norm, args.loss,
                                                                                 targeted_str)
            images_path = "{}@images.npy".format(save_path_prefix)
            shape_path = "{}@shape.txt".format(save_path_prefix)
            logits_path = "{}@logits.npy".format(save_path_prefix)
            log.info("Begin attack {}, the images will be saved to {}".format(arch_name, images_path))
            for batch_idx, (images, true_labels) in enumerate(image_label_list):
                images = images.cuda()
                true_labels = true_labels.cuda()
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
                        logits = target_model(images)
                        target_labels = logits.argmin(dim=1)
                    elif args.target_type == "increment":
                        target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                    else:
                        raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
                else:
                    target_labels = None
                loss_type = "cw_loss" if not self.targeted else "xent_loss"
                labels = true_labels if not self.targeted else target_labels
                if self.norm == "l2":
                    saved_images, saved_logits = self.square_attack_l2(target_model, images.detach().cpu().numpy(),
                                            labels.detach().cpu().numpy(), args.epsilon, args.max_queries, args.p, loss_type)
                elif self.norm == "linf":
                    saved_images, saved_logits = self.square_attack_linf(target_model, images.detach().cpu().numpy(),
                                                                labels.detach().cpu().numpy(),
                                                                args.epsilon, args.max_queries, args.p, loss_type)
                all_image_list.extend(saved_images)  # B,T,C,H,W
                all_logits_list.extend(saved_logits)   # B,T,#classes

            all_image_list = np.stack(all_image_list)  # B,T,C,H,W
            all_logits_list = np.stack(all_logits_list)

            store_shape = str(all_image_list.shape)
            with open(shape_path, "w") as file_shape:
                file_shape.write(store_shape)
                file_shape.flush()
            fp = np.memmap(images_path, dtype='float32', mode='w+', shape=all_image_list.shape)
            fp[:, :, :, :, :] = all_image_list[:, :, :, :, :]
            del fp
            np.save(logits_path, all_logits_list)
            log.info('{} is attacked finished, save to {}'.format(arch_name, images_path))
            target_model.cpu()

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

def get_log_path(dataset, loss, norm, targeted, target_type):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    dirname = 'generate_data-{}-{}loss-{}-{}.log'.format(dataset, loss, norm, target_str)
    return dirname

def main():
    parser = argparse.ArgumentParser(description='Square Attack Hyperparameters.')
    parser.add_argument('--norm', type=str, required=True, choices=['l2', 'linf'])
    parser.add_argument('--dataset',type=str, required=True)
    parser.add_argument('--gpu', type=str,required=True, help='GPU number. Multiple GPUs are possible for PT models.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--epsilon', type=float,  help='Radius of the Lp ball.')
    parser.add_argument('--max_queries',type=int,default=1000)
    parser.add_argument('--json-config', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/configures/square_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--batch_size',type=int,default=100)
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='random', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--loss', type=str)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)

    if args.targeted and args.dataset == "ImageNet":
        args.max_queries = 10000

    save_dir_path = "{}/data_square_attack/{}/{}".format(PY_ROOT, args.dataset,
                                                         "targeted_attack" if args.targeted else "untargeted_attack")
    os.makedirs(save_dir_path, exist_ok=True)
    loss_type = "cw" if not args.targeted else "xent"
    args.loss = loss_type
    log_path = osp.join(save_dir_path,
                        get_log_path(args.dataset, loss_type, args.norm, args.targeted, args.target_type))

    set_log_file(log_path)

    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_path))
    log.info('Called with args:')
    print_args(args)
    trn_data_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, is_train=True)
    models = []
    for arch in MODELS_TRAIN_STANDARD[args.dataset]:
        if StandardModel.check_arch(arch, args.dataset):
            model = StandardModel(args.dataset, arch, True)
            model = model.eval()
            models.append({"arch_name": arch, "model": model})
    model_data_dict = defaultdict(list)
    for images, labels in trn_data_loader:
        model_info = random.choice(models)
        arch = model_info["arch_name"]
        model = model_info["model"]
        if images.size(-1) != model.input_size[-1]:
            images = F.interpolate(images, size=model.input_size[-1], mode='bilinear', align_corners=True)
        model_data_dict[(arch, model)].append((images, labels))

    log.info("Assign data to multiple models over!")
    attacker = SquareAttack(args.dataset, args.targeted, args.target_type, args.epsilon, args.norm, max_queries=args.max_queries)
    attacker.attack_all_images(args, model_data_dict, save_dir_path)
    log.info("All done!")

if __name__ == "__main__":
    main()