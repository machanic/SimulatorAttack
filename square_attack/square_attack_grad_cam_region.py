import os
import sys
import time
import cv2
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
from types import SimpleNamespace
from square_attack.grad_cam import GradCAM
import glob
import glog as log
import json
import numpy as np
import torch
from torch.nn import functional as F
import os.path as osp
from config import PY_ROOT, MODELS_TEST_STANDARD, CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.defensive_model import DefensiveModel
from dataset.standard_model import StandardModel

np.set_printoptions(precision=5, suppress=True)

class SquareAttackGradCAMRegion(object):
    def __init__(self, dataset, batch_size, targeted, target_type, epsilon, norm, lower_bound=0.0, upper_bound=1.0,
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

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._proj = None
        self.is_new_batch = False
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)
        self.targeted = targeted
        self.target_type = target_type

        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.total_images = len(self.dataset_loader.dataset)

        self.query_all = torch.zeros(self.total_images)
        self.query_success_times_all = torch.zeros_like(self.query_all)
        self.query_fail_times_all = torch.zeros_like(self.query_all)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)


    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
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
        # log.info('Initial square side={} for bumps'.format(s))
        sp_init = (h - s * 5) // 2
        center_h = sp_init + 0
        for counter in range(h // s):
            center_w = sp_init + 0
            for counter2 in range(w // s):
                delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += self.meta_pseudo_gaussian_pert(s).reshape(
                    [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
                center_w += s
            center_h += s

        x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, self.lower_bound, self.upper_bound)

        logits = model(torch.from_numpy(x_best).cuda().float())
        loss_min = self.loss(logits, torch.from_numpy(y).long().cuda(), loss_type=loss_type).detach().cpu().numpy()
        margin_min = self.loss(logits, torch.from_numpy(y).long().cuda(), loss_type='cw_loss').detach().cpu().numpy()
        n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

        time_start = time.time()
        s_init = int(np.sqrt(p_init * n_features / c))
        n_iters = max_queries - 1
        metrics = np.zeros([n_iters, 7])
        query_fail_times = np.zeros(x.shape[0])
        query_success_times = np.zeros(x.shape[0])
        for i_iter in range(n_iters):
            idx_to_fool = (margin_min > 0.0).astype(np.bool)
            x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
            y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
            loss_min_curr = loss_min[idx_to_fool]
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

            # hps_str = 's={}->{}'.format(s_init, s)
            x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
            x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
            # curr_norms_image = np.sqrt(np.sum((x_new - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))

            logits = model(torch.from_numpy(x_new).cuda().float())
            loss = self.loss(logits, torch.from_numpy(y_curr).long().cuda(), loss_type=loss_type).detach().cpu().numpy()
            margin = self.loss(logits, torch.from_numpy(y_curr).long().cuda(), loss_type='cw_loss').detach().cpu().numpy()

            idx_improved = (loss < loss_min_curr).astype(np.bool)

            qq = query_success_times[idx_to_fool]
            qq[idx_improved] += 1
            query_success_times[idx_to_fool] = qq
            qq = query_fail_times[idx_to_fool]
            qq[~idx_improved] += 1
            query_fail_times[idx_to_fool] = qq
            loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
            margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

            idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
            n_queries[idx_to_fool] += 1

            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_corr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq, median_nq_ae = np.mean(n_queries), np.mean(
                n_queries[margin_min <= 0]), np.median(n_queries), np.median(n_queries[margin_min <= 0])

            time_total = time.time() - time_start
            metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq, margin_min.mean(), time_total]
            if acc == 0:
                curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
                log.info('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
                break

        curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
        log.info('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))

        return n_queries, query_success_times, query_fail_times, x_best

    def square_attack_linf(self, model, grad_cam, x, y, eps, max_queries, p_init, loss_type):
        """ The Linf square attack """
        np.random.seed(0)  # important to leave it here as well
        c, h, w = x.shape[1:]
        n_features = c * h * w
        n_ex_total = x.shape[0]
        # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
        init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
        x_best = np.clip(x + init_delta, self.lower_bound, self.upper_bound)
        all_x = torch.from_numpy(x).cuda()
        all_y = torch.from_numpy(y).cuda().long()
        all_binary_maps = []
        for i in range(x.shape[0]//5):
            mini_batch = 5
            saliency_maps = grad_cam.forward(all_x[i*mini_batch:(i+1)*mini_batch],all_y[i*mini_batch:(i+1)*mini_batch])  # B,1,H,W, FIXME 注意只能用于untargeted attack
            binary = (saliency_maps > 0.8).long().squeeze()
            binary = binary.detach().cpu().numpy()  # B,H,W
            all_binary_maps.append(binary)
        binary_maps = np.concatenate(all_binary_maps,0)
        saliency_rois = []
        for binary_map in binary_maps:
            connect_arr = cv2.connectedComponents(binary_map.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
            component_num = connect_arr[0]
            label_matrix = connect_arr[1]
            roi_coordinate = None
            roi_area = 0
            for component_label in range(1, component_num):
                row_col = list(zip(*np.where(label_matrix == component_label)))
                row_col = np.array(row_col)
                y_min_index = np.argmin(row_col[:, 0])
                y_min = row_col[y_min_index, 0]
                x_min_index = np.argmin(row_col[:, 1])
                x_min = row_col[x_min_index, 1]
                y_max_index = np.argmax(row_col[:, 0])
                y_max = row_col[y_max_index, 0]
                x_max_index = np.argmax(row_col[:, 1])
                x_max = row_col[x_max_index, 1]
                coordinates = [y_min, x_min, y_max, x_max]
                current_area = (y_max-y_min) * (x_max-x_min)
                if current_area > roi_area:
                    roi_area = current_area
                    roi_coordinate = coordinates
            if roi_coordinate is not None:
                saliency_rois.append(roi_coordinate)
                
        saliency_rois = np.array(saliency_rois)  # B, 4 and each box is [y_min, x_min, y_max, x_max] format
        assert saliency_rois.shape[0] == x_best.shape[0], "x batch size is {}, roi batch size is {}".format(x_best.shape[0], saliency_rois.shape[0])
        with torch.no_grad():
            logits = model(torch.from_numpy(x_best).cuda().float())
            loss_min = self.loss(logits, torch.from_numpy(y).long().cuda(), loss_type=loss_type).detach().cpu().numpy()
            margin_min = self.loss(logits, torch.from_numpy(y).long().cuda(), loss_type='cw_loss').detach().cpu().numpy()
        n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

        time_start = time.time()
        n_iters = max_queries - 1
        metrics = np.zeros([n_iters, 7])
        query_fail_times = np.zeros(x.shape[0])
        query_success_times = np.zeros(x.shape[0])
        for i_iter in range(n_iters - 1):
            idx_to_fool = (margin_min > 0).astype(np.bool)
            x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
            loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
            deltas = x_best_curr - x_curr

            p = self.p_selection(p_init, i_iter, n_iters)  # p_init = 0.05
            for i_img in range(x_best_curr.shape[0]):
                saliency_roi = saliency_rois[idx_to_fool][i_img]  # [y_min, x_min, y_max, x_max]
                y_min, x_min, y_max, x_max = saliency_roi.tolist()
                roi_features = (y_max-y_min) * (x_max - x_min)
                # current_p = (n_features / c) * p / float(roi_features)
                # if current_p > 1:
                #     current_p = 0.5
                s = int(round(np.sqrt(p * roi_features)))  # sqrt(p * h * w) p一开始是0.5,面积开方，边长
                s1 = min(max(s, 1), y_max - y_min - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                s2 = min(max(s, 1), x_max - x_min - 1)
                s = min(s1,s2)
                center_h = np.random.randint(y_min, y_max - s)
                center_w = np.random.randint(x_min, x_max - s)

                x_curr_window = x_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
                x_best_curr_window = x_best_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
                # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h + s, center_w:center_w + s],
                                            self.lower_bound, self.upper_bound) - x_best_curr_window) < 10 ** -7) == c * s * s:
                    deltas[i_img, :, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-eps, eps],
                                                                                                      size=[c, 1, 1])

            x_new = np.clip(x_curr + deltas, self.lower_bound, self.upper_bound)
            with torch.no_grad():
                logits = model(torch.from_numpy(x_new).cuda().float())
                loss = self.loss(logits, torch.from_numpy(y_curr).long().cuda(), loss_type=loss_type).detach().cpu().numpy()
                margin = self.loss(logits,torch.from_numpy(y_curr).long().cuda(), loss_type='cw_loss').detach().cpu().numpy()

            idx_improved = (loss < loss_min_curr).astype(np.bool)
            qq = query_success_times[idx_to_fool]
            qq[idx_improved] += 1
            query_success_times[idx_to_fool] = qq
            qq = query_fail_times[idx_to_fool]
            qq[~idx_improved] += 1
            query_fail_times[idx_to_fool] = qq
            loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
            margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
            idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
            n_queries[idx_to_fool] += 1

            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_corr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(
                n_queries[margin_min <= 0])
            # avg_margin_min = np.mean(margin_min)
            time_total = time.time() - time_start

            metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]
            if acc == 0:
                break

        return n_queries, query_success_times, query_fail_times, x_best

    def attack_all_images(self, args, arch_name, target_model, grad_cam, result_dump_path):

        for batch_idx, data_tuple in enumerate(self.dataset_loader):
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
            selected = torch.arange(batch_idx * args.batch_size,
                                    min((batch_idx + 1) * args.batch_size, self.total_images))
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

            with torch.no_grad():
                logit = target_model(images)
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels).float()
            # correct_np = correct.detach().cpu().numpy()
            # correct_indexes = np.nonzero(correct_np)[0]
            loss_type = "cw_loss" if not self.targeted else "xent_loss"
            labels = true_labels if not self.targeted else target_labels
            if self.norm == "l2":
                query, query_success_times, query_fail_times, adv_images = self.square_attack_l2(target_model, images.detach().cpu().numpy(),
                                                          labels.detach().cpu().numpy(),
                                         args.epsilon, args.max_queries, args.p, loss_type)
            elif self.norm == "linf":
                query, query_success_times, query_fail_times, adv_images = self.square_attack_linf(target_model, grad_cam,
                                                            images.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                                                            args.epsilon, args.max_queries, args.p, loss_type)
            query = torch.from_numpy(query).float().cuda()
            query_success_times = torch.from_numpy(query_success_times)
            query_fail_times = torch.from_numpy(query_fail_times)
            adv_images = torch.from_numpy(adv_images).float().cuda()
            with torch.no_grad():
                adv_logit = target_model(adv_images)
                adv_prob = F.softmax(adv_logit, dim=1)
            adv_pred = adv_logit.argmax(dim=1)
            if args.targeted:
                not_done =  (1 - adv_pred.eq(target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done =  adv_pred.eq(true_labels).float()
            success = (1 - not_done) * correct
            success_query = success * query
            not_done_prob = adv_prob[torch.arange(args.batch_size), true_labels] * not_done
            for key in ['query', 'query_success_times','query_fail_times', 'correct', 'not_done',
                        'success', 'success_query', 'not_done_prob']:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来
            log.info("{}-th batch (size={}), current batch success rate:{:.3f}".format(batch_idx, adv_images.size(0), success.mean().item()))

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('     avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('     avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.info('     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            log.info('     avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
            log.info('     max query: {}'.format(self.success_query_all[self.success_all.byte()].max().item()))
        if self.not_done_all.sum().item() > 0:
            log.info('  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "mean_query_success_times": self.query_success_times_all[self.success_all.byte()].mean().item(),
                          "mean_query_fail_times":self.query_fail_times_all[self.success_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_success_times_all": self.query_success_times_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_fail_times_all": self.query_fail_times_all.detach().cpu().numpy().astype(
                              np.int32).tolist(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def get_exp_dir_name(dataset, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'square_attack_grad_cam_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        dirname = 'square_attack_grad_cam-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def main():
    parser = argparse.ArgumentParser(description='Square Attack Hyperparameters.')
    parser.add_argument('--norm', type=str, required=True, choices=['l2', 'linf'])
    parser.add_argument('--dataset',type=str, required=True)
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--gpu', type=str,required=True, help='GPU number. Multiple GPUs are possible for PT models.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--epsilon', type=float,  help='Radius of the Lp ball.')
    parser.add_argument('--max_queries',type=int,default=10000)
    parser.add_argument('--json-config', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/configures/square_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--batch_size',type=int,default=100)
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
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
        args.max_queries = 50000
    args.exp_dir = os.path.join(args.exp_dir,
                            get_exp_dir_name(args.dataset,  args.norm, args.targeted, args.target_type,
                                             args))
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
    attacker = SquareAttackGradCAMRegion(args.dataset, args.batch_size,
                                         args.targeted, args.target_type, args.epsilon, args.norm, max_queries=args.max_queries)
    target_layer_dict = {"pnasnet5large":"cell_11","senet154":"layer4","inceptionv3":"Mixed_7c","inceptionv4":"features"}
    for arch in archs:
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=False, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=False)
        model.cuda()
        model.eval()
        grad_cam = GradCAM({"model_type":arch, "layer_name":target_layer_dict[arch], "arch":model,
                            "input_size":(model.input_size[-2], model.input_size[-1])})
        attacker.attack_all_images(args, arch, model, grad_cam, save_result_path)

if __name__ == "__main__":
    main()