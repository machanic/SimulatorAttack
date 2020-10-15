import sys
from collections import OrderedDict

import random

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import os
import glog as log
import torch
from torch.nn import functional as F
import json
import glob
import os.path as osp
import numpy as np
from config import PY_ROOT, MODELS_TEST_STANDARD, CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.defensive_model import DefensiveModel
from dataset.standard_model import StandardModel
import argparse
from types import SimpleNamespace
from torch import nn

class ImageIdxToOrigBatchIdx(object):
    def __init__(self, batch_size):
        self.proj_dict = OrderedDict()
        for img_idx in range(batch_size):
            self.proj_dict[img_idx] = img_idx

    def del_by_index_list(self, del_img_idx_list):
        for del_img_idx in del_img_idx_list:
            del self.proj_dict[del_img_idx]
        all_key_value = sorted(list(self.proj_dict.items()), key=lambda e: e[0])
        for seq_idx, (img_idx, batch_idx) in enumerate(all_key_value):
            del self.proj_dict[img_idx]
            self.proj_dict[seq_idx] = batch_idx

    def __getitem__(self, img_idx):
        return self.proj_dict[img_idx]


class CombineSurrogateGradientAttack(object):
    def __init__(self, dataset, batch_size, targeted, target_type, epsilon, norm, lower_bound=0.0, upper_bound=1.0,
                 max_queries=10000):
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        self.epsilon = epsilon
        self.norm = norm
        self.max_queries = max_queries
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.targeted = targeted
        self.target_type = target_type
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)
        # self.cos_similarity_all = torch.zeros(self.total_images, max_queries)   # N, T

    def delete_tensor_by_index_list(self, del_index_list,  *tensors):
        return_tensors = []
        del_index_list = sorted(del_index_list)
        for tensor in tensors:
            if tensor is None:   # target_label may be None
                return_tensors.append(tensor)
                continue
            concatenate_tensor = []
            for i, each_tensor_element in enumerate(tensor):
                if i not in del_index_list:
                    concatenate_tensor.append(each_tensor_element)
            if len(concatenate_tensor) == 0:
                return [None for _ in tensors]  # delete all
            concatenate_tensor = torch.stack(concatenate_tensor, 0)
            # tensor = torch.cat([tensor[0:del_index], tensor[del_index + 1:]], 0)
            return_tensors.append(concatenate_tensor)
        return return_tensors

    def xent_loss(self, logit, label, target=None):
        if target is not None:
            return -F.cross_entropy(logit, target, reduction='none')
        else:
            return F.cross_entropy(logit, label, reduction='none')

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

    def normalize(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def l2_image_step(self, x, g, lr):
        return x + lr * g / self.normalize(g)

    def linf_image_step(self, x, g, lr):
        return x + lr * torch.sign(g)

    def linf_proj_step(self, image, epsilon, adv_image):
        return image + torch.clamp(adv_image - image, -epsilon, epsilon)

    def l2_proj_step(self, image, epsilon, adv_image):
        delta = adv_image - image
        out_of_bounds_mask = (self.normalize(delta) > epsilon).float()
        return out_of_bounds_mask * (image + epsilon * delta / self.normalize(delta)) + (1 - out_of_bounds_mask) * adv_image


    def get_grad(self, model, loss_fn, x, true_labels, target_labels):
        with torch.enable_grad():
            x.requires_grad_()
            logits = model(x)
            loss = loss_fn(logits, true_labels, target_labels)
            gradient = torch.autograd.grad(loss, x, torch.ones_like(loss), retain_graph=False)[0].detach()
        return gradient

    # def get_P_RGF_gradient(self, loss_fn, l, surrogate_gradient, target_model, adv_images,sigma, true_labels, target_labels, args):
    #     # 注意sigma的shape= (B,),而非标量
    #     total_q = torch.zeros(adv_images.size(0))
    #     if sigma != args.sigma: # FIXME 修改了
    #         rand = torch.randn_like(adv_images)
    #         rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
    #         logits_1 = target_model(adv_images + args.sigma * rand)
    #         rand_loss = loss_fn(logits_1, true_labels, target_labels)
    #         total_q += 1
    #         rand = torch.randn_like(adv_images)
    #         rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
    #         logits_2 = target_model(adv_images + args.sigma * rand)
    #         rand_loss2 = loss_fn(logits_2, true_labels, target_labels)
    #         total_q += 1
    #         if (rand_loss - l)[0].item() != 0 and (rand_loss2 - l)[0].item() != 0:
    #             sigma = args.sigma
    #             log.info("set sigma back to 1e-4, sigma={:.4f}".format(sigma))
    #     prior = surrogate_gradient
    #     prior = prior / torch.clamp(torch.sqrt(torch.mean(torch.mul(prior, prior))), min=1e-12)
    #     s = 10
    #     # pert shape = 10,C,H,W
    #     pert = torch.randn(size=(s, adv_images.size(1), adv_images.size(2), adv_images.size(3)))
    #     for i in range(s):
    #         pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))), min=1e-12)
    #     pert = pert.cuda()
    #     pert = pert.unsqueeze(0) # 1, 10, C, H, W
    #     # pert = (1,10,C,H,W), adv_image = (B,1,C,H,W) = (B,10,C,H,W)
    #     eval_points = adv_images.unsqueeze(1) + sigma * pert
    #     # FIXME 一次送入GPU可能显存不够
    #     eval_points = eval_points.view(-1, eval_points.size(-3), eval_points.size(-2), eval_points.size(-1))  # B*10, C,H,W
    #     target_labels_s = None
    #     if target_labels is not None:  # B,
    #         target_labels_s = target_labels.unsqueeze(1).repeat(1, s).view(-1) # B*10
    #     losses = loss_fn(target_model(eval_points), true_labels.unsqueeze(1).repeat(1, s).view(-1), target_labels_s)  # shape = (B*10)
    #     total_q += s
    #     norm_square = torch.mean((((losses.view(-1, s) - l.view(-1,1)) / sigma) ** 2),dim=1)  # shape = (B,)
    #     while True:
    #         logits_for_prior_loss = target_model(adv_images + sigma * prior)  # B,#classes
    #         prior_loss = loss_fn(logits_for_prior_loss, true_labels, target_labels)  # shape = (batch_size,)
    #         total_q += 1
    #         diff_prior = prior_loss - l
    #         if (diff_prior==0).byte().any().item():
    #             sigma[diff_prior==0] *= 2
    #         else:
    #             break
    #     # shape = (B,)
    #     est_alpha = torch.div(torch.div(diff_prior, sigma),
    #                           torch.clamp(torch.sqrt(torch.mul(prior,prior).view(prior.size(0),-1).sum(1) * norm_square), min=1e-12))
    #     alpha = est_alpha  # shape = (B,)
    #     prior[alpha<0] = -prior[alpha<0]
    #     alpha[alpha<0] = -alpha[alpha<0]
    #     q = args.samples_per_draw
    #     n = adv_images.size(-3) * adv_images.size(-2) * adv_images.size(-1)
    #     d = 50 * 50 * adv_images.size(-3)
    #     gamma = 3.5
    #     A_square = d / n * gamma
    #     return_prior = [False for _ in range(adv_images.size(0))]
    #     if args.dataprior:
    #         best_lambda = A_square * (A_square - alpha ** 2 * (d + 2 * q - 2)) / (
    #                 A_square ** 2 + alpha ** 4 * d ** 2 - 2 * A_square * alpha ** 2 * (q + d * q - 1))
    #     else:
    #         best_lambda = (torch.ones_like(alpha) - alpha ** 2) * (1 - alpha ** 2 * (n + 2 * q - 2)) / (
    #                 alpha ** 4 * n * (n + 2 * q - 2) - 2 * alpha ** 2 * n * q + 1)
    #     lmda = torch.zeros_like(best_lambda)
    #     lmda[(best_lambda<1).byte() & (best_lambda>0).byte()] = best_lambda[(best_lambda<1).byte() & (best_lambda>0).byte()]
    #     lmda[(alpha ** 2 * (n + 2 * q - 2) < 1).byte() & ~((best_lambda<1).byte() & (best_lambda>0).byte())] = 0
    #     lmda[(alpha ** 2 * (n + 2 * q - 2) >= 1).byte() & ~((best_lambda < 1).byte() & (best_lambda > 0).byte())] = 1.0
    #     lmda[(torch.abs(alpha)>=1).byte()] = 1.0
    #     return_prior[(lmda == 1.0).byte()] = True


    def get_P_RGF_gradient(self, loss_fn, l, surrogate_gradient, target_model, adv_images, sigma, true_labels,
                           target_labels, args):
        total_q = 0
        if sigma != args.sigma:
            rand = torch.randn_like(adv_images)
            rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
            logits_1 = target_model(adv_images + args.sigma * rand)
            rand_loss = loss_fn(logits_1, true_labels, target_labels)  # shape = (batch_size,)
            total_q += 1
            rand = torch.randn_like(adv_images)
            rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
            logits_2 = target_model(adv_images + args.sigma * rand)
            rand_loss2 = loss_fn(logits_2, true_labels, target_labels)  # shape = (batch_size,)
            total_q += 1
            if (rand_loss - l)[0].item() != 0 and (rand_loss2 - l)[0].item() != 0:
                sigma = args.sigma
                log.info("set sigma back to 1e-4, sigma={:.4f}".format(sigma))
        prior = torch.squeeze(surrogate_gradient)  # C,H,W
        prior = prior / torch.clamp(torch.sqrt(torch.mean(torch.mul(prior, prior))), min=1e-12)
        s = 10
        pert = torch.randn(size=(s, adv_images.size(1), adv_images.size(2), adv_images.size(3)))
        for i in range(s):
            pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))), min=1e-12)
        pert = pert.cuda()
        eval_points = adv_images + sigma * pert  # pert = (10,C,H,W), adv_images = (1,C,H,W)
        eval_points = eval_points.view(-1, adv_images.size(1), adv_images.size(2), adv_images.size(3))
        target_labels_s = None
        if target_labels is not None:
            target_labels_s = target_labels.repeat(s)
        losses = loss_fn(target_model(eval_points), true_labels.repeat(s), target_labels_s)
        total_q += s
        norm_square = torch.mean(((losses - l) / sigma) ** 2)  # scalar
        while True:
            logits_for_prior_loss = target_model(adv_images + sigma * prior)  # prior may be C,H,W
            prior_loss = loss_fn(logits_for_prior_loss, true_labels, target_labels)  # shape = (batch_size,)
            total_q += 1
            diff_prior = (prior_loss - l)[0].item()
            if diff_prior == 0:
                sigma *= 2
                # log.info("sigma={:.4f}, multiply sigma by 2".format(sigma))
            else:
                break
        est_alpha = diff_prior / sigma / torch.clamp(torch.sqrt(torch.sum(torch.mul(prior, prior)) * norm_square),
                                                     min=1e-12)
        est_alpha = est_alpha.item()
        # log.info("Estimated alpha = {:.3f}".format(est_alpha))
        alpha = est_alpha
        if alpha < 0:  # 夹角大于90度，cos变成负数
            prior = -prior  # v = -v , negative the transfer gradient,
            alpha = -alpha
        q = args.samples_per_draw
        n = adv_images.size(-3) * adv_images.size(-2) * adv_images.size(-1)
        d = 50 * 50 * adv_images.size(-3)
        gamma = 3.5
        A_square = d / n * gamma
        return_prior = False
        if args.dataprior:
            best_lambda = A_square * (A_square - alpha ** 2 * (d + 2 * q - 2)) / (
                    A_square ** 2 + alpha ** 4 * d ** 2 - 2 * A_square * alpha ** 2 * (q + d * q - 1))
        else:
            best_lambda = (1 - alpha ** 2) * (1 - alpha ** 2 * (n + 2 * q - 2)) / (
                    alpha ** 4 * n * (n + 2 * q - 2) - 2 * alpha ** 2 * n * q + 1)
        # log.info("best_lambda = {:.4f}".format(best_lambda))
        if best_lambda < 1 and best_lambda > 0:
            lmda = best_lambda
        else:
            if alpha ** 2 * (n + 2 * q - 2) < 1:
                lmda = 0
            else:
                lmda = 1
        if abs(alpha) >= 1:
            lmda = 1
        # log.info("lambda = {:.3f}".format(lmda))
        if lmda == 1:
            return_prior = True  # lmda =1, we trust this prior as true gradient
        if not return_prior:
            if args.dataprior:
                upsample = nn.UpsamplingNearest2d(
                    size=(adv_images.size(-2), adv_images.size(-1)))  # H, W of original image
                pert = torch.randn(size=(q, adv_images.size(-3), 50, 50))
                pert = upsample(pert)
            else:
                pert = torch.randn(size=(q, adv_images.size(-3), adv_images.size(-2), adv_images.size(-1)))  # q,C,H,W
            pert = pert.cuda()
            for i in range(q):
                angle_prior = torch.sum(pert[i] * prior) / \
                              torch.clamp(torch.sqrt(torch.sum(pert[i] * pert[i]) * torch.sum(prior * prior)),
                                          min=1e-12)  # C,H,W x B,C,H,W
                pert[i] = pert[i] - angle_prior * prior  # prior = B,C,H,W so pert[i] = B,C,H,W
                pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))), min=1e-12)
                # pert[i]就是论文算法1的第九行第二项的最右边的一串
                pert[i] = np.sqrt(1 - lmda) * pert[i] + np.sqrt(lmda) * prior  # paper's Algorithm 1: line 9
            while True:
                eval_points = adv_images + sigma * pert  # (1,C,H,W)  pert=(q,C,H,W)
                logits_ = target_model(eval_points)
                target_labels_q = None
                if target_labels is not None:
                    target_labels_q = target_labels.repeat(q)
                losses = loss_fn(logits_, true_labels.repeat(q), target_labels_q)  # shape = (q,)
                total_q += q
                grad = (losses - l).view(-1, 1, 1, 1) * pert  # (q,1,1,1) * (q,C,H,W)
                grad = torch.mean(grad, dim=0, keepdim=True)  # 1,C,H,W
                norm_grad = torch.sqrt(torch.mean(torch.mul(grad, grad)))
                if norm_grad.item() == 0:
                    sigma *= 5
                    log.info("estimated grad == 0, multiply sigma by 5. Now sigma={:.4f}".format(sigma))
                else:
                    break
        else:
            grad = prior
        # 注意l2 norm的lr = 2.0, linf norm 的lr=0.005
        return torch.squeeze(grad,dim=0), total_q, sigma



    def get_RGF_gradient(self, loss_fn, l, target_model, adv_images, sigma, true_labels,
                           target_labels, args):
        total_q = 0
        ite = random.randint(0,1)
        if sigma != args.sigma and ite == 1:
            rand = torch.randn_like(adv_images)
            rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
            logits_1 = target_model(adv_images + args.sigma * rand)
            rand_loss = loss_fn(logits_1, true_labels, target_labels)  # shape = (batch_size,)
            total_q += 1
            rand = torch.randn_like(adv_images)
            rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
            logits_2 = target_model(adv_images + args.sigma * rand)
            rand_loss2 = loss_fn(logits_2, true_labels, target_labels)  # shape = (batch_size,)
            total_q += 1
            if (rand_loss - l)[0].item() != 0 and (rand_loss2 - l)[0].item() != 0:
                sigma = args.sigma
                log.info("set sigma back to 1e-4, sigma={:.4f}".format(sigma))
        q = args.samples_per_draw
        if args.dataprior:
            upsample = nn.UpsamplingNearest2d(
                size=(adv_images.size(-2), adv_images.size(-1)))  # H, W of original image
            pert = torch.randn(size=(q, adv_images.size(-3), 50, 50))
            pert = upsample(pert)
        else:
            pert = torch.randn(size=(q, adv_images.size(-3), adv_images.size(-2), adv_images.size(-1)))  # q,C,H,W
        pert = pert.cuda()
        for i in range(q):
            pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))), min=1e-12)
        while True:
            eval_points = adv_images + sigma * pert  # (1,C,H,W)  pert=(q,C,H,W)
            logits_ = target_model(eval_points)
            target_labels_q = None
            if target_labels is not None:
                target_labels_q = target_labels.repeat(q)
            losses = loss_fn(logits_, true_labels.repeat(q), target_labels_q)  # shape = (q,)
            total_q += q
            grad = (losses - l).view(-1, 1, 1, 1) * pert  # (q,1,1,1) * (q,C,H,W)
            grad = torch.mean(grad, dim=0, keepdim=True)  # 1,C,H,W
            norm_grad = torch.sqrt(torch.mean(torch.mul(grad, grad)))
            if norm_grad.item() == 0:
                sigma *= 5
                log.info("estimated grad == 0, multiply sigma by 5. Now sigma={:.4f}".format(sigma))
            else:
                break
        # # 注意l2 norm的lr = 2.0, linf norm 的lr=0.005
        return torch.squeeze(grad,dim=0), total_q, sigma

    def compute_gradient_similarity(self, grad_a, grad_b):
        grad_a = grad_a.view(grad_a.size(0),-1)
        grad_b = grad_b.view(grad_b.size(0),-1)
        cos_similarity = (grad_a * grad_b).sum(dim=1) / torch.sqrt((grad_a * grad_a).sum(dim=1)
                                                                    * (grad_b * grad_b).sum(dim=1))

        assert cos_similarity.size(0) == grad_a.size(0)
        return cos_similarity

    def image_step(self, images, grad, surrogate_indexes, est_indexes, surrogate_image_lr, est_lr, norm):
        surrogate_grad = grad[surrogate_indexes]
        est_grad = grad[est_indexes]
        if norm == "l2":
            adv_images_surrogate = self.l2_image_step(images[surrogate_indexes], surrogate_grad, surrogate_image_lr)
            adv_images_est = self.l2_image_step(images[est_indexes], est_grad, est_lr)
        elif norm == "linf":
            adv_images_surrogate = self.linf_image_step(images[surrogate_indexes], surrogate_grad, surrogate_image_lr)
            adv_images_est = self.linf_image_step(images[est_indexes], est_grad, est_lr)
        adv_images = torch.zeros_like(images).cuda()
        for idx, index in enumerate(surrogate_indexes):
            adv_images[index] = adv_images_surrogate[idx]
        for idx, index in enumerate(est_indexes):
            adv_images[index] = adv_images_est[idx]
        return adv_images

    def attack_images(self, batch_index, images, true_labels, target_labels, target_model, surrogate_model, args):
        image_step = self.l2_image_step if args.norm == 'l2' else self.linf_image_step
        img_idx_to_batch_idx = ImageIdxToOrigBatchIdx(args.batch_size)
        proj_step = self.l2_proj_step if args.norm == 'l2' else self.linf_proj_step
        criterion = self.cw_loss if args.loss == "cw" else self.xent_loss
        adv_images = images.clone()
        query = torch.zeros(images.size(0)).cuda()
        # cos_similarity = torch.zeros(images.size(0), args.max_queries)
        with torch.no_grad():
            logit = target_model(images)
            l = criterion(logit, true_labels, target_labels)
        pred = logit.argmax(dim=1)
        correct = pred.eq(true_labels).float()  # shape = (batch_size,)
        not_done = correct.clone()
        selected = torch.arange(batch_index * args.batch_size,
                                min((batch_index + 1) * args.batch_size, self.total_images))  # 选择这个batch的所有图片的index
        sigma = torch.ones(args.batch_size).float() * args.sigma
        for step_index in range(args.max_queries):
            # true_gradients = self.get_grad(target_model, criterion, adv_images, true_labels,target_labels)
            surrogate_gradients = self.get_grad(surrogate_model, criterion, adv_images, true_labels, target_labels) # FIXME
            # cos_similarity = torch.sum(true_gradients * surrogate_gradients) / \
            #                  torch.clamp(torch.sqrt(torch.sum(true_gradients * true_gradients) *
            #                                         torch.sum(surrogate_gradients * surrogate_gradients)), min=1e-12)
            # log.info("Transfer cosine angle :{:.4f}".format(cos_similarity))
            # FIXME 删去了下面这些行，恢复回来！
            # attempt_images = image_step(adv_images, surrogate_gradients, args.image_lr)
            # with torch.no_grad():
            #     attempt_logits = target_model(attempt_images)
            #     attempt_loss = criterion(attempt_logits, true_labels, target_labels)
            #
            # idx_not_improved = (attempt_loss < l).detach().cpu().numpy().astype(np.int32)
            # idx_improved = 1-idx_not_improved
            grad = torch.zeros_like(adv_images).float().cuda()
            # log.info("{}-th iteration, surrogate img count:{}, {} est_grad img count:{}".format(step_index, np.sum(idx_improved).item(), args.est_method, np.sum(idx_not_improved).item()))
            idx_not_improved = np.arange(adv_images.size(0))
            if np.sum(idx_not_improved).item() > 0:
                for image_idx in np.nonzero(idx_not_improved)[0]:
                    img_sigma = sigma[img_idx_to_batch_idx[image_idx]]
                    if args.est_method == "P-RGF":
                        _target_labels = target_labels[image_idx].unsqueeze(0) if target_labels is not None else None
                        img_gradient, total_q, img_sigma = self.get_P_RGF_gradient(criterion,l[image_idx].unsqueeze(0),
                                                                                   surrogate_gradients[image_idx].unsqueeze(0),
                                                                                   target_model,
                                                           adv_images[image_idx].unsqueeze(0), img_sigma, true_labels[image_idx].unsqueeze(0),
                                                                            _target_labels, args)
                        grad[image_idx] = img_gradient
                    elif args.est_method == "RGF":
                        _target_labels = target_labels[image_idx].unsqueeze(0) if target_labels is not None else None
                        img_gradient, total_q, img_sigma = self.get_RGF_gradient(criterion,l[image_idx].unsqueeze(0), target_model,
                                                                                 adv_images[image_idx].unsqueeze(0),
                                                                                 img_sigma, true_labels[image_idx].unsqueeze(0),
                                                                                 _target_labels, args)
                        grad[image_idx] = img_gradient
                    elif args.est_method == "None":
                        grad[image_idx] = surrogate_gradients[image_idx]
                        total_q = 0
                    query[image_idx] = query[image_idx] + not_done[image_idx] * total_q
                    sigma[img_idx_to_batch_idx[image_idx]] = img_sigma
            # FIXME 删去了下面这些行，恢复回来！
            # if np.sum(idx_improved).item() > 0:
            #     for image_idx in np.nonzero(idx_improved)[0]:
            #         grad[image_idx] = surrogate_gradients[image_idx]

            # similarity = self.compute_gradient_similarity(grad, true_gradients)  # B
            # cos_similarity[:, step_index] = similarity.detach().cpu()
            adv_images = image_step(adv_images, grad, args.image_lr)
            # adv_images = self.image_step(adv_images, grad, np.nonzero(idx_improved)[0].tolist(), np.nonzero(idx_not_improved)[0].tolist(),
            #                              args.image_lr, args.RGF_L2_lr if args.norm == "l2" else args.RGF_Linf_lr, args.norm)
            adv_images = proj_step(images, args.epsilon, adv_images)
            adv_images = torch.clamp(adv_images, 0, 1).detach()
            with torch.no_grad():
                adv_logit = target_model(adv_images)
                adv_pred = adv_logit.argmax(dim=1)
                adv_prob = F.softmax(adv_logit, dim=1)
                l = criterion(adv_logit, true_labels, target_labels)
            query = query + not_done
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels).float()

            not_done[np.where(query.detach().cpu().numpy() > args.max_queries)[0].tolist()] = 0
            success = (1 - not_done) * correct
            success[np.where(query.detach().cpu().numpy() > args.max_queries)[0].tolist()] = 0

            success_query = success * query
            not_done_prob = adv_prob[torch.arange(adv_images.size(0)), true_labels] * not_done
            log.info('Attacking image {} - {} / {}, step {}, max query {}'.format(
                batch_index * args.batch_size, (batch_index + 1) * args.batch_size,
                self.total_images, step_index + 1, int(query.max().item())
            ))
            log.info('        correct: {:.4f}'.format(correct.mean().item()))
            log.info('       not_done: {:.4f}'.format(float(not_done.detach().cpu().sum().item()) / float(args.batch_size)))
            if success.sum().item() > 0:
                log.info('     mean_query: {:.4f}'.format(success_query[success.byte()].mean().item()))
                log.info('   median_query: {:.4f}'.format(success_query[success.byte()].median().item()))
            if not_done.sum().item() > 0:
                log.info('  not_done_prob: {:.4f}'.format(not_done_prob[not_done.byte()].mean().item()))

            not_done_np = not_done.detach().cpu().numpy().astype(np.int32)
            done_img_idx_list = np.where(not_done_np == 0)[0].tolist()
            delete_all = False
            if done_img_idx_list:
                for skip_index in done_img_idx_list:
                    batch_idx = img_idx_to_batch_idx[skip_index]
                    pos = selected[batch_idx].item()
                    for key in ['query', 'correct', 'not_done',
                                'success', 'success_query', 'not_done_prob']:
                        value_all = getattr(self, key + "_all")
                        value = eval(key)[skip_index].item()
                        value_all[pos] = value
                images, adv_images, query, true_labels, target_labels, correct, not_done, l, surrogate_gradients, grad = \
                    self.delete_tensor_by_index_list(done_img_idx_list, images, adv_images, query,
                                                     true_labels, target_labels, correct, not_done, l, surrogate_gradients, grad)
                img_idx_to_batch_idx.del_by_index_list(done_img_idx_list)
                delete_all = images is None
            if delete_all:
                break

        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_prob']:
            for img_idx, batch_idx in img_idx_to_batch_idx.proj_dict.items():
                pos = selected[batch_idx].item()
                value_all = getattr(self, key + "_all")
                value = eval(key)[img_idx].item()
                value_all[pos] = value  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来
        img_idx_to_batch_idx.proj_dict.clear()

    def attack_all_images(self, args, arch_name, target_model, surrogate_model, result_dump_path):
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

            self.attack_images(batch_idx, images, true_labels, target_labels, target_model, surrogate_model, args)

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.info(
                '     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            log.info(
                '   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
            log.info('     max query: {}'.format(self.success_query_all[self.success_all.byte()].max().item()))
        if self.not_done_all.sum().item() > 0:
            log.info(
                '  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        not_done_all = (1 - self.success_all.detach().cpu().numpy().astype(np.int32)).tolist()
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": 1.0 - self.success_all[self.correct_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": not_done_all,
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          # "gradient_cos_similarity": self.cos_similarity_all.detach().cpu().numpy().astype(np.float32).tolist(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def get_exp_dir_name(est_method, dataset, loss, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        # FIXME
        dirname = 'combine_surrogate_and_{}_grad_attack_on_defensive_model-{}-{}_loss-{}-{}'.format(est_method, dataset, loss, norm, target_str)
    else:
        dirname = 'DEBUG_{}_grad_attack-{}-{}_loss-{}-{}'.format(est_method, dataset, loss, norm, target_str)
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
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    # P-RGF/RGF arguments
    parser.add_argument('--RGF_L2_lr', type=float, default=2.0)
    parser.add_argument('--RGF_Linf_lr', type=float, default=0.005)
    parser.add_argument('--est_method', type=str, required=True, choices=["RGF","P-RGF","None"])
    parser.add_argument("--dataprior", action="store_true",
                        help="Whether to use data prior in the attack.")
    parser.add_argument("--samples_per_draw", type=int, default=50, help="Number of samples to estimate the gradient.")
    parser.add_argument("--sigma", type=float, default=1e-4, help="Sampling variance.")
    # P-RGF/RGF arguments over
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument("--loss", type=str, required=True, choices=["xent", "cw"])
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size for bandits attack.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument("--surrogate_arch", type=str, default="resnet-110", help="The architecture of surrogate model,"
                                                                                 " in original paper it is resnet152")
    parser.add_argument('--json-config', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/configures/surrogate_gradient_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random', 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument('--defense_model',type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"
    print("using GPU {}".format(args.gpu))
    if "RGF" in args.est_method:
        if args.dataset == "ImageNet":
            args.dataprior = True
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
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.est_method,
                            args.dataset, args.loss, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
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
    surrogate_model = StandardModel(args.dataset, args.surrogate_arch, False)
    surrogate_model.cuda()
    surrogate_model.eval()
    attacker = CombineSurrogateGradientAttack(args.dataset, args.batch_size, args.targeted, args.target_type, args.epsilon,
                                              args.norm, 0.0, 1.0, args.max_queries)
    for arch in archs:
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        # if os.path.exists(save_result_path):
        #     continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker.attack_all_images(args, arch,model, surrogate_model, save_result_path)
        model.cpu()
        log.info("Save result of attacking {} done".format(arch))
