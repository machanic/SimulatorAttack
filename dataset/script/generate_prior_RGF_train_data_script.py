import random
import sys
from collections import defaultdict, deque

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
import json
import os

from types import SimpleNamespace

import glog as log
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import IMAGE_SIZE, IN_CHANNELS, PY_ROOT, CLASS_NUM, MODELS_TRAIN_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel


class PriorRGFAttack(object):
    # 目前只能一张图一张图做对抗样本
    def __init__(self, dataset_name, model, surrogate_model, targeted, target_type):
        self.dataset_name = dataset_name
        self.image_height = IMAGE_SIZE[self.dataset_name][0]
        self.image_width =IMAGE_SIZE[self.dataset_name][1]
        self.in_channels = IN_CHANNELS[self.dataset_name]
        self.model = model
        self.surrogate_model = surrogate_model
        self.model.cuda().eval()
        self.surrogate_model.cuda().eval()
        self.targeted = targeted # only support untargeted attack now
        self.target_type = target_type
        self.clip_min = 0.0
        self.clip_max = 1.0


    def xent_loss(self, logit, true_labels, target_labels=None):
        if self.targeted:
            return -F.cross_entropy(logit, target_labels, reduction='none')
        else:
            return F.cross_entropy(logit, true_labels, reduction='none')

    def get_grad(self, model, x, true_labels, target_labels):
        with torch.enable_grad():
            x.requires_grad_()
            logits = model(x)
            loss = self.xent_loss(logits, true_labels, target_labels).mean()
            gradient = torch.autograd.grad(loss, x)[0]
        return gradient

    def get_pred(self, model, x):
        with torch.no_grad():
            logits = model(x)
        return logits.max(1)[1]

    def norm(self, t, p=2):
        assert len(t.shape) == 4
        if p == 2:
            norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        elif p == 1:
            norm_vec = t.abs().sum(dim=[1, 2, 3]).view(-1, 1, 1, 1)
        else:
            raise NotImplementedError('Unknown norm p={}'.format(p))
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def l2_proj_step(self, image, epsilon, adv_image):
        delta = adv_image - image
        out_of_bounds_mask = (self.norm(delta) > epsilon).float()
        return out_of_bounds_mask * (image + epsilon * delta / self.norm(delta)) + (1 - out_of_bounds_mask) * adv_image

    def attack_dataset(self, args, model_to_data, arch, save_dir):
        save_path_prefix = "{}/dataset_{}@arch_{}@norm_{}@{}".format(save_dir, args.dataset, arch,
                                                                     args.norm,
                                                                     "untargeted" if not args.targeted else "targeted_{}".format(
                                                                         args.target_type))
        q1_path = "{}@q1.npy".format(save_path_prefix)
        if os.path.exists(q1_path):
            log.info("Skip {}".format(arch))
            return
        success = 0
        queries = []
        not_done = []
        correct_all = []
        total = 0
        q1_list = []  # all the data
        q2_list = []
        logits_q1_list = []
        logits_q2_list = []

        for batch_idx, (images, true_labels) in enumerate(model_to_data[arch]):
            log.info("read data {}".format(batch_idx))
            input_q1_collection = deque(maxlen=2)
            input_q2_collection = deque(maxlen=2)
            logits_q1_collection = deque(maxlen=2)
            logits_q2_collection = deque(maxlen=2)

            self.image_height = images.size(2)
            self.image_width = images.size(3)
            eps = args.epsilon
            if args.norm == 'l2':
                # epsilon = 1e-3
                # eps = np.sqrt(epsilon * model.input_size[-1] * model.input_size[-1] * self.in_channels)  # 1.752
                learning_rate = 2.0 / np.sqrt(self.image_height * self.image_width * self.in_channels)
            else:
                learning_rate = 0.005

            images = images.cuda()
            true_labels = true_labels.cuda()

            # log.info("Begin dump {}-th image".format(batch_idx))
            with torch.no_grad():
                logits = self.model(images)
                pred = logits.argmax(dim=1)
                correct = pred.eq(true_labels).detach().cpu().numpy().astype(np.int32)
                correct_all.append(correct)
                if correct[0].item() == 0:
                    queries.append(0)
                    not_done.append(0)  # 原本就分类错了，not_done = 0
                    log.info("The {}-th image is already classified incorrectly.".format(batch_idx))
                    continue

            if self.targeted:
                if self.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logits.shape[1],
                                  size=target_labels[invalid_target_index].shape).long().cuda()     
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    target_labels = logits.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None

            total += images.size(0)
            sigma = args.sigma
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            adv_images = images.clone().cuda()
            assert images.size(0) == 1
            logits_real_images = self.model(images)
            l = self.xent_loss(logits_real_images, true_labels, target_labels)  # 按照元素论文来写的，好奇怪
            lr = float(learning_rate)
            total_q = 0
            ite = 0
            max_iters = random.randint(50, args.max_queries)
            current_not_done = 1
            dead_loop = 0
            is_dead_in_loop = False
            while (total_q <= max_iters and current_not_done ==1) or (len(input_q1_collection) < 2 or len(input_q2_collection) < 2):
                total_q += 1
                # true = torch.squeeze(self.get_grad(self.model, adv_images, true_labels, target_labels))  # C,H,W, # 其实没啥用，只是为了看看估计的准不准
                # log.info("Grad norm : {:.3f}".format(torch.sqrt(torch.sum(true * true)).item()))
                log.info("query :{}".format(total_q))
                if ite % 2 == 0 and sigma != args.sigma:
                    # log.info("checking if sigma could be set to be 1e-4")
                    rand = torch.randn_like(adv_images)
                    rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
                    logits_1 = self.model(adv_images + args.sigma * rand)
                    rand_loss = self.xent_loss(logits_1, true_labels, target_labels)  # shape = (batch_size,)
                    total_q += 1
                    rand =  torch.randn_like(adv_images)
                    rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
                    logits_2 = self.model(adv_images + args.sigma * rand)
                    rand_loss2= self.xent_loss(logits_2, true_labels, target_labels) # shape = (batch_size,)
                    total_q += 1
                    if (rand_loss - l)[0].item() != 0 and (rand_loss2 - l)[0].item() != 0:
                        sigma = args.sigma
                        log.info("set sigma back to 1e-4, sigma={:.4f}".format(sigma))

                if args.method != "uniform":
                    prior = torch.squeeze(self.get_grad(self.surrogate_model, adv_images, true_labels, target_labels))  # C,H,W
                    # 下面求得余弦值
                    # alpha = torch.sum(true * prior) / torch.clamp(torch.sqrt(torch.sum(true * true) * torch.sum(prior * prior)), min=1e-12)  # 这个alpha仅仅用来看看梯度对不对，后续会更新
                    # log.info("alpha = {:.3}".format(alpha))
                    prior = prior / torch.clamp(torch.sqrt(torch.mean(torch.mul(prior, prior))),min=1e-12)
                if args.method == "biased":
                    start_iter = 3  # 是只有start_iter=3的时候算一下gradient norm
                    if ite % 10 == 0 or ite == start_iter:
                        # Estimate norm of true gradient
                        s = 10
                        # pert shape = 10,C,H,W
                        pert = torch.randn(size=(s, adv_images.size(1), adv_images.size(2), adv_images.size(3)))
                        for i in range(s):
                            pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))), min=1e-12)
                        pert = pert.cuda()
                        # pert = (10,C,H,W), adv_images = (1,C,H,W)
                        eval_points =  adv_images + sigma * pert # broadcast, because tensor shape doesn't match exactly
                        # eval_points shape = (10,C,H,W) reshape to (10*1, C, H, W)
                        eval_points = eval_points.view(-1, adv_images.size(1), adv_images.size(2), adv_images.size(3))
                        target_labels_s = None
                        if target_labels is not None:
                            target_labels_s = target_labels.repeat(s)
                        logits_eval_points = self.model(eval_points)  # 10, #class
                        losses = self.xent_loss(logits_eval_points, true_labels.repeat(s), target_labels_s)  # shape = (10*B,)  # 收集
                        total_q += s
                        dead_loop += 1
                        if dead_loop > 500:
                            is_dead_in_loop = True
                            break
                        input_q1_collection.append(eval_points.detach().cpu().numpy())  # 10,C,H,W
                        logits_q1_collection.append(logits_eval_points.detach().cpu().numpy()) # 10, #class

                        log.info("{}-th image collect Q1, current length:{}".format(batch_idx, len(input_q1_collection)))
                        norm_square = torch.mean(((losses - l) / sigma) ** 2) # scalar
                    for jj in range(1000):
                        logits_for_prior_loss = self.model(adv_images + sigma* prior) # prior may be C,H,W
                        prior_loss = self.xent_loss(logits_for_prior_loss, true_labels, target_labels)  # shape = (batch_size,)
                        total_q += 1
                        diff_prior = (prior_loss - l)[0].item()
                        if diff_prior == 0:
                            sigma *= 2
                            # log.info("sigma={:.4f}, multiply sigma by 2".format(sigma))
                        else:
                            break
                    est_alpha = diff_prior / sigma / torch.clamp(torch.sqrt(torch.sum(torch.mul(prior,prior)) * norm_square), min=1e-12)
                    est_alpha = est_alpha.item()
                    # log.info("Estimated alpha = {:.3f}".format(est_alpha))
                    alpha = est_alpha   # alpha描述了替代模型的梯度是否有用，alpha越大λ也越大，λ=1表示相信这个prior
                    if alpha < 0:  #  夹角大于90度，cos变成负数
                        prior = -prior  # v = -v , negative the transfer gradient,
                        alpha = -alpha
                q = args.samples_per_draw
                n = self.image_height * self.image_width * self.in_channels
                d = 50 * 50 * self.in_channels
                gamma = 3.5
                A_square = d / n * gamma
                return_prior = False
                if args.method == 'biased':
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
                        return_prior = True   # lmda =1, we trust this prior as true gradient
                elif args.method == "fixed_biased":
                    lmda = 0.5
                if not return_prior:
                    if args.dataprior:
                        upsample = nn.UpsamplingNearest2d(size=(adv_images.size(-2), adv_images.size(-1)))  # H, W of original image
                        pert = torch.randn(size=(q, self.in_channels, 50, 50))
                        pert = upsample(pert)
                    else:
                        pert = torch.randn(size=(q, adv_images.size(-3), adv_images.size(-2), adv_images.size(-1)))  # q,C,H,W
                    pert = pert.cuda()
                    for i in range(q):
                        if args.method == 'biased' or args.method == 'fixed_biased':
                            angle_prior = torch.sum(pert[i] * prior) / \
                                          torch.clamp(torch.sqrt(torch.sum(pert[i] * pert[i]) * torch.sum(prior * prior)),min=1e-12)  # C,H,W x B,C,H,W
                            pert[i] = pert[i] - angle_prior * prior  # prior = B,C,H,W so pert[i] = B,C,H,W  # FIXME 这里不支持batch模式
                            pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))), min=1e-12)
                            # pert[i]就是论文算法1的第九行第二项的最右边的一串
                            pert[i] = np.sqrt(1-lmda) * pert[i] + np.sqrt(lmda) * prior  # paper's Algorithm 1: line 9
                        else:
                            pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))),min=1e-12)
                    for jjj in range(1000):
                        eval_points = adv_images + sigma * pert  # (1,C,H,W)  pert=(q,C,H,W), q = 50 for default setting
                        logits_ = self.model(eval_points)
                        input_q2_collection.append(eval_points.detach().cpu().numpy())  # q,C,H,W
                        logits_q2_collection.append(logits_.detach().cpu().numpy())
                        target_labels_q = None
                        if target_labels is not None:
                            target_labels_q = target_labels.repeat(q)
                        losses = self.xent_loss(logits_, true_labels.repeat(q), target_labels_q)  # shape = (q,)
                        log.info("{}-th image collect Q2, current length:{}".format(batch_idx, len(input_q2_collection)))
                        total_q += q
                        grad = (losses - l).view(-1, 1, 1, 1) * pert  # (q,1,1,1) * (q,C,H,W)
                        grad = torch.mean(grad,dim=0,keepdim=True)  # 1,C,H,W
                        norm_grad = torch.sqrt(torch.mean(torch.mul(grad,grad)))
                        if norm_grad.item() == 0:
                            sigma *= 5
                            # log.info("estimated grad == 0, multiply sigma by 5. Now sigma={:.4f}".format(sigma))
                        else:
                            break
                    grad = grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))), min=1e-12)

                    def print_loss(model, direction):
                        length = [1e-4, 1e-3]
                        les = []
                        for ss in length:
                            logits_p = model(adv_images + ss * direction)
                            loss_p = self.xent_loss(logits_p, true_labels, target_labels)
                            les.append((loss_p - l)[0].item())
                        # log.info("losses: ".format(les))

                    if args.show_loss:
                        if args.method == 'biased' or args.method == 'fixed_biased':
                            show_input = adv_images + lr * prior
                            logits_show = self.model(show_input)
                            lprior = self.xent_loss(logits_show, true_labels, target_labels) - l
                            print_loss(self.model, prior)
                            show_input_2 = adv_images + lr * grad
                            logits_show2 = self.model(show_input_2)
                            lgrad = self.xent_loss(logits_show2, true_labels, target_labels) - l
                            print_loss(self.model, grad)
                            # log.info(lprior, lgrad)
                else:
                    grad = prior
                # log.info("angle = {:.4f}".format(torch.sum(true*grad) /
                #                                  torch.clamp(torch.sqrt(torch.sum(true*true) * torch.sum(grad*grad)),min=1e-12)))
                if args.norm == "l2":
                    # Bandits版本
                    adv_images = adv_images + lr * grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))),min=1e-12)
                    adv_images = self.l2_proj_step(images, eps, adv_images)
                    # Below is the original author's L2 norm projection-based update
                    # adv_images = adv_images + lr * grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))),min=1e-12)
                    # norm = torch.clamp(torch.norm(adv_images - images),min=1e-12).item()
                    # factor = min(1, eps / norm)
                    # adv_images = images + (adv_images - images) * factor
                else:
                    if grad.dim() == 3:
                        grad = grad.unsqueeze(0)
                    adv_images = adv_images + lr * torch.sign(grad)
                    adv_images = torch.min(torch.max(adv_images, images - eps), images + eps)
                adv_images = torch.clamp(adv_images, self.clip_min, self.clip_max)
                adv_labels = self.get_pred(self.model, adv_images)
                logits_ = self.model(adv_images)
                l = self.xent_loss(logits_, true_labels, target_labels)
                # log.info('queries:{}, loss: {}, ', total_q, 'loss:', l, 'learning rate:', lr, 'sigma:', sigma, 'prediction:', adv_labels,
                #       'distortion:', torch.max(torch.abs(adv_images - images)).item(), torch.norm((adv_images - images).view(images.size(0),-1)).item())
                ite += 1

                if (self.targeted and adv_labels[0].item() == target_labels[0].item()) \
                        or (not self.targeted and adv_labels[0].item() != true_labels[0].item()):
                    # log.info("Success stop at queries : {}".format(total_q))
                    # success += 1
                    current_not_done = 0
                    # not_done.append(0)
                    # queries.append(total_q)
            # else:
            #     log.info("Failed stop at queries : {}".format(total_q))
            #
            #     current_not_done = 1
                # queries.append(args.max_queries) # 因此不能用np.mean(queries)来计算，平均query次数
            if not is_dead_in_loop:
                queries.append(total_q)
                not_done.append(current_not_done)
                success += (1-current_not_done)
                q1_list.append(np.stack(list(input_q1_collection)))  # each is 2,10,C,H,W, where 2 is T which can be splitted into meta-train and meta-test set
                q2_list.append(np.stack(list(input_q2_collection)))
                logits_q1_list.append(np.stack(list(logits_q1_collection)))
                logits_q2_list.append(np.stack(list(logits_q2_collection)))



        q1_list = np.stack(q1_list)   # B, 2, 10, C, H, W
        q2_list = np.stack(q2_list)   # B, 2, 50, C, H, W
        logits_q1_list = np.stack(logits_q1_list)   # B, 2, 10, #class
        logits_q2_list = np.stack(logits_q2_list)   # B, 2, 50, #class

        save_path_prefix = "{}/dataset_{}@arch_{}@norm_{}@{}".format(save_dir, args.dataset, arch,
                                                                               args.norm, "untargeted" if not args.targeted else "targeted_{}".format(args.target_type))

        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

        q1_path = "{}@q1.npy".format(save_path_prefix)
        q2_path = "{}@q2.npy".format(save_path_prefix)
        logits_q1_path = "{}@logits_q1.npy".format(save_path_prefix)
        logits_q2_path = "{}@logits_q2.npy".format(save_path_prefix)
        shape_q1_path = "{}@q1_shape.txt".format(save_path_prefix)
        shape_q2_path = "{}@q2_shape.txt".format(save_path_prefix)

        log.info('dump to {}'.format(save_path_prefix))
        log.info('Attack {} success rate: {:.3f} Queries_mean: {:.3f} Queries_median: {:.3f}'.format(arch, success/total,
                                                                                           np.mean(queries), np.median(queries)))
        with open(shape_q1_path, "w") as shape_file:
            shape_file.write(str(q1_list.shape))
            shape_file.flush()
        with open(shape_q2_path, "w") as shape_file:
            shape_file.write(str(q2_list.shape))
            shape_file.flush()

        fp = np.memmap(q1_path, dtype='float32', mode='w+', shape=q1_list.shape)
        fp[:, :, :, :, :, :] = q1_list[:, :, :, :, :, :]
        del fp
        del q1_list

        fp = np.memmap(q2_path, dtype='float32', mode='w+', shape=q2_list.shape)
        fp[:, :, :, :, :, :] = q2_list[:, :, :, :, :, :]
        del fp
        del q2_list

        fp = np.memmap(logits_q1_path, dtype='float32', mode='w+', shape=logits_q1_list.shape)
        fp[:, :, :, :] = logits_q1_list[:, :, :, :]
        del fp
        del logits_q1_list

        fp = np.memmap(logits_q2_path, dtype='float32', mode='w+', shape=logits_q2_list.shape)
        fp[:, :, :, :] = logits_q2_list[:, :, :, :]
        del fp
        del logits_q2_list


def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def get_log_path(dataset, norm, targeted, target_type):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    dirname = 'generate_data-{}-{}-{}.log'.format(dataset, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def partition_dataset(archs, data_loader, total_images):
    model_to_data = defaultdict(list)
    for idx, (images, true_labels) in enumerate(data_loader):
        if images.size(0) * idx >= total_images:
            break
        model_to_data[random.choice(archs)].append((images, true_labels))
    return model_to_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["CIFAR-10", "CIFAR-100", "MNIST", "FashionMNIST", "TinyImageNet", "ImageNet"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument('--json-config', default="/home1/machen/meta_perturbations_black_box_attack/configures/prior_RGF_attack_conf.json",
                        type=str, help='a configures file to be passed in instead of arguments')
    parser.add_argument("--surrogate_arch", type=str, help="The architecture of surrogate model,"
                                                                                 " in original paper it is resnet152")
    parser.add_argument("--method", type=str, default="biased", choices=['uniform', 'biased', 'fixed_biased'],
                        help='Methods used in the attack. uniform: RGF, biased: P-RGF (\lambda^*), fixed_biased: P-RGF (\lambda=0.5)')
    parser.add_argument("--dataprior", default=None, action="store_true",
                        help="Whether to use data prior in the attack.")
    parser.add_argument("--total-images",type=int,default=50000)
    parser.add_argument('--targeted', action="store_true", help="the targeted attack data")
    parser.add_argument("--target_type",type=str, default="random", choices=["least_likely","random","increment"])
    parser.add_argument("--max-queries", type=int,default=10000)
    parser.add_argument("--norm",type=str, choices=['linf','l2'], required=True)
    parser.add_argument("--samples_per_draw", type=int, default=50, help="Number of samples to estimate the gradient.")
    parser.add_argument("--epsilon", type=float, default=4.6, help='Default of epsilon is L2 epsilon')
    parser.add_argument("--sigma", type=float, default=1e-4, help="Sampling variance.")
    parser.add_argument("--show_loss", action="store_true", help="Whether to print loss in some given step sizes.")
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument("--max_queries", type=int, default=10000, help="Maximum number of queries.")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.backends.cudnn.deterministic = True
    assert args.batch_size == 1, 'The code does not support batch_size > 1 yet.'
    save_dir_path = "{}/data_prior_RGF_attack/{}/{}/".format(PY_ROOT, args.dataset, "targeted_attack" if args.targeted else "untargeted_attack")
    os.makedirs(save_dir_path, exist_ok=True)
    log_path = os.path.join(save_dir_path, get_log_path(args.dataset,  args.norm, args.targeted, args.target_type))  # 随机产生一个目录用于实验
    set_log_file(log_path)
    log.info("Log file is located in {}".format(log_path))
    log.info("All the data will be saved into {}".format(save_dir_path))
    log.info("Using GPU {}".format(args.gpu))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)
    defaults = json.load(open(args.json_config))[args.dataset]
    arg_vars = vars(args)
    arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
    defaults.update(arg_vars)
    args = SimpleNamespace(**defaults)
    if args.norm == "linf":
        args.epsilon = defaults["linf_epsilon"]
    args.surrogate_arch = "resnet-110" if args.dataset.startswith("CIFAR") else "resnet101"
    surrogate_model = StandardModel(args.dataset, args.surrogate_arch, False)
    trn_data_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, is_train=True)  # 生成的是训练集而非测试集
    archs = []
    for arch in MODELS_TRAIN_STANDARD[args.dataset]:
        if StandardModel.check_arch(arch, args.dataset):
            archs.append(arch)
    print("It will be use {} architectures".format(",".join(archs)))
    model_to_data = partition_dataset(archs, trn_data_loader, args.total_images)
    for arch in archs:
        model = StandardModel(args.dataset, arch, True)
        attacker = PriorRGFAttack(args.dataset, model, surrogate_model, args.targeted, args.target_type)
        log.info("Begin attack {}".format(arch))
        with torch.no_grad():
            attacker.attack_dataset(args, model_to_data, arch, save_dir_path)
        model.cpu()
        log.info("Attack {} with surrogate model {} done!".format(arch, args.surrogate_arch))

