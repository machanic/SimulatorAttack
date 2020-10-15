import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
import json

import os

import glob

import random

from types import SimpleNamespace

import torch
from torch import nn

from config import IMAGE_SIZE, IN_CHANNELS, CLASS_NUM, MODELS_TEST_STANDARD, PY_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
import glog as log
import numpy as np
from torch.nn import functional as F

from dataset.defensive_model import DefensiveModel
from dataset.standard_model import StandardModel


class FrankWolfeWhiteBoxAttack(object):

    def __init__(self, args, dataset, targeted, target_type, epsilon, norm, lower_bound=0.0, upper_bound=1.0,
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
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)
        self.targeted = targeted
        self.target_type = target_type

        self.data_loader = DataLoaderMaker.get_test_attacked_data(dataset, args.batch_size)
        self.total_images = len(self.data_loader.dataset)
        self.att_iter = args.max_queries
        self.correct_all = torch.zeros(self.total_images)  # number of images
        self.not_done_all = torch.zeros(self.total_images)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros(self.total_images)
        self.not_done_prob_all = torch.zeros(self.total_images)
        self.stop_iter_all = torch.zeros(self.total_images)
        self.ord =  args.norm # linf, l1, l2
        self.clip_min = args.clip_min
        self.clip_max = args.clip_max
        self.lr = args.lr
        self.beta1 = args.beta1
        self.loss_fn = nn.CrossEntropyLoss().cuda()

    def get_grad(self, model, inputs, targets):
        output = model(inputs)
        loss = self.loss_fn(output, targets)
        return torch.autograd.grad(loss, inputs)[0]

    def eval_image(self, model, inputs, true_labels, target_labels):
        output = model(inputs)
        pred = output.max(1)[1]
        if self.targeted:
            loss = self.loss_fn(output, target_labels)
            adv_correct = pred.eq(target_labels).long()
        else:
            loss = self.loss_fn(output, true_labels)
            adv_correct = pred.eq(true_labels).long()
        correct = pred.eq(true_labels).long()
        return loss.item(), output, correct, adv_correct

    def grad_normalization(self, gradients, order):
        if order == "linf":
            signed_grad = torch.sign(gradients)
        elif order in ["l1", "l2"]:
            reduce_indexes = list(range(1, gradients.ndimension()))
            if order == "l1":
                norm = gradients.clone().abs()
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(reduce_ind,keepdim=True)
            elif order == "l2":
                norm = gradients.clone()
                norm = torch.mul(norm, norm)
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(reduce_ind, keepdim=True)
                norm = torch.sqrt(norm)
            signed_grad = gradients / norm
        return signed_grad

    # Norm Ball Projection
    def norm_ball_proj_inner(self, eta, order, eps):
        if order == "linf":
            eta = torch.clamp(eta, -eps, eps)
        elif order in ["l1", "l2"]:
            reduce_indexes = list(range(1, len(eta.shape)))
            if order == 1:
                norm = eta.abs()
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(dim=reduce_ind, keepdim=True)
            elif order == 2:
                norm = torch.mul(eta, eta)
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(dim=reduce_ind, keepdim=True)
                norm = torch.sqrt(norm)
            if norm.item() > eps:
                eta = torch.mul(eta, torch.div(eps, norm))
        return eta

    def attack_batch_images(self, model, batch_index, inputs, true_labels, target_labels):
        x = inputs.clone()
        stop_iter =torch.zeros(inputs.size(0)).cuda()
        m_t = torch.zeros_like(inputs).cuda()
        loss_init, _,  correct, adv_correct = self.eval_image(model, inputs, true_labels, target_labels)
        finished_mask = 1.0 - adv_correct if not self.targeted else adv_correct
        succ_sum = torch.sum(finished_mask).item()
        log.info("Init Loss : % 5.3f, Finished: % 3d ".format(loss_init, succ_sum))
        batch_size = x.size(0)
        selected = torch.arange(batch_index * batch_size,
                                min((batch_index + 1) * batch_size, self.total_images))
        current_lr = self.lr
        for iteration in range(self.att_iter):
            if self.targeted:
                grad = self.get_grad(model, x, target_labels)
            else:
                grad = self.get_grad(model, x, true_labels)
            m_t = m_t * self.beta1 + grad * (1 - self.beta1)
            grad_normalized = self.grad_normalization(m_t, self.ord)
            v_t = - self.epsilon * grad_normalized + inputs
            d_t = v_t - x
            new_x = x + (-1 if not self.targeted else 1) * current_lr * d_t
            new_x = inputs + self.norm_ball_proj_inner(new_x - inputs, self.ord, self.epsilon)
            new_x = torch.clamp(new_x, self.clip_min, self.clip_max)
            mask = finished_mask.view(-1, *[1]*3)
            x = new_x * (1.0 - mask) + x * mask
            stop_iter += 1 * (1. - finished_mask)
            loss, adv_logit, correct, adv_correct = self.eval_image(model, x, true_labels, target_labels)
            tmp = 1.0 - adv_correct if not self.targeted else adv_correct
            finished_mask = finished_mask.byte() | tmp.byte()
            finished_mask = finished_mask.float()
            not_done = 1.0 - finished_mask
            adv_prob = F.softmax(adv_logit, dim=1)
            success = (1 - not_done) * correct
            not_done_prob = adv_prob[torch.arange(inputs.size(0)), true_labels] * not_done
            succ_sum = finished_mask.sum().item()
            if int(succ_sum) == inputs.size(0):
                break

        for key in ['stop_iter', 'correct',  'not_done',
                    'success', 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()

        return x, stop_iter, finished_mask


    def attack_all_images(self, args, arch_name, target_model, result_dump_path):

        for batch_idx, data_tuple in enumerate(self.data_loader):
            if args.dataset == "ImageNet":
                if target_model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            if images.size(-1) != target_model.input_size[-1]:
                images = F.interpolate(images, size=target_model.input_size[-1], mode='bilinear',align_corners=True)
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                                            size=target_labels[
                                                                                invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    logit = target_model(images)
                    target_labels = logit.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None
            self.attack_batch_images(target_model, batch_idx, images.cuda(), true_labels.cuda(),target_labels.cuda())
        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.not_done_all.sum().item() > 0:
            log.info('  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "stop_iter": self.stop_iter_all[self.success_all.byte()].mean().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          "args":vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


class FrankWolfeBlackBoxAttack(object):

    def __init__(self, args, dataset, targeted, target_type, epsilon, norm, sensing_type, grad_est_batch_size, delta,
                 beta1,
                 lower_bound=0.0, upper_bound=1.0, max_queries=10000):
        """
            :param epsilon: perturbation limit according to lp-ball
            :param norm: norm for the lp-ball constraint
            :param lower_bound: minimum value data point can take in any coordinate
            :param upper_bound: maximum value data point can take in any coordinate
            :param max_queries: max number of calls to model per data point
            :param max_crit_queries: max number of calls to early stopping criterion  per data poinr
        """
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        # super(FrankWolfeBlackBoxAttack, self).__init__(args, dataset, targeted, target_type, epsilon, norm,
        #                                                lower_bound, upper_bound, max_queries)
        self.epsilon = epsilon
        self.norm = norm
        self.max_queries = max_queries
        self.sensing_type = sensing_type
        self.delta = delta
        self.beta1 = beta1
        self.grad_est_batch_size = grad_est_batch_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)
        self.targeted = targeted
        self.target_type = target_type
        self.single_shape = (IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][0])
        self.data_loader = DataLoaderMaker.get_test_attacked_data(dataset, args.batch_size)
        self.total_images = len(self.data_loader.dataset)

        self.correct_all = torch.zeros(self.total_images)  # number of images
        self.not_done_all = torch.zeros(self.total_images) # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros(self.total_images)
        self.success_query_all = torch.zeros(self.total_images)
        self.not_done_prob_all = torch.zeros(self.total_images)
        self.query_all = torch.zeros(self.total_images)

        self.ord =  args.norm # linf, l1, l2
        self.clip_min = args.clip_min
        self.clip_max = args.clip_max
        self.lr = args.lr
        self.loss_fn = nn.CrossEntropyLoss().cuda()

    def get_grad_est(self, model, inputs, labels, num_batches):
        losses = []
        grads = []
        for _ in range(num_batches):
            noise_pos = torch.randn((self.grad_est_batch_size,)+self.single_shape)
            if self.sensing_type == 'sphere':
                reduce_indexes = list(range(1, inputs.dim()))
                noise_norm = torch.mul(noise_pos,noise_pos)
                for reduc_ind in reduce_indexes:
                    noise_norm = noise_norm.sum(reduc_ind, keepdim=True)
                noise_norm = torch.sqrt(noise_norm)
                noise_pos = noise_pos / noise_norm
                d = np.prod(self.single_shape).item()
                noise_pos = noise_pos * (d ** 0.5)
            noise = torch.cat([noise_pos, -noise_pos], dim=0).cuda()  # 2 * grad_est_batch_size,C,H,W
            grad_est_imgs = inputs + self.delta * noise
            grad_est_labs = labels.repeat(self.grad_est_batch_size * 2)
            assert grad_est_labs.size(0) == grad_est_imgs.size(0)
            grad_est_logits = model(grad_est_imgs)
            grad_est_losses = self.loss_fn(grad_est_logits, grad_est_labs)
            grad_est_losses_tiled = grad_est_losses.view(-1,1,1,1).repeat(1, self.single_shape[0], self.single_shape[1], self.single_shape[2])
            grad_estimates = torch.mean(grad_est_losses_tiled * noise, dim=0, keepdim=True)/self.delta  # 1,C,H,W
            losses.append(grad_est_losses)
            grads.append(grad_estimates)
        return torch.mean(torch.stack(losses)), torch.mean(torch.stack(grads),dim=0)

    def grad_normalization(self, gradients, order):
        if order == "linf":
            signed_grad = torch.sign(gradients)
        elif order in ["l1", "l2"]:
            reduce_indexes = list(range(1, gradients.ndimension()))
            if order == "l1":
                norm = gradients.clone().abs()
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(reduce_ind,keepdim=True)
            elif order == "l2":
                norm = gradients.clone()
                norm = torch.mul(norm, norm)
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(reduce_ind, keepdim=True)
                norm = torch.sqrt(norm)
            signed_grad = gradients / norm
        return signed_grad

    def eval_image(self, model, inputs, true_labels, target_labels):
        output = model(inputs)
        pred = output.max(1)[1]
        if self.targeted:
            loss = self.loss_fn(output, target_labels)
            adv_correct = pred.eq(target_labels).long()
        else:
            loss = self.loss_fn(output, true_labels)
            adv_correct = pred.eq(true_labels).long()
        correct = pred.eq(true_labels).long()
        return loss.item(), output, correct, adv_correct

    def attack_batch_images(self, model, batch_index, inputs, true_labels, target_labels):
        adv_images = inputs.clone()

        loss_init, example_output, correct, adv_correct = self.eval_image(model, inputs, true_labels, target_labels)
        finished_mask = 1.0 - adv_correct if not self.targeted else adv_correct
        succ_sum = torch.sum(finished_mask).item()
        batch_size = inputs.size(0)
        selected = torch.arange(batch_index * batch_size,
                                min((batch_index + 1) * batch_size, self.total_images))
        query = torch.zeros(inputs.size(0))
        if succ_sum == inputs.size(0):
            return adv_images, query, finished_mask
        adv_logits = torch.zeros_like(example_output)
        for i in range(inputs.size(0)):
            data = inputs[i:i + 1]
            true_label = true_labels[i:i + 1]
            if self.targeted:
                target_label = target_labels[i:i + 1]
            ori = inputs[i:i + 1].clone()
            x = data
            num_batches = 1
            m_t = torch.zeros_like(data).cuda()
            last_ls = []
            hist_len = 5
            start_decay = 0
            iteration = 0
            while query[i].item() < self.max_queries:
                iteration += 1
                query[i] += num_batches * self.grad_est_batch_size * 2
                if query[i].item() > self.max_queries:
                    query[i] = self.max_queries
                    break
                # Get zeroth-order gradient estimates
                if self.targeted:
                    _, grad = self.get_grad_est(model, x, target_label, num_batches)
                else:
                    _, grad = self.get_grad_est(model, x, true_label, num_batches)
                # momentum
                m_t = m_t * self.beta1 + grad * (1 - self.beta1)
                grad_normalized = self.grad_normalization(m_t, self.ord)
                s_t = - (-1 if not self.targeted else 1) * self.epsilon * grad_normalized + ori
                d_t = s_t - x
                current_lr = self.lr if start_decay == 0 else self.lr / (iteration - start_decay + 1) ** 0.5
                new_x = x + current_lr * d_t
                new_x = torch.clamp(new_x, self.clip_min, self.clip_max)
                x = new_x
                loss, adv_logit, _, adv_correct = self.eval_image(model, x, true_labels, target_labels)
                last_ls.append(loss)
                last_ls = last_ls[-hist_len:]
                if last_ls[-1] > 0.999 * last_ls[0] and len(last_ls) == hist_len:
                    if start_decay == 0:
                        start_decay = iteration - 1
                        # print("[log] start decaying lr")
                    last_ls = []
                finished_mask[i] = 1 - adv_correct[0] if not self.targeted else adv_correct[0]
                adv_logits[i] = adv_logit
                if finished_mask[i].item() == 1:
                    break
            adv_images[i] = new_x
        not_done = 1.0 - finished_mask
        adv_prob = F.softmax(adv_logits, dim=1)
        success = (1 - not_done) * correct
        not_done_prob = adv_prob[torch.arange(inputs.size(0)), true_labels] * not_done.float()
        success_query = success.detach().float().cpu() * query.detach().float().cpu()
        for key in ['query', 'correct',  'not_done',
                    'success', "success_query", 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()

        return adv_images, query, finished_mask

    def attack_all_images(self, args, arch_name, target_model, result_dump_path):

        for batch_idx, data_tuple in enumerate(self.data_loader):
            if args.dataset == "ImageNet":
                if target_model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            if images.size(-1) != target_model.input_size[-1]:
                images = F.interpolate(images, size=target_model.input_size[-1], mode='bilinear',align_corners=True)
            images = images.cuda()
            true_labels = true_labels.cuda()
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                                            size=target_labels[
                                                                                invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    logit = target_model(images)
                    target_labels = logit.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None
            adv_images, query, finished_mask = self.attack_batch_images(target_model, batch_idx, images, true_labels, target_labels)
            log.info("attack {}-th batch images over, avg. query = {}, success = {}".format(batch_idx+1, query.mean().item(), finished_mask[0].item()))

        log.info('{} is attacked finished ({} images)!'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.not_done_all.sum().item() > 0:
            log.info('  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
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
                          "args":vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def get_exp_dir_name(dataset, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    attack_str = "frank_wolfe_attack"
    if args.attack_defense:
        dirname = '{}_on_defensive_model-{}-{}-{}'.format(attack_str, dataset, norm, target_str)
    else:
        dirname = '{}-{}-{}-{}'.format(attack_str, dataset, norm, target_str)
    return dirname

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.04, type=float, help='attack learning rate')
    parser.add_argument('--norm', required=True, choices=["l2","linf"], type=str)
    parser.add_argument('--method', '-m', default="PGD", help='attack method')
    parser.add_argument('--arch', '-a', help='target architecture')
    parser.add_argument('--test_archs', action='store_true')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--epsilon', type=float, help='attack epsilon')
    parser.add_argument('--att_iter', default=25000, type=int, help='max number of attack iterations')
    parser.add_argument('--grad_est', default=25, type=int, help='gradient estimation batch size')
    parser.add_argument('--sensing', default='gaussian', help='sensing vector type: gaussian / sphere')
    parser.add_argument('--delta', default=0.01, type=float, help='delta for zero order estimiation')
    parser.add_argument('--beta1', default=0.99, type=float, help='beta1 for FW')
    parser.add_argument('--max_queries',type=int,default=10000)
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--json-config', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/configures/frank_wolfe_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--clip_min',type=float, default=0.0)
    parser.add_argument('--clip_max',type=float,default=1.0)
    args = parser.parse_args()
    return args

def main():
    args = get_parse_args()
    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
    args.exp_dir = os.path.join(args.exp_dir,
                                get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))
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
    attacker = FrankWolfeBlackBoxAttack(args, args.dataset, args.targeted, args.target_type, args.epsilon,
                                        args.norm, args.sensing, args.grad_est, args.delta, 0, 1, max_queries=args.max_queries)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.test_archs:
        archs = []
        if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT,
                    args.dataset, arch)
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

if __name__ == "__main__":
    main()