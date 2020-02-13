import sys

from utils.statistics_toolkit import success_rate_and_query_coorelation, success_rate_avg_query

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
import json
import os
import os.path as osp
import random

import time
from types import SimpleNamespace

import glog as log
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.modules import Upsample

from config import IMAGE_SIZE, IN_CHANNELS, CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker
from meta_simulator_attack.meta_model_finetune import MetaModelFinetune
from target_models.standard_model import StandardModel
from collections import deque

class BanditsAttack(object):
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

    def norm(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    ###
    # Different optimization steps
    # All take the form of func(x, g, lr)
    # eg: exponentiated gradients
    # l2/linf: projected gradient descent
    ###

    def eg_step(self, x, g, lr):
        real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
        pos = real_x * torch.exp(lr * g)
        neg = (1 - real_x) * torch.exp(-lr * g)
        new_x = pos / (pos + neg)
        return new_x * 2 - 1

    def linf_step(self, x, g, lr):
        return x + lr * torch.sign(g)

    def l2_prior_step(self, x, g, lr):
        new_x = x + lr * g / self.norm(g)
        norm_new_x = self.norm(new_x)
        norm_mask = (norm_new_x < 1.0).float()
        return new_x * norm_mask + (1 - norm_mask) * new_x / norm_new_x

    def gd_prior_step(self, x, g, lr):
        return x + lr * g

    def l2_image_step(self, x, g, lr):
        return x + lr * g / self.norm(g)


    def l2_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            delta = new_x - orig
            out_of_bounds_mask = (self.norm(delta) > eps).float()
            x = (orig + eps * delta / self.norm(delta)) * out_of_bounds_mask
            x += new_x * (1 - out_of_bounds_mask)
            return x
        return proj

    def linf_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            return orig + torch.clamp(new_x - orig, -eps, eps)
        return proj

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

    def make_adversarial_examples(self, batch_index, images, true_labels, args, target_model, meta_finetuner):
        '''
        The attack process for generating adversarial examples with priors.
        '''
        if args.dataset in ["CIFAR-10", "MNIST", "FashionMNIST", "TinyImageNet"]:
            upsampler = lambda x: x
        else:
            upsampler = Upsample(size=(IMAGE_SIZE[args.dataset][0], IMAGE_SIZE[args.dataset][1]))
        with torch.no_grad():
            logit = target_model(images)

        q1_images_for_finetune = deque(maxlen=args.meta_seq_len)
        q2_images_for_finetune = deque(maxlen=args.meta_seq_len)
        q1_logits_for_finetune = deque(maxlen=args.meta_seq_len)
        q2_logits_for_finetune = deque(maxlen=args.meta_seq_len)

        pred = logit.argmax(dim=1)
        query = torch.zeros(args.batch_size).cuda()
        correct = pred.eq(true_labels).float()  # shape = (batch_size,)
        not_done = correct.clone()  # shape = (batch_size,)
        selected = torch.arange(batch_index * args.batch_size,
                                (batch_index + 1) * args.batch_size)  # 选择这个batch的所有图片的index
        if args.targeted:
            if args.target_type == 'random':
                target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset], size=true_labels.size()).long().cuda()
                invalid_target_index = target_labels.eq(true_labels)
                while invalid_target_index.sum().item() > 0:
                    target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                        size=target_labels[
                                                                            invalid_target_index].shape).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
            elif args.target_type == 'least_likely':
                target_labels = logit.argmin(dim=1)
            elif args.target_type == "increment":
                target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
            else:
                raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
        else:
            target_labels = None
        prior = torch.zeros(args.batch_size, IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset][0], IMAGE_SIZE[args.dataset][1])
        prior = prior.cuda()
        dim = prior.nelement() / args.batch_size
        prior_step = self.gd_prior_step if args.norm == 'l2' else self.eg_step
        image_step = self.l2_image_step if args.norm == 'l2' else self.linf_step
        proj_maker = self.l2_proj if args.norm == 'l2' else self.linf_proj  # 调用proj_maker返回的是一个函数
        proj_step = proj_maker(images, args.epsilon)
        criterion = self.cw_loss if args.loss == "cw" else self.xent_loss
        adv_images = images.clone()
        query_count = 0
        step_index = 1
        while query_count < args.max_queries:
            # Create noise for exporation, estimate the gradient, and take a PGD step
            exp_noise = args.exploration * torch.randn_like(prior) / (dim ** 0.5)  # parameterizes the exploration to be done around the prior
            # Query deltas for finite difference estimator
            exp_noise = exp_noise.cuda()
            q1 = upsampler(prior + exp_noise)  # 这就是Finite Difference算法， prior相当于论文里的v，这个prior也会更新，把梯度累积上去
            q2 = upsampler(prior - exp_noise)   # prior 相当于累积的更新量，用这个更新量，再去修改image，就会变得非常准
            # Loss points for finite difference estimator
            q1_images = adv_images + args.fd_eta * q1 / self.norm(q1)
            q2_images = adv_images + args.fd_eta * q2 / self.norm(q2)
            predict_by_target_model = False
            if step_index <= args.warm_up_steps or (step_index - args.warm_up_steps) % args.meta_predict_steps == 0:
                log.info("predict from target model")
                predict_by_target_model = True
                with torch.no_grad():
                    q1_logits = target_model(q1_images)
                    q2_logits = target_model(q2_images)
                q1_images_for_finetune.append(q1_images.detach())
                q2_images_for_finetune.append(q2_images.detach())
                q1_logits_for_finetune.append(q1_logits.detach())
                q2_logits_for_finetune.append(q2_logits.detach())
                if step_index > args.warm_up_steps:
                    q1_images_seq = torch.stack(list(q1_images_for_finetune)).permute(1, 0, 2, 3, 4).contiguous()  # B,T,C,H,W
                    q2_images_seq = torch.stack(list(q2_images_for_finetune)).permute(1, 0, 2, 3, 4).contiguous()  # B,T,C,H,W
                    q1_logits_seq = torch.stack(list(q1_logits_for_finetune)).permute(1, 0, 2).contiguous()  # B,T,#class
                    q2_logits_seq = torch.stack(list(q2_logits_for_finetune)).permute(1, 0, 2).contiguous()  # B,T,#class
                    finetune_times = random.randint(1,3)
                    meta_finetuner.finetune(q1_images_seq, q2_images_seq, q1_logits_seq, q2_logits_seq, finetune_times)
            else:
                with torch.no_grad():
                    q1_logits, q2_logits = meta_finetuner.predict(q1_images, q2_images)

            l1 = criterion(q1_logits, true_labels, target_labels)
            l2 = criterion(q2_logits, true_labels, target_labels)
            # Finite differences estimate of directional derivative
            est_deriv = (l1 - l2) / (args.fd_eta * args.exploration)  # 方向导数 , l1和l2是loss
            # 2-query gradient estimate
            est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise  # B, C, H, W,
            # Update the prior with the estimated gradient
            prior = prior_step(prior, est_grad, args.online_lr)  # 注意，修正的是prior,这就是bandit算法的精髓
            grad = upsampler(prior)  # prior相当于梯度
            ## Update the image:
            # take a pgd step using the prior
            image_lr = args.image_lr
            # if not_done.mean().item() <= 0.1:
            #     image_lr = 0.1  # < 0.15就很难了
            adv_images = image_step(adv_images, grad * correct.view(-1, 1, 1, 1), image_lr)  # prior放大后相当于累积的更新量，可以用来更新
            adv_images = proj_step(adv_images)
            adv_images = torch.clamp(adv_images, 0, 1)
            # check and stats
            with torch.no_grad():
                adv_logit = target_model(adv_images)
            adv_pred = adv_logit.argmax(dim=1)
            adv_prob = F.softmax(adv_logit, dim=1)
            adv_loss = self.xent_loss(adv_logit, true_labels, target_labels)
            ## Continue query count
            if predict_by_target_model:
                query_count += 2
                query = query + 2 * not_done
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels)).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels).float()  # 只要是跟原始label相等的，就还需要query，还没有成功
            success = (1 - not_done) * correct
            success_query = success * query
            not_done_loss = adv_loss * not_done
            not_done_prob = adv_prob[torch.arange(args.batch_size), true_labels] * not_done

            log.info('Attacking image {} - {} / {}, step {}, max query {}'.format(
                batch_index * args.batch_size, (batch_index + 1) * args.batch_size,
                self.total_images, step_index, int(query.max().item())
            ))
            log.info('        correct: {:.4f}'.format(correct.mean().item()))
            log.info('       not_done: {:.4f}'.format(not_done.mean().item()))
            log.info('      fd_scalar: {:.4f}'.format((l1 - l2).mean().item()))
            if success.sum().item() > 0:
                log.info('     mean_query: {:.4f}'.format(success_query[success.byte()].mean().item()))
                log.info('   median_query: {:.4f}'.format(success_query[success.byte()].median().item()))
            if not_done.sum().item() > 0:
                log.info('  not_done_loss: {:.4f}'.format(not_done_loss[not_done.byte()].mean().item()))
                log.info('  not_done_prob: {:.4f}'.format(not_done_prob[not_done.byte()].mean().item()))
            step_index += 1
            if not not_done.byte().any(): # all success
                break
        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_loss', 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来

    def attack_all_images(self, args, arch, target_model, meta_finetuner, result_dump_path):
        for batch_idx, data_tuple in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet":
                if target_model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]

            if images.size(-1) != target_model.input_size[-1]:
                images = F.interpolate(images, size=target_model.input_size[-1], mode='bilinear')
            self.make_adversarial_examples(batch_idx, images.cuda(), true_labels.cuda(), args, target_model,
                                                        meta_finetuner)
        query_all_ = self.query_all.detach().cpu().numpy().astype(np.int32)
        not_done_all_ = self.not_done_all.detach().cpu().numpy().astype(np.int32)
        query_threshold_success_rate, query_success_rate = success_rate_and_query_coorelation(query_all_, not_done_all_)
        success_rate_to_avg_query = success_rate_avg_query(query_all_, not_done_all_)
        log.info('{} is attacked finished ({} images)'.format(arch, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.info('     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            log.info('   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
        if self.not_done_all.sum().item() > 0:
            log.info('  avg not_done_loss: {:.4f}'.format(self.not_done_loss_all[self.not_done_all.byte()].mean().item()))
            log.info('  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all.mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "not_done_loss": self.not_done_loss_all[self.not_done_all.byte()].mean().item(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_threshold_success_rate_dict": query_threshold_success_rate,
                          "query_success_rate_dict": query_success_rate,
                          "success_rate_to_avg_query":success_rate_to_avg_query,
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, indent=4, sort_keys=True)


def get_exp_dir_name(dataset, norm, targeted, target_type):
    from datetime import datetime
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    dir_name = 'meta_simulator_bandits_-{}-{}-{}-{}'.format( dataset, norm, target_str, date_str)
    return dir_name

def print_args():
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

# def arange_sequence_to_meta_model(seq_id, warm_up_steps, meta_predict_steps, not_done_value):
#     # 步骤的序列安排:
#     # 0,1,...,19 target model; 然后finetune,之后meta model:20,21,22,23 target_model24, meta_model25,...,28,target_model 29
#     if seq_id in list(range(warm_up_steps)):
#        return False
#     else:
#         if not_done_value <= 0.16:
#             return False  # 用target model
#         if (seq_id - warm_up_steps + 1) % meta_predict_steps == 0:
#             return False
#         return True
#
# def check_if_finetune(seq_id, warm_up_steps, meta_predict_steps, finetune_times, not_done_value):
#     # finetune步骤:20, 25,30
#     seq_id = seq_id - warm_up_steps
#     if seq_id < 0:
#         return False, 0
#     if not_done_value <= 0.16:
#         return False, 0
#     if seq_id == 0:
#         return True, finetune_times
#     elif seq_id % meta_predict_steps == 0:
#         return True, random.randint(1,3)
#     return False, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--online-lr', type=float, help='Learning rate for the prior')
    parser.add_argument('--norm', type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--exploration', type=float,
                        help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile-size', type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--json-configures', type=str, help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, help='batch size for bandits')
    parser.add_argument('--log-progress', action='store_true')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--arch', default='wrn-28-10-drop', type=str, help='network architecture')
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target-type', default='increment', type=str, choices=['random', 'least_likely', "increment"],
                        help='how to choose target class for targeted attack, could be random or least_likely')
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')

    parser.add_argument("--meta_train_type", type=str, choices=["logits_distillation", "2q_distillation"])
    parser.add_argument("--meta_train_data", type=str, choices=["xent", "linf", "l2"])
    parser.add_argument("--distillation_loss", type=str, default="MSE", choices=["CSE", "MSE"])
    parser.add_argument("--finetune_times", type=int, default=20)
    parser.add_argument('--seed', default=int(time.time()), type=int, help='random seed')
    parser.add_argument('--phase', default='test', type=str, choices=['validation', 'test', "train"],
                        help='train, validation, test')

    parser.add_argument("--meta_predict_steps",type=int,default=60)
    parser.add_argument("--warm_up_steps", type=int,default=20)
    parser.add_argument("--meta_seq_len",type=int,default=20)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    print("using GPU {}".format(args.gpu))

    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
        args_dict = defaults

    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    set_log_file(osp.join(args.exp_dir, 'run.log'))

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    meta_finetuner = MetaModelFinetune(args.dataset, args.batch_size, args.meta_train_type, args.norm,
                                       args.distillation_loss)
    args.meta_model_path = meta_finetuner.meta_model_path
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args()
    target_model = StandardModel(args.dataset, args.arch, no_grad=True, train_data='full', epoch='final').eval()
    log.info("initializing target model {} on {}".format(args.arch, args.dataset))
    attacker = BanditsAttack(args)
    save_result_path = args.exp_dir + "/{}_result.json".format(arch)
    attacker.attack_all_images(args, arch, target_model, meta_finetuner, save_result_path)
