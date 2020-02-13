import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
import glob
import json
import os

from types import SimpleNamespace

import glog as log
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import IMAGE_SIZE, IN_CHANNELS, PY_ROOT, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.model_constructor import StandardModel
from utils.statistics_toolkit import success_rate_and_query_coorelation, success_rate_avg_query


class PriorRGFAttack(object):
    # 目前只能一张图一张图做对抗样本
    def __init__(self, dataset_name, model, surrogate_model, targeted, target_type):
        self.dataset_name = dataset_name
        self.data_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, 1)
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


    def xent_loss(self, logit, true_label, target=None):
        if self.targeted:
            return -F.cross_entropy(logit, target, reduction='none')
        else:
            return F.cross_entropy(logit, true_label, reduction='none')

    def get_grad(self, model, x, labels):
        with torch.enable_grad():
            x.requires_grad_()
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
            gradient = torch.autograd.grad(loss, x)[0]
        return gradient

    def get_pred(self, model, x):
        with torch.no_grad():
            logits = model(x)
        return logits.max(1)[1]


    def attack_dataset(self, args, arch, result_dump_path):

        success = 0
        queries = []
        not_done = []
        correct_all = []
        total = 0
        for batch_idx, data_tuple in enumerate(self.data_loader):
            if args.dataset == "ImageNet":
                if self.model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]

            if images.size(-1) != self.model.input_size[-1]:
                images = F.interpolate(images, size=self.model.input_size[-1], mode='bilinear')
            self.image_height = images.size(2)
            self.image_width = images.size(3)

            if args.norm == 'l2':
                # epsilon = 1e-3
                # eps = np.sqrt(epsilon * model.input_size[-1] * model.input_size[-1] * self.in_channels)  # 1.752
                eps = 4.6
                learning_rate = 2.0 / np.sqrt(self.image_height * self.image_width * self.in_channels)
            else:
                epsilon = 0.031372
                if args.dataset == "ImageNet":
                    epsilon = 0.05
                eps = epsilon
                learning_rate = 0.005

            images = images.cuda()
            true_labels = true_labels.cuda()

            with torch.no_grad():
                logits = self.model(images)
                pred = logits.argmax(dim=1)
                correct = pred.eq(true_labels).detach().cpu().numpy().astype(np.int32)
                correct_all.append(correct)
                if correct[0].item() == 0:
                    queries.append(0)
                    not_done.append(0)  # 原本就分类错了，not_done = 0
                    log.info("The {}-th image is already classified incorrectly.")
                    continue


            total += images.size(0)
            sigma = args.sigma
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            adv_images = images.clone().cuda()
            assert images.size(0) == 1
            logits_real_images = self.model(images)
            l = self.xent_loss(logits_real_images, true_labels)  # 按照元素论文来写的，好奇怪
            lr = float(learning_rate)
            total_q = 0
            ite = 0
            while total_q <= args.max_queries:
                total_q += 1
                true = torch.squeeze(self.get_grad(self.model, adv_images, true_labels))  # C,H,W
                log.info("Grad norm : {:.3f}".format(torch.sqrt(torch.sum(true * true)).item()))

                if ite % 2 == 0 and sigma != args.sigma:
                    log.info("checking if sigma could be set to be 1e-4")
                    rand = torch.randn_like(adv_images)
                    rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
                    logits_1 = self.model(adv_images + args.sigma * rand)
                    rand_loss = self.xent_loss(logits_1, true_labels)  # shape = (batch_size,)
                    total_q += 1
                    rand =  torch.randn_like(adv_images)
                    rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
                    logits_2 = self.model(adv_images + args.sigma * rand)
                    rand_loss2= self.xent_loss(logits_2, true_labels) # shape = (batch_size,)
                    total_q += 1
                    if (rand_loss - l)[0].item() != 0 and (rand_loss2 - l)[0].item() != 0:
                        sigma = args.sigma
                        log.info("set sigma back to 1e-4, sigma={:.4f}".format(sigma))

                if args.method != "uniform":
                    prior = torch.squeeze(self.get_grad(self.surrogate_model, adv_images, true_labels))  # C,H,W
                    alpha = torch.sum(true * prior) / torch.clamp(torch.sqrt(torch.sum(true * true) * torch.sum(prior * prior)), min=1e-12)
                    log.info("alpha = {:.3}".format(alpha))
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
                        losses = self.xent_loss(self.model(eval_points), true_labels.repeat(s))  # shape = (10*B,)
                        total_q += s
                        norm_square = torch.mean(((losses - l) / sigma) ** 2) # scalar
                    while True:
                        logits_for_prior_loss = self.model(adv_images + sigma* prior) # prior may be C,H,W
                        prior_loss = self.xent_loss(logits_for_prior_loss, true_labels)  # shape = (batch_size,)
                        total_q += 1
                        diff_prior = (prior_loss - l)[0].item()   # FIXME batch模式下是否[0]?
                        if diff_prior == 0:
                            sigma *= 2
                            log.info("sigma={:.4f}, multiply sigma by 2".format(sigma))
                        else:
                            break
                    est_alpha = diff_prior / sigma / torch.clamp(torch.sqrt(torch.sum(torch.mul(prior,prior)) * norm_square), min=1e-12)
                    est_alpha = est_alpha.item()
                    log.info("Estimated alpha = {:.3f}".format(est_alpha))
                    alpha = est_alpha
                    if alpha < 0:
                        prior = -prior
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
                    log.info("best_lambda = {:.4f}".format(best_lambda))
                    if best_lambda < 1 and best_lambda > 0:
                        lmda = best_lambda
                    else:
                        if alpha ** 2 * (n + 2 * q - 2) < 1:
                            lmda = 0
                        else:
                            lmda = 1
                    if abs(alpha) >= 1:
                        lmda = 1
                    log.info("lambda = {:.3f}".format(lmda))
                    if lmda == 1:
                        return_prior = True
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
                            pert[i] = np.sqrt(1-lmda) * pert[i] + np.sqrt(lmda) * prior
                        else:
                            pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))),min=1e-12)
                    while True:
                        eval_points = adv_images + sigma * pert  # (1,C,H,W)  pert=(q,C,H,W)
                        logits_ = self.model(eval_points)
                        losses = self.xent_loss(logits_, true_labels.repeat(q))  # shape = (q,)
                        total_q += q
                        grad = (losses - l).view(-1, 1, 1, 1) * pert  # (q,1,1,1) * (q,C,H,W)
                        grad = torch.mean(grad,dim=0,keepdim=True)  # 1,C,H,W
                        norm_grad = torch.sqrt(torch.mean(torch.mul(grad,grad)))
                        if norm_grad.item() == 0:
                            sigma *= 5
                            log.info("estimated grad == 0, multiply sigma by 5. Now sigma={:.4f}".format(sigma))
                        else:
                            break
                    grad = grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))), min=1e-12)

                    def print_loss(model, direction):
                        length = [1e-4, 1e-3]
                        les = []
                        for ss in length:
                            logits_p = model(adv_images + ss * direction)
                            loss_p = self.xent_loss(logits_p, true_labels)
                            les.append((loss_p - l)[0].item())
                        log.info("losses: ".format(les))

                    if args.show_loss:
                        if args.method == 'biased' or args.method == 'fixed_biased':
                            show_input = adv_images + lr * prior
                            logits_show = self.model(show_input)
                            lprior = self.xent_loss(logits_show, true_labels) - l
                            print_loss(self.model, prior)
                            show_input_2 = adv_images + lr * grad
                            logits_show2 = self.model(show_input_2)
                            lgrad = self.xent_loss(logits_show2, true_labels) - l
                            print_loss(self.model, grad)
                            log.info(lprior, lgrad)
                else:
                    grad = prior
                log.info("angle = {:.4f}".format(torch.sum(true*grad) /
                                                 torch.clamp(torch.sqrt(torch.sum(true*true) * torch.sum(grad*grad)),min=1e-12)))
                if args.norm == "l2":
                    adv_images = adv_images + lr * grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))),min=1e-12)
                    norm = torch.clamp(torch.norm(adv_images - images),min=1e-12).item()
                    factor = min(1, eps / norm)
                    adv_images = images + (adv_images - images) * factor
                else:
                    adv_images = adv_images + lr * torch.sign(grad)
                    adv_images = torch.min(torch.max(adv_images, images - eps), images + eps)
                adv_images = torch.clamp(adv_images, self.clip_min, self.clip_max)
                adv_labels = self.get_pred(self.model, adv_images)
                logits_ = self.model(adv_images)
                l = self.xent_loss(logits_, true_labels)
                log.info('queries:', total_q, 'loss:', l, 'learning rate:', lr, 'sigma:', sigma, 'prediction:', adv_labels,
                      'distortion:', torch.max(torch.abs(adv_images - images)).item(), torch.norm(adv_images - images).item())
                ite += 1
                if adv_labels[0].item() != true_labels[0].item():
                    log.info("Stop at queries : {}".format(total_q))
                    success += 1
                    not_done.append(0)
                    queries.append(total_q)
                    break
            else:
                not_done.append(1)
                queries.append(args.max_queries) # 因此不能用np.mean(queries)来计算，平均query次数



        log.info('Attack {} success rate: {:.3f} Queries_mean: {:.3f} Queries_median: {:.3f}'.format(arch, success/total,
                                                                                           np.mean(queries), np.median(queries)))
        correct_all = np.concatenate(correct_all, axis=0).astype(np.int32)
        query_all = np.array(queries).astype(np.int32)
        not_done_all = np.array(not_done).astype(np.int32)
        success = (1 - not_done_all) * correct_all
        success_query = success * query_all
        query_threshold_success_rate, query_success_rate = success_rate_and_query_coorelation(query_all, not_done_all)
        success_rate_to_avg_query = success_rate_avg_query(query_all, not_done_all)
        meta_info_dict = {"query_all":query_all.tolist(),"not_done_all":not_done_all.tolist(),
                          "correct_all":correct_all.tolist(),
                          "mean_query": np.mean(success_query[np.nonzero(success)[0]]).item(),
                          "max_query":np.max(success_query[np.nonzero(success)[0]]).item(),
                          "median_query": np.median(success_query[np.nonzero(success)[0]]).item(),
                          "query_threshold_success_rate_dict": query_threshold_success_rate,
                          "query_success_rate_dict": query_success_rate,
                          "success_rate_to_avg_query": success_rate_to_avg_query}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def get_expr_dir_name(dataset, method, surrogate_arch, norm, targeted, target_type):
    from datetime import datetime
    # dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    dirname = 'P-RGF_{}_attack_{}_surrogate_arch_{}_{}_{}'.format(method, dataset,surrogate_arch,norm,target_str)
    return dirname

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


def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],help='which dataset to use')
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target-type', default='increment', type=str, choices=['random', 'least_likely', "increment"],
                        help='how to choose target class for targeted attack, could be random or least_likely')
    parser.add_argument("--arch", type=str, help='The architecture of target model, '
                                                                'in original paper it is inception-v3')
    parser.add_argument("--test_archs",action="store_true")
    parser.add_argument("--surrogate_arch", type=str, default="resnet-110", help="The architecture of surrogate model,"
                                                                          " in original paper it is resnet152")
    parser.add_argument("--norm",type=str, required=True, choices=["l2", "linf"], help='The norm used in the attack.')
    parser.add_argument("--method",type=str,default="biased", choices=['uniform', 'biased', 'fixed_biased'],
                        help='Methods used in the attack. uniform: RGF, biased: P-RGF (\lambda^*), fixed_biased: P-RGF (\lambda=0.5)')
    parser.add_argument("--dataprior", default=None, action="store_true", help="Whether to use data prior in the attack.")
    parser.add_argument('--json-config', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/configures/prior_RGF_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument("--show_loss", action="store_true", help="Whether to print loss in some given step sizes.")
    parser.add_argument("--samples_per_draw",type=int, default=50, help="Number of samples to estimate the gradient.")
    parser.add_argument("--sigma", type=float,default=1e-4, help="Sampling variance.")
    # parser.add_argument("--number_images", type=int, default=100000,  help='Number of images for evaluation.')
    parser.add_argument("--max_queries", type=int, default=10000, help="Maximum number of queries.")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    assert args.batch_size == 1, 'The code does not support batch_size > 1 yet.'
    args.exp_dir = os.path.join(args.exp_dir, get_expr_dir_name(args.dataset, args.method, args.surrogate_arch, args.norm,
                                                                args.targeted, args.target_type))
    os.makedirs(args.exp_dir, exist_ok=True)
    set_log_file(os.path.join(args.exp_dir, 'run.log'))
    defaults = json.load(open(args.json_config))[args.dataset]
    arg_vars = vars(args)
    arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
    defaults.update(arg_vars)
    args = SimpleNamespace(**defaults)


    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 50000

    torch.backends.cudnn.deterministic = True
    surrogate_model = StandardModel(args.dataset, args.surrogate_arch, False)

    if args.test_archs:
        archs = []
        for arch in MODELS_TEST_STANDARD[args.dataset]:
            if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT,
                    args.dataset, arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
            elif args.dataset == "ImageNet":
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}-*.pth".format(
                    PY_ROOT,
                    args.dataset, arch)
                test_model_path = list(glob.glob(test_model_list_path))
                if test_model_path and os.path.exists(test_model_path[0]):
                    archs.append(arch)
    else:
        archs = [args.arch]
    args.arch = ", ".join(archs)
    log.info("using GPU {}".format(args.gpu))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)
    for arch in archs:
        save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        model = StandardModel(args.dataset, arch, no_grad=False)
        model.cuda()
        model.eval()
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        attacker = PriorRGFAttack(args.dataset, model, surrogate_model, args.targeted, args.target_type)
        with torch.no_grad():
            attacker.attack_dataset(args, arch, save_result_path)
        attacker.model.cpu()
        log.info("Attack {} with surrogate model {} done!".format(arch, args.surrogate_arch))
