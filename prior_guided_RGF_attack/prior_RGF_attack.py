import glob
import os
import random
import sys
import numpy as np
import argparse
import glog as log
from torchvision.transforms import transforms

from config import IMAGE_SIZE, IN_CHANNELS, PY_ROOT
import torch
from torch.nn import functional as F
from torch import nn

from dataset_loader_maker import DataLoaderMaker
from model_constructor import ModelConstructor

class PriorRGFAttack(object):
    # 目前只能一张图一张图做对抗样本
    def __init__(self, dataset_name, model, surrogate_model):
        self.dataset_name = dataset_name
        self.image_height = IMAGE_SIZE[self.dataset_name][0]
        self.image_width =IMAGE_SIZE[self.dataset_name][1]
        self.in_channels = IN_CHANNELS[self.dataset_name]
        self.model = model
        self.surrogate_model = surrogate_model
        self.model.eval()
        self.surrogate_model.eval()
        self.targeted = False # only support untargeted attack now
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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

    def normalized_image(self, x):
        x_copy = x.clone()
        x_copy = torch.stack([self.normalize(x_copy[i]) for i in range(x.size(0))])
        return x_copy

    def attack_dataset(self, args, dataset_loader):
        if args.norm == 'l2':
            epsilon = 1e-3
            eps = np.sqrt(epsilon * self.image_height * self.image_width * 3)
            learning_rate = 2.0 / 299 / 1.7320508
        else:
            epsilon = 0.05
            eps= epsilon
            learning_rate = 0.005
        success = 0
        queries = []
        total = 0
        for batch_idx, (image_id, images, true_labels) in enumerate(dataset_loader):
            if batch_idx * args.batch_size >= args.number_images:
                break
            images = images.cuda()
            true_labels = true_labels.cuda()
            total += images.size(0)
            sigma = args.sigma
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            adv_images = images.clone()
            assert images.size(0) == 1
            if self.dataset_name == "ImageNet":
                logits_real_images = self.model(self.normalized_image(images))
            else:
                logits_real_images = self.model(images)
            l = self.xent_loss(logits_real_images, true_labels)  # 按照元素论文来写的，好奇怪
            lr = learning_rate
            total_q = 0
            ite = 0
            while total_q <= args.max_queries:
                total_q += 1
                if self.dataset_name == "ImageNet":
                    true = torch.squeeze(self.get_grad(self.model, self.normalized_image(adv_images), true_labels))  # C,H,W
                else:
                    true = torch.squeeze(self.get_grad(self.model, adv_images, true_labels))  # C,H,W
                log.info("Grad norm : {:.3f}".format(torch.sqrt(torch.sum(true * true)).item()))

                if ite % 2 == 0 and sigma != args.sigma:
                    log.info("checking if sigma could be set to be 1e-4")
                    rand = torch.randn_like(adv_images)
                    rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
                    if self.dataset_name == "ImageNet":
                        logits_1 = self.model(self.normalized_image(adv_images + args.sigma * rand))
                    else:
                        logits_1 = self.model(adv_images + args.sigma * rand)
                    rand_loss = self.xent_loss(logits_1, true_labels)  # shape = (batch_size,)
                    total_q += 1
                    rand =  torch.randn_like(adv_images)
                    rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
                    if self.dataset_name == "ImageNet":
                        logits_2 = self.model(self.normalized_image(adv_images + args.sigma * rand))
                    else:
                        logits_2 = self.model(adv_images + args.sigma * rand)
                    rand_loss2= self.xent_loss(logits_2, true_labels) # shape = (batch_size,)
                    total_q += 1
                    if (rand_loss - l)[0].item() != 0 and (rand_loss2 - l)[0].item() != 0:
                        sigma = args.sigma
                        log.info("set sigma back to 1e-4, sigma={:.4f}".format(sigma))

                if args.method != "uniform":
                    if self.dataset_name == "ImageNet":
                        prior = torch.squeeze(self.get_grad(self.surrogate_model, self.normalized_image(adv_images), true_labels))  # C,H,W
                    else:
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
                        if self.dataset_name == "ImageNet":
                            losses = self.xent_loss(self.model(self.normalized_image(eval_points)), true_labels.repeat(s))  # shape = (10*B,)
                        else:
                            losses = self.xent_loss(self.model(eval_points), true_labels.repeat(s))  # shape = (10*B,)
                        total_q += s
                        norm_square = torch.mean(((losses - l) / sigma) ** 2) # scalar
                    while True:
                        if self.dataset_name == "ImageNet":
                            logits_for_prior_loss = self.model(self.normalized_image(adv_images + sigma * prior))  # prior may be C,H,W
                        else:
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
                        if self.dataset_name == "ImageNet":
                            logits_ = self.model(self.normalized_image(eval_points))
                        else:
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
                            if self.dataset_name == "ImageNet":
                                logits_p = model(self.normalized_image(adv_images + ss * direction))
                            else:
                                logits_p = model(adv_images + ss * direction)
                            loss_p = self.xent_loss(logits_p, true_labels)
                            les.append((loss_p - l)[0].item())
                        log.info("losses: ".format(les))

                    if args.show_loss:
                        if args.method == 'biased' or args.method == 'fixed_biased':
                            show_input = adv_images + lr * prior
                            if self.dataset_name == "ImageNet":
                                show_input = self.normalized_image(show_input)
                            logits_show = self.model(show_input)
                            lprior = self.xent_loss(logits_show, true_labels) - l
                            print_loss(self.model, prior)
                            show_input_2 = adv_images + lr * grad
                            if self.dataset_name == "ImageNet":
                                show_input_2 = self.normalized_image(show_input_2)
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
                if self.dataset_name == "ImageNet":
                    adv_labels = self.get_pred(self.model, self.normalized_image(adv_images))
                    logits_ = self.model(self.normalized_image(adv_images))
                else:
                    adv_labels = self.get_pred(self.model, adv_images)
                    logits_ = self.model(adv_images)

                l = self.xent_loss(logits_, true_labels)
                log.info('queries:', total_q, 'loss:', l, 'learning rate:', lr, 'sigma:', sigma, 'prediction:', adv_labels,
                      'distortion:', torch.max(torch.abs(adv_images - images)).item(), torch.norm(adv_images - images).item())
                ite += 1
                if adv_labels[0].item() != true_labels[0].item():
                    log.info("Stop at queries : {}".format(total_q))
                    success += 1
                    queries.append(total_q)
                    break
            else:
                # images converts to H,W,C
                np.save(args.exp_dir + "/failed_images/{}.npy".format(image_id[0].item(),
                                                                      np.transpose(images[0].detach().cpu().numpy(),axes=(1,2,0))))
                log.info("Failed at image id {}, save image to {}".format(image_id[0].item(),
                                                                    args.exp_dir + "/failed_images/{}.npy".format(image_id[0].item())))

        log.info('Success rate: {:.3f} Queries_mean: {:.3f} Queries_median: {:.3f}'.format(success/total,
                                                                                           np.mean(queries), np.median(queries)))


def get_random_dir_name(dataset, arch, surrogate_arch, norm):
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = 'P-RGF_attack_{}_arch_{}_surrogate_arch_{}_attack_'.format(dataset, arch,surrogate_arch,norm)\
              + dirname + ''.join(random.choice(vocab) for _ in range(8))
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
                        choices=['CIFAR-10', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],help='which dataset to use')
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='random', choices=["random", "least_likely"])
    parser.add_argument("--arch", type=str, required=True, help='The architecture of target model, '
                                                                'in original paper it is inception-v3')
    parser.add_argument("--surrogate_arch", type=str, required=True, help="The architecture of surrogate model,"
                                                                          " in original paper it is resnet152")
    parser.add_argument("--norm",type=str,default="l2", choices=["l2", "linf"], help='The norm used in the attack.')
    parser.add_argument("--method",type=str,default="biased", choices=['uniform', 'biased', 'fixed_biased'], help='Methods used in the attack.')
    parser.add_argument("--dataprior", action="store_true", help="Whether to use data prior in the attack.")
    parser.add_argument("--show_loss", action="store_true", help="Whether to print loss in some given step sizes.")
    parser.add_argument("--samples_per_draw",type=int, default=50, help="Number of samples to estimate the gradient.")
    parser.add_argument("--sigma", type=float,default=1e-4, help="Sampling variance.")
    parser.add_argument("--number_images", type=int, default=100000,  help='Number of images for evaluation.')
    parser.add_argument("--max_queries", type=int, default=10000, help="Maximum number of queries.")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    assert args.batch_size == 1, 'The code does not support batch_size > 1 yet.'
    args.exp_dir = os.path.join(args.exp_dir, get_random_dir_name(args.dataset, args.arch, args.surrogate_arch, args.norm))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir + "/failed_images/", exist_ok=True)
    set_log_file(os.path.join(args.exp_dir, 'run.log'))
    log.info("using GPU {}".format(args.gpu))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)
    torch.backends.cudnn.deterministic = True
    if args.dataset in ["CIFAR-10", "MNIST", "FashionMNIST"]:
        model = ModelConstructor.construct_cifar_model(args.arch, args.dataset)
        surrogate_model = ModelConstructor.construct_cifar_model(args.surrogate_arch, args.dataset)
    elif args.dataset == "TinyImageNet":
        model = ModelConstructor.construct_tiny_imagenet_model(args.arch, args.dataset)
        surrogate_model = ModelConstructor.construct_tiny_imagenet_model(args.surrogate_arch, args.dataset)
    elif args.dataset == "ImageNet":
        model = ModelConstructor.construct_imagenet_model(args.arch)
        surrogate_model = ModelConstructor.construct_imagenet_model(args.surrogate_arch)

    if args.dataset != "ImageNet":
        target_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(root=PY_ROOT, dataset=args.dataset,
                                                                                                        arch=args.arch)
        surrogate_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(root=PY_ROOT,
                                                                                                           dataset=args.dataset,
                                                                                                           arch=args.surrogate_arch)
        target_model_path = list(glob.glob(target_model_path))[0]
        surrogate_model_path = list(glob.glob(surrogate_model_path))[0]
        model.load_state_dict(torch.load(target_model_path, map_location=lambda storage, location: storage)["state_dict"])
        surrogate_model.load_state_dict(torch.load(surrogate_model_path,
                                                   map_location=lambda storage, location: storage)["state_dict"])
    model.cuda()
    surrogate_model.cuda()
    phase = "validation" if "ImageNet" in args.dataset else "test"
    data_loader = DataLoaderMaker.get_imgid_img_label_data_loader(args.dataset, args.batch_size, False)
    attacker = PriorRGFAttack(args.dataset, model, surrogate_model)
    with torch.no_grad():
        attacker.attack_dataset(args, data_loader)
    log.info("Construct target model {} and surrogate model {} done. Now attacking {}!".format(args.arch, args.surrogate_arch, args.dataset))

