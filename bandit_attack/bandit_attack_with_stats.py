import sys

from attacker_with_statistics.attacker_statistics_base import Attacker

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

from config import IMAGE_SIZE, IN_CHANNELS
from target_models.standard_model import StandardModel


class BanditsAttack(Attacker):
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

    ##
    # Projection steps for l2 and linf constraints:
    # All take the form of func(new_x, old_x, epsilon)
    ##
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

    def make_adv_examples_iteration(self, step_index, adv_images, true_labels, target_labels, args):
        if step_index == 0:  # initialization
            if self.dataset_name in ["CIFAR-10", "MNIST", "FashionMNIST", "TinyImageNet"]:
                upsampler = lambda x: x
            else:
                upsampler = Upsample(size=(IMAGE_SIZE[self.dataset_name][0], IMAGE_SIZE[self.dataset_name][1]))
            prior_step = self.gd_prior_step if args.norm == 'l2' else self.eg_step
            image_step = self.l2_image_step if args.norm == 'l2' else self.linf_step
            proj_maker = self.l2_proj if args.norm == 'l2' else self.linf_proj  # 调用proj_maker返回的是一个函数
            proj_step = proj_maker(adv_images.clone(), args.epsilon)
            prior = torch.zeros(args.batch_size, IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset][0],
                                IMAGE_SIZE[args.dataset][1]).cuda()
            dim = prior.nelement() / args.batch_size  # nelement() --> total number of elements

        # Create noise for exporation, estimate the gradient, and take a PGD step
        exp_noise = args.exploration * torch.randn_like(prior) / (dim ** 0.5)  # parameterizes the exploration to be done around the prior
        # Query deltas for finite difference estimator
        exp_noise = exp_noise.cuda()
        q1 = upsampler(prior + exp_noise)  # 这就是Finite Difference算法， prior相当于论文里的v，这个prior也会更新，把梯度累积上去
        q2 = upsampler(prior - exp_noise)  # prior 相当于累积的更新量，用这个更新量，再去修改image，就会变得非常准
        # Loss points for finite difference estimator

        q1_images = adv_images + args.fd_eta * q1 / self.norm(q1)
        q2_images = adv_images + args.fd_eta * q2 / self.norm(q2)
        with torch.no_grad():
            q1_logits = target_model(q1_images)
            q2_logits = target_model(q2_images)
        l1 = self.xent_loss(q1_logits, true_labels, target_labels)
        l2 = self.xent_loss(q2_logits, true_labels, target_labels)
        # Finite differences estimate of directional derivative
        est_deriv = (l1 - l2) / (args.fd_eta * args.exploration)  # 方向导数 , l1和l2是loss
        # 2-query gradient estimate
        est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise  # B, C, H, W,
        # Update the prior with the estimated gradient
        prior = prior_step(prior, est_grad, args.online_lr)  # 注意，修正的是prior,这就是bandit算法的精髓
        grad = upsampler(prior)  # prior相当于梯度
        # Take a pgd step using the prior to update images
        # adv_images = image_step(adv_images, grad * correct.view(-1, 1, 1, 1),
        #                         args.image_lr)  # prior放大后相当于累积的更新量，可以用来更新
        adv_images = image_step(adv_images, grad, args.image_lr)
        adv_images = proj_step(adv_images)
        adv_images = torch.clamp(adv_images, 0, 1)
        return adv_images, False, True

def get_random_dir_name(mode):
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = 'bandits_attack_{}_'.format(mode) + dirname + ''.join(random.choice(vocab) for _ in range(8))
    return dirname

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
    parser.add_argument('--json-config', type=str, help='a config file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, help='batch size for bandits')
    parser.add_argument('--log-progress', action='store_true')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--arch', default='wrn-28-10-drop', type=str, help='network architecture')
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='random', choices=["random", "least_likely"])
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--seed', default=int(time.time()), type=int, help='random seed')
    parser.add_argument('--phase', default='test', type=str, choices=['validation', 'test', "train"],
                        help='train, validation, test')
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

    args.exp_dir = osp.join(args.exp_dir, get_random_dir_name(args.norm))  # 随机产生一个目录用于实验
    if not osp.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    set_log_file(osp.join(args.exp_dir, 'run.log'))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args()
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    target_model = StandardModel(args.dataset, args.arch, no_grad=True, train_data='full', epoch='final').eval()
    log.info("initializing target model {} on {}".format(args.arch, args.dataset))

    attacker = BanditsAttack(target_model, args.targeted, args.target_type, args.dataset, args.batch_size)
    result_dump_path = args.exp_dir + "/hyper_params_and_result.json"
    attacker.attack_dataset(args, args.max_queries, result_dump_path)
