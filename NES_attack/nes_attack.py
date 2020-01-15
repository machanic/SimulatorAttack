import os
import random
import sys
from collections import defaultdict

import argparse
import glog as log
import json
import numpy as np
import torch
from torch.nn import functional as F
from types import SimpleNamespace

from attacker_with_statistics.attacker_statistics_base import Attacker
from config import CLASS_NUM
from target_models.standard_model import StandardModel


class NesAttacker(Attacker):
    def __init__(self, dataset_name, *args):
        super(NesAttacker, self).__init__(*args)
        self.dataset_name = dataset_name
        self.num_classes = CLASS_NUM[self.dataset_name]
        self.dataset = self.dataset_loader.dataset

    def get_label_dataset(self, dataset):
        label_index = defaultdict(list)
        for index, label in enumerate(dataset.labels):  # 保证这个data_loader没有shuffle过
            label_index[label].append(index)

    def get_image_of_class(self, target_labels, dataset):
        images = []
        label_index = self.get_label_dataset(dataset)
        for label in target_labels:
            index = random.choice(label_index[label])
            image, *_ = dataset[index]
            images.append(image)
        return torch.stack(images)  # B,C,H,W

    def xent_loss(self, logit, true_label, target=None):
        if self.targeted:
            return -F.cross_entropy(logit, target, reduction='none')
        else:
            return F.cross_entropy(logit, true_label, reduction='none')

    # @deprecated
    # def partial_info_loss(self, input, labels,target_labels, noise):
    #     logits = self.model(input)
    #     loss = F.cross_entropy(logits, labels)
    #     vals, inds = torch.topk(logits, k=self.top_k)  # inds shape = batch_size, k
    #     inds = inds.detach().cpu().numpy()
    #     target_labels = target_labels.detach().cpu().numpy()
    #
    #     good_inds = np.where(inds == target_labels)  # FIXME
    #     good_images = good_inds[:, 0]
    #     good_images = torch.from_numpy(good_images).cuda()
    #     loss = torch.gather(loss, 0, good_images)
    #     noise = torch.gather(noise,0,good_images)
    #     return loss, noise

    #  STEP CONDITION (important for partial-info attacks)
    def robust_in_top_k(self, target_labels, adv_images, top_k): # target_label shape=(batch_size,)
        if top_k == self.num_classes:
            return True
        eval_logits = self.model(adv_images)
        _,  top_pred_indices = eval_logits.topk(k=top_k,largest=True,sorted=True)  # top_pred_indices shape = (batch_size, top_k)
        target_labels = target_labels.view(-1,1).repeat(1, top_k)
        exists_check_array = (target_labels == top_pred_indices).byte()
        exists_check_array = exists_check_array.cpu().detach().numpy()  # shape = (batch_size, top_k)
        exists_check_array = np.bitwise_or.reduce(exists_check_array, axis=1)  # shape = (batch_size)
        if np.all(exists_check_array):  # all is True: all target_label is in top_k prediction
            return True
        return False

    def norm(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def get_grad(self, x, true_labels, target_labels):
        # loss_fn = self.partial_info_loss if k < self.num_classes else self.xent_loss
        dim = x.size(1) * x.size(2) * x.size(3)
        noise = torch.randn_like(x) / (dim ** 0.5)
        query_1 = x + self.sigma * noise
        query_2 = x - self.sigma * noise
        logits_q1 = self.model(query_1)
        logits_q2 = self.model(query_2)
        loss_1 = self.xent_loss(logits_q1, true_labels, target_labels)  # losses shape = (batch_size,)
        loss_2 = self.xent_loss(logits_q2, true_labels, target_labels)  # losses shape = (batch_size,)
        est_deriv = (loss_1 - loss_2) / self.sigma
        grad_estimate = est_deriv.view(-1,1,1,1) * noise  # B,C,H,W
        final_loss = (loss_1 + loss_2) / 2.0  #  shape = (batch_size,)
        return grad_estimate, final_loss

    def untargeted_attack_iteration(self, step_index, adv_images, true_labels, target_labels, args):  # like PGD
        if step_index == 0:
            initial_img = adv_images.clone()
            g = torch.zeros_like(initial_img).cuda()  # 梯度，(B,C,H,W)
        prev_g = g.clone()
        g, _ = self.get_grad(adv_images, true_labels, target_labels)
        # SIMPLE MOMENTUM
        g = args.momentum * prev_g + (1.0 - args.momentum) * g
        adv_images = adv_images + args.lr * torch.sign(g)
        eta = torch.clamp(adv_images - initial_img, min=-args.epsilon, max=args.epsilon)
        proposed_adv = torch.clamp(initial_img + eta, min=args.clip_min, max=args.clip_max).detach_()
        return proposed_adv


    def targeted_attack_iteration(self, step_index, adv_images, true_labels, target_labels, args):
        if step_index == 0:
            initial_img = adv_images.clone()
            last_loss = []
            adv_thresh = args.adv_thresh
            max_lr = args.max_lr
            goal_epsilon = args.epsilon
            clip_min = args.clip_min
            clip_max = args.clip_max
            g = torch.zeros_like(initial_img).cuda()  # 梯度，(B,C,H,W)
            adv_images = self.get_image_of_class(target_labels, self.dataset)
            epsilon = args.starting_eps
            delta_epsilon = args.starting_delta_eps
            lower = torch.clamp(initial_img - epsilon, clip_min, clip_max)
            upper = torch.clamp(initial_img + epsilon, clip_min, clip_max)
            adv_images = torch.min(torch.max(adv_images, lower), upper)
        prev_g = g.clone()
        g, l = self.get_grad(adv_images, true_labels, target_labels)
        # SIMPLE MOMENTUM
        g = args.momentum * prev_g + (1.0 - args.momentum) * g
        # PLATEAU LR ANNEALING
        last_loss.append(l)
        last_loss = last_loss[-args.plateau_length:]
        if last_loss[-1] > last_loss[0] and len(last_loss) == args.plateau_length:
            if max_lr > args.min_lr:
                print("[log] Annealing max_lr")
                max_lr = max(max_lr / args.plateau_drop, args.min_lr)
            last_loss.clear()
        # SEARCH FOR LR AND EPSILON DECAY
        current_lr = max_lr
        proposed_adv = adv_images - current_lr * torch.sign(g)
        prop_de = 0.0
        if l < adv_thresh and epsilon > goal_epsilon:
            prop_de = delta_epsilon
        while current_lr > args.min_lr:
            proposed_adv = adv_images - int(self.targeted) * current_lr * torch.sign(g)
            lower = torch.clamp(initial_img - epsilon, clip_min, clip_max)
            upper = torch.clamp(initial_img + epsilon, clip_min, clip_max)
            proposed_adv = torch.min(torch.max(proposed_adv, lower), upper)
            # if self.robust_in_top_k(target_labels, proposed_adv, k):  # target_label=(batch_size, 10),
            #     adv_images = proposed_adv
            #     epsilon = max(epsilon - prop_de / args.conservative, goal_epsilon)
            #     return adv_images
            if current_lr >= args.min_lr * 2:
                current_lr = current_lr / 2
            else:
                prop_de = prop_de / 2
                if prop_de == 0:
                    raise ValueError("Did not converge.")
                if prop_de < 2e-3:
                    prop_de = 0
                # current_lr = max_lr
                log.info("[log] backtracking eps to %3f" % (epsilon - prop_de,))
            epsilon = max(epsilon - prop_de / args.conservative, goal_epsilon)  # goal_epsilon是最低的eps

        return proposed_adv

    def make_adv_examples_iteration(self, step_index, adv_images, true_labels, target_labels, args):
        if self.targeted:
            return self.targeted_attack_iteration(step_index, adv_images, true_labels, target_labels, args)
        else:
            return self.untargeted_attack_iteration(step_index, adv_images, true_labels, target_labels, args)

def get_random_dir_name(dataset, arch, mode):
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = 'nes_attack_{}_{}_{}_'.format(dataset, arch, mode) + dirname + ''.join(random.choice(vocab) for _ in range(8))
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

def print_args():
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--samples-per-draw', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--target-class', type=int, help='negative => untargeted')
    parser.add_argument('--orig-class', type=int)
    parser.add_argument('--sigma', type=float, default=1e-3)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--img-index', type=int)
    parser.add_argument('--out-dir', type=str, required=True,
                        help='dir to save to if not gridding; otherwise parent \
                            dir of grid directories')
    parser.add_argument('--log-iters', type=int, default=1)
    parser.add_argument('--restore', type=str, help='restore path of img')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--save-iters', type=int, default=50)
    parser.add_argument('--plateau-drop', type=float, default=2.0)
    parser.add_argument('--min-lr-ratio', type=int, default=200)
    parser.add_argument('--plateau-length', type=int, default=5)
    parser.add_argument('--gpus', type=int, help='number of GPUs to use')
    parser.add_argument('--imagenet-path', type=str)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max-lr', type=float, default=1e-2)
    parser.add_argument('--min-lr', type=float, default=5e-5)
    # PARTIAL INFORMATION ARGUMENTS
    parser.add_argument('--top-k', type=int, default=-1)
    parser.add_argument('--adv-thresh', type=float, default=-1.0)
    # LABEL ONLY ARGUMENTS
    parser.add_argument('--label-only', action='store_true')
    parser.add_argument('--zero-iters', type=int, default=100, help="how many points to use for the proxy score")
    parser.add_argument('--label-only-sigma', type=float, default=1e-3, help="distribution width for proxy score")
    parser.add_argument('--starting-eps', type=float, default=1.0)
    parser.add_argument('--starting-delta-eps', type=float, default=0.5)
    parser.add_argument('--min-delta-eps', type=float, default=0.1)
    parser.add_argument('--conservative', type=int, default=2,
                        help="How conservative we should be in epsilon decay; increase if no convergence")
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],help='which dataset to use')
    parser.add_argument('--arch', default='wrn-28-10-drop', type=str, help='network architecture')
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='random', choices=["random", "least_likely"])

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    print("using GPU {}".format(args.gpu))
    args.exp_dir = os.path.join(args.exp_dir, get_random_dir_name(args.dataset,  args.arch, "targeted" if args.targeted else "untargeted"))  # 随机产生一个目录用于实验
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    set_log_file(os.path.join(args.exp_dir, 'run.log'))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args()
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    target_model = StandardModel(args.dataset, args.arch, no_grad=True, train_data='full', epoch='final').eval()
    log.info("initializing target model {} on {}".format(args.arch, args.dataset))
    attacker = NesAttacker(args.dataset, target_model, args.targeted, args.target_type, args.dataset, args.batch_size)
    result_dump_path = args.exp_dir + "/hyper_params_and_result.json"
    attacker.attack_dataset(args, args.max_queries, result_dump_path)
