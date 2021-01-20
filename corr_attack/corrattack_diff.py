import argparse
import os
import sys
sys.path.append(os.getcwd())

import json

from config import MODELS_TEST_STANDARD, CLASS_NUM, IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.defensive_model import DefensiveModel
from dataset.standard_model import StandardModel
from utils import *
import itertools
import math
import torch.nn as nn
import torch
import heapq
import yaml
from corr_attack.gaussian_process import attack_bayesian_EI
import random
from sklearn.decomposition import PCA
import numpy as np
from torch.nn import functional as F
from corr_attack.utils import perturb_image, Function, change_noise
import glog as log
import os.path as osp


class CorrAttack_Diff(object):

    def __init__(self, function, config, device):
        self.config = config
        self.batch_size = config['batch_size']
        self.function = function
        self.model = function.model
        self.device = device
        self.epsilon = self.config['epsilon']
        self.gp = attack_bayesian_EI.Attack(
            f=self,
            dim=4,
            max_evals=1000,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=30,
            device=device,
            dtype="float32",
        )
        self.query_limit = self.config['query_limit']
        self.max_iters = self.config['max_iters']
        self.init_iter = self.config["init_iter"]
        self.init_batch = self.config["init_batch"]
        self.memory_size = self.config["memory_size"]
        self.channels = self.config["channels"]
        self.image_height = self.config["image_height"]
        self.image_width = self.config["image_width"]
        self.gp_emptyX = torch.zeros((1,4), device=device)
        self.gp_emptyfX = torch.zeros((1), device=device)
        self.local_forget_threshold = self.config['local_forget_threshold']
        self.lr = self.config['lr']

        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.maximum_queries = self.config["max_queries"]

    def split_block(self, image, upper_left, lower_right, block_size):
        blocks = []
        xs = np.arange(upper_left[0], lower_right[0], block_size)
        ys = np.arange(upper_left[1], lower_right[1], block_size)
        features = []
        for x, y in itertools.product(xs, ys):
            for c in range(self.channels):
                features.append(image[c, x:x + block_size, y:y + block_size].cpu().numpy().reshape(-1))
        pca = PCA(n_components=1)
        features = pca.fit_transform(features)
        i = 0
        features[:, 0] = (features[:, 0] - features[:, 0].min()) / (features[:, 0].max() - features[:, 0].min() + 0.1)
        for x, y in itertools.product(xs, ys):
            for c in range(self.channels):
                blocks.append((x // block_size, y // block_size, c, features[i, 0]))
                i += 1
        return blocks


    def local_bayes(self, blocks):
        select_blocks = []
        for i, block in enumerate(blocks):
            x, y, c = block[0:3]
            x *= self.block_size
            y *= self.block_size
            select_blocks.append(block)

        blocks = torch.tensor(select_blocks, dtype=torch.float32, device=self.device)
        init_batch_size = max(blocks.size(0)//self.init_batch, 5)
        init_iteration = self.init_iter
        init_iteration = init_batch_size*(init_iteration-1)
        self.gp.init(blocks/self.gp_normalize, n_init=init_batch_size, batch_size=1, iteration=init_iteration)
        self.gp.X_pool = blocks/self.gp_normalize

        memory_size = int(len(self.gp.X) * self.memory_size)
        priority_X = torch.arange(0, len(self.gp.X)).to(self.gp.X.device)
        priority = torch.tensor(len(self.gp.X)).to(priority_X.device)
        init_size = len(self.gp.X)

        local_forget_threshold = self.local_forget_threshold[self.block_size]
        for i in range(blocks.size(0)):
            training_steps = 1
            x_cand, y_cand, self.gp.hypers = self.gp.create_candidates(self.gp.X, self.gp.fX, self.gp.X_pool, n_training_steps=training_steps, hypers=self.gp.hypers, sample_number=1)
            block, self.gp.X_pool = self.gp.select_candidates(x_cand, y_cand, get_loss=False)
            block = block[0] * self.gp_normalize
            if i >= blocks.size(0)//2 and y_cand.min()>-1e-4:
                return False

            noise_p = change_noise(self.noise, block, self.block_size, self.sigma, self.epsilon)
            query_image_p = perturb_image(self.image, noise_p)
            logit, loss_p = self.function(query_image_p, self.label)

            noise_n = change_noise(self.noise, block, self.block_size, -self.sigma, self.epsilon)
            query_image_n = perturb_image(self.image, noise_n)
            logit, loss_n = self.function(query_image_n, self.label)

            if loss_p < 0:
                self.loss = loss_p
                self.noise = noise_p
                return True
            elif loss_n < 0:
                self.loss = loss_n
                self.noise = noise_n
                return True

            if self.function.current_counts > self.query_limit:
                return False

            if self.config['print_log']:
                log.info("queries {}, new loss {:4f}, old loss {:4f}, gaussian size {}".format(self.function.current_counts, torch.min(loss_p, loss_n).item(), self.loss.item(), len(self.gp.X)))

            if loss_p < self.loss or loss_n < self.loss:
                if loss_p < loss_n:
                    self.noise = noise_p
                    self.loss = loss_p
                else:
                    self.noise = noise_n
                    self.loss = loss_n

                diff = (self.gp.X*self.gp_normalize - block)[:,0:2].abs().max(dim=1)[0]
                index = diff > (local_forget_threshold + 0.5)
                self.gp.X = self.gp.X[index]
                self.gp.fX = self.gp.fX[index]
                priority_X = priority_X[index]

                if priority_X.size(0) >= memory_size:
                    index = torch.argmin(priority_X)
                    priority_X = torch.cat((priority_X[:index], priority_X[index+1:]))
                    self.gp.X = torch.cat((self.gp.X[:index], self.gp.X[index+1:]), dim=0)
                    self.gp.fX = torch.cat((self.gp.fX[:index], self.gp.fX[index + 1:]), dim=0)

                if len(self.gp.X_pool) == 0:
                    break

                if self.gp.X.size(0) <= 1:
                    new_index = random.randint(0, len(self.gp.X_pool)-1)
                    new_block = self.gp.X_pool[new_index] * self.gp_normalize

                    query_image = perturb_image(self.image, change_noise(self.noise, new_block, self.block_size, self.sigma, self.epsilon))
                    _, query_loss_p = self.function(query_image, self.label)

                    query_image = perturb_image(self.image, change_noise(self.noise, new_block, self.block_size, -self.sigma, self.epsilon))
                    _, query_loss_n = self.function(query_image, self.label)
                    self.gp.X = torch.cat((self.gp.X, (new_block/self.gp_normalize).unsqueeze(0)), dim=0)
                    self.gp.fX = torch.cat((self.gp.fX, torch.min(query_loss_p, query_loss_n) - self.loss), dim=0)

                    priority_X = torch.cat((priority_X, priority.unsqueeze(0)), dim=0)
                    priority += 1
            else:
                diff = (self.gp.X - block/self.gp_normalize).abs().sum(dim=1)
                min_diff, history_index = torch.min(diff, dim=0)
                if min_diff < 1e-5:
                    update_index = history_index
                elif priority_X.size(0) < memory_size:
                    update_index = priority_X.size(0)
                    self.gp.X = torch.cat((self.gp.X, self.gp_emptyX), dim=0)
                    self.gp.fX = torch.cat((self.gp.fX, self.gp_emptyfX), dim=0)
                    priority_X = torch.cat((priority_X, priority.unsqueeze(0)), dim=0)
                else:
                    update_index = torch.argmin(priority_X)

                self.gp.X[update_index] = block / self.gp_normalize
                self.gp.fX[update_index] = torch.min(loss_p, loss_n) - self.loss
                priority_X[update_index] = priority
                priority += 1
            if self.function.current_counts > self.maximum_queries:
                return False

        return False

    def get_loss(self, indices):
        indices = indices * self.gp_normalize
        batch_size = self.batch_size
        num_batches = int(math.ceil(len(indices)/batch_size))
        losses = torch.zeros(len(indices), device=self.device)
        for ibatch in range(num_batches):
            bstart = ibatch * batch_size
            bend = min(bstart + batch_size, len(indices))
            images = self.image.unsqueeze(0).repeat(bend - bstart, 1, 1, 1)

            for i, index in enumerate(indices[bstart:bend]):
                noise_flip = change_noise(self.noise, index, self.block_size, self.sigma, self.epsilon)
                images[i] = perturb_image(self.image, noise_flip)
            logit, loss_p = self.function(images, self.label)
            for i, index in enumerate(indices[bstart:bend]):
                noise_flip = change_noise(self.noise, index, self.block_size, -self.sigma, self.epsilon)
                images[i] = perturb_image(self.image, noise_flip)
            logit, loss_n = self.function(images, self.label)
            losses[bstart:bend] = torch.min(loss_n, loss_p)

        return losses - self.loss

    def attack(self, image, label):
        self.function.new_counter()
        self.noise = torch.zeros((self.channels, self.image_height, self.image_width), dtype=torch.float32, device=self.device)

        self.image = image.clone()
        self.label = label
        self.block_size = self.config['block_size']["{}x{}".format(self.image_height,self.image_width)]
        _, self.loss = self.function(perturb_image(image, self.noise), label)

        upper_left = [0, 0]
        lower_right = [self.image_height, self.image_width]
        blocks = self.split_block(self.image, upper_left, lower_right, self.block_size)

        while True:
            self.gp_normalize = torch.tensor([self.image_height/self.block_size, self.image_width/self.block_size, self.channels, 1],
                                             dtype=torch.float32, device=self.device)
            for iter in range(self.max_iters):
                self.sigma = self.lr
                success = self.local_bayes(blocks)
                if success or self.function.current_counts > self.query_limit:
                    image = perturb_image(self.image, self.noise)
                    return image, success

                if self.config['print_log']:
                    log.info("Block size: {}, loss: {:.4f}, num queries: {}".format(self.block_size, self.loss.item(), self.function.current_counts))

            if self.block_size >= 2:
                if self.block_size % 2 != 0:
                    temp_block_size = self.block_size // 2
                    if temp_block_size < 10:
                        for t in range(1,10):
                            if self.image_height % t == 0:
                                self.block_size = t
                    else:
                        while self.image_height % temp_block_size != 0:
                            temp_block_size += 1
                        self.block_size = temp_block_size
                else:
                    self.block_size //= 2
                blocks = self.split_block(self.image, upper_left, lower_right, self.block_size)
            if self.function.current_counts > self.maximum_queries:
                image = perturb_image(self.image, self.noise)
                return image, False
    def attack_all_images(self, args, arch_name, result_dump_path):

        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            images = images.to(self.device)
            true_labels = true_labels.to(self.device)
            with torch.no_grad():
                logit = self.model(images)
            pred = logit.argmax(dim=1)
            query = torch.zeros(args.batch_size).to(self.device)
            correct = pred.eq(true_labels).float()  # shape = (batch_size,)

            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().to(self.device)
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                            size=target_labels[
                                                                                invalid_target_index].shape).long().to(
                            self.device)
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
                label = target_labels[0].item()
            else:
                label = true_labels[0].item()
            adv_images, success = self.attack(images[0], label)
            log.info("{}-th image, query: {} success:{}".format(batch_index+1, self.function.current_counts, success))
            query = torch.tensor([self.function.current_counts]).float()
            success = torch.tensor([int(success)]).float()
            not_done = torch.ones_like(success) - success
            success_query = success * query
            selected = torch.arange(batch_index * args.batch_size,
                                    min((batch_index + 1) * args.batch_size, self.total_images))  # 选择这个batch的所有图片的index
            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query']:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完

        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def get_exp_dir_name(dataset, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'CorrAttack_diff_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        dirname = 'CorrAttack_diff-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def print_args(state):
    keys = sorted(state.keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, state[key]))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configures/corr_attack/corrattack_diff.yaml', help='config file')
    parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--targeted',  action='store_true')
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--epsilon', type=float)
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--batch-size', type=int,default=1, help='batch size for bandits attack.')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--norm', type=str, default="linf", help='Which lp constraint to run bandits [linf|l2]')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config) as config_file:
        state = yaml.load(config_file, Loader=yaml.FullLoader)

    if args.save_prefix is not None:
        state['save_prefix'] = args.save_prefix
    if args.model_name is not None:
        state['model_name'] = args.model_name
    if args.epsilon is not None:
        state['epsilon'] = args.epsilon
    state['target'] = args.targeted
    if 'defense' not in state:
        state['defense'] = False
    state["max_queries"] = args.max_queries
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    if args.targeted and args.dataset == "ImageNet":
        args.max_queries = 50000
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset,  args.norm, args.targeted, args.target_type,
                                             args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.all_archs:
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
    if args.attack_defense:
        assert args.defense_model is not None
    torch.backends.cudnn.deterministic = True

    if args.all_archs:
        archs = MODELS_TEST_STANDARD[args.dataset]
    else:
        assert args.arch is not None
        archs = [args.arch]
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(state)
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
        model.to(device)
        model.eval()
        state["channels"] = IN_CHANNELS[args.dataset]
        state["image_height"] = model.input_size[-2]
        state["image_width"] = model.input_size[-1]

        function = Function(model, state['batch_size'], state['margin'], CLASS_NUM[args.dataset], state['target'])
        attacker = CorrAttack_Diff(function, state, device)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()

