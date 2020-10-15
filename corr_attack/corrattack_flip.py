import argparse
import torchvision.models as models
import os
import json
import DataLoader
from utils import *
from Normalize import Normalize, Permute
import itertools
import math
import torch.nn as nn
import torch
import yaml
from corr_attack.gaussian_process import attack_bayesian_EI
import random
from sklearn.decomposition import PCA
import numpy as np

def split_block(image, upper_left, lower_right, block_size):
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    features = []
    for x, y in itertools.product(xs, ys):
        for c in range(3):
            features.append(image[c, x:x+block_size, y:y+block_size].cpu().numpy().reshape(-1))
    pca = PCA(n_components=1)
    features = pca.fit_transform(features)
    i = 0
    features[:, 0] = (features[:, 0] - features[:, 0].min())/(features[:, 0].max()-features[:, 0].min()+0.1)
    for x, y in itertools.product(xs, ys):
        for c in range(3):
            blocks.append((x//block_size, y//block_size, c, features[i, 0]))
            i += 1
    return blocks


class CorrAttack_Flip:

    def __init__(self, function, config, device):
        self.config = config
        self.batch_size = config['batch_size']
        self.function = function
        self.device = device
        self.epsilon = self.config['epsilon']
        self.gp = attack_bayesian_EI.Attack(
            f=self,
            dim=4,
            max_evals=1000,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=10,
            device=device,
            dtype="float32",
        )
        self.query_limit = self.config['query_limit']
        self.max_iters = self.config['max_iters']
        self.init_iter = self.config["init_iter"]
        self.init_batch = self.config["init_batch"]
        self.memory_size = self.config["memory_size"]
        self.gp_emptyX = torch.zeros((1,4), device=device)
        self.gp_emptyfX = torch.zeros((1), device=device)
        self.local_forget_threshold = self.config['local_forget_threshold']

    def noise_init(self):
        h = 224//self.block_size
        w = 224//self.block_size
        noise = torch.sign(torch.randn((1, 3, h, w), dtype=torch.float32, device=device))*self.epsilon
        noise = torch.nn.functional.interpolate(noise, (224, 224), mode='nearest').squeeze(0)
        return noise

    def attack(self, image, label):
        self.function.new_counter()
        self.block_size = self.config['block_size']
        self.noise = self.noise_init()
        self.image = image.clone()
        self.label = label
        _, self.loss = self.function(perturb_image(image, self.noise), label)
        if self.loss < 0:
            image = perturb_image(self.image, self.noise)
            return image, True
        upper_left = [0, 0]
        lower_right = [224, 224]
        blocks = split_block(self.image, upper_left, lower_right, self.block_size)

        while True:
            # Run local search algorithm on the mini-batch
            self.gp_normalize = torch.tensor([224/self.block_size, 224/self.block_size, 3, 1], dtype=torch.float32, device=self.device)
            for iter in range(self.max_iters):
                success = self.local_bayes(blocks, "positive")
                if success or self.function.current_counts > self.query_limit:
                    image = perturb_image(self.image, self.noise)
                    return image, success

                success = self.local_bayes(blocks, "negative")
                if success or self.function.current_counts > self.query_limit:
                    image = perturb_image(self.image, self.noise)
                    return image, success

                if self.config['print_log']:
                    print("Block size: {}, loss: {:.4f}, num queries: {}".format(self.block_size, self.loss.item(), self.function.current_counts))

            if self.block_size >= 2:
                self.block_size //= 2
                blocks = split_block(self.image, upper_left, lower_right, self.block_size)

    def local_bayes(self, blocks, direction):
        select_blocks = []
        for i, block in enumerate(blocks):
            x, y, c = block[0:3]
            x *= self.block_size
            y *= self.block_size
            if direction=="positive" and self.noise[c, x, y] < 0 or direction=="negative" and self.noise[c, x, y] > 0:
                select_blocks.append(block)

        blocks = torch.tensor(select_blocks, dtype=torch.float32, device=self.device)
        init_batch_size = max(len(blocks)//self.init_batch, 5)
        init_iteration = self.init_iter
        if len(blocks) < 2:
            return False
        if init_batch_size * init_iteration > len(blocks):
            if len(blocks)//init_iteration < 2:
                init_iteration = len(blocks)//2
                init_batch_size = 2
            else:
                init_batch_size = len(blocks)//init_iteration
        init_iteration = init_batch_size*(init_iteration-1)
        self.gp.init(blocks/self.gp_normalize, n_init=init_batch_size, batch_size=1, iteration=init_iteration)

        self.gp.X_pool = blocks/self.gp_normalize

        memory_size = int(len(self.gp.X) * self.memory_size)
        priority_X = torch.arange(0, len(self.gp.X)).to(self.gp.X.device)
        priority = torch.tensor(len(self.gp.X)).to(priority_X.device)

        local_forget_threshold = self.local_forget_threshold[self.block_size]
        for i in range(len(blocks)):
            training_steps = 1
            x_cand, y_cand, self.gp.hypers = self.gp.create_candidates(self.gp.X, self.gp.fX, self.gp.X_pool, n_training_steps=training_steps, hypers=self.gp.hypers, sample_number=1)
            block, self.gp.X_pool = self.gp.select_candidates(x_cand, y_cand, get_loss=False)
            block = block[0] * self.gp_normalize
            if i>=len(blocks)//2 and y_cand.min()>-1e-4:
                return False

            noise = flip_noise(self.noise, block, self.block_size)
            query_image = perturb_image(self.image, noise)
            logit, loss = self.function(query_image, self.label)

            if loss < 0:
                self.loss = loss
                return True

            if self.function.current_counts > self.query_limit:
                return False

            if self.config['print_log']:
                print("queries {}, new loss {:4f}, old loss {:4f}, gaussian size {}".format(self.function.current_counts, loss.item(), self.loss.item(), len(self.gp.X)))

            if loss < self.loss:
                self.noise = noise.clone()
                self.loss = loss

                diff = (self.gp.X*self.gp_normalize - block)[:,0:2].abs().max(dim=1)[0]
                index = diff > (local_forget_threshold + 0.5)
                self.gp.X = self.gp.X[index]
                self.gp.fX = self.gp.fX[index]
                priority_X = priority_X[index]

                if len(priority_X) >= memory_size:
                    index = torch.argmin(priority_X)
                    priority_X = torch.cat((priority_X[:index], priority_X[index+1:]))
                    self.gp.X = torch.cat((self.gp.X[:index], self.gp.X[index+1:]), dim=0)
                    self.gp.fX = torch.cat((self.gp.fX[:index], self.gp.fX[index + 1:]), dim=0)

                if len(self.gp.X_pool) == 0:
                    break

                if len(self.gp.X) <= 1:
                    new_index = random.randint(0, len(self.gp.X_pool)-1)
                    new_block = self.gp.X_pool[new_index] * self.gp_normalize

                    query_image = perturb_image(self.image, flip_noise(self.noise, new_block, self.block_size))
                    _, query_loss = self.function(query_image, self.label)

                    self.gp.X = torch.cat((self.gp.X, (new_block/self.gp_normalize).unsqueeze(0)), dim=0)
                    self.gp.fX = torch.cat((self.gp.fX, query_loss - self.loss), dim=0)

                    priority_X = torch.cat((priority_X, priority.unsqueeze(0)), dim=0)
                    priority += 1
            else:
                diff = (self.gp.X - block/self.gp_normalize).abs().sum(dim=1)
                min_diff, history_index = torch.min(diff, dim=0)
                if min_diff < 1e-5:
                    update_index = history_index
                elif len(priority_X) < memory_size:
                    update_index = len(priority_X)
                    self.gp.X = torch.cat((self.gp.X, self.gp_emptyX), dim=0)
                    self.gp.fX = torch.cat((self.gp.fX, self.gp_emptyfX), dim=0)
                    priority_X = torch.cat((priority_X, priority.unsqueeze(0)), dim=0)
                else:
                    update_index = torch.argmin(priority_X)

                self.gp.X[update_index] = block / self.gp_normalize
                self.gp.fX[update_index] = loss - self.loss
                priority_X[update_index] = priority
                priority += 1
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
                noise_flip = flip_noise(self.noise, index, self.block_size)
                images[i] = perturb_image(self.image, noise_flip)
            logit, loss = self.function(images, self.label)
            losses[bstart:bend] = loss

        return losses - self.loss


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.json', help='config file')
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
parser.add_argument('--model_name', default=None)
parser.add_argument('--target', default=False, action='store_true')
parser.add_argument('--epsilon', default=None, type=float)

args = parser.parse_args()

with open(args.config) as config_file:
    state = yaml.load(config_file, Loader=yaml.FullLoader)

if args.save_prefix is not None:
    state['save_prefix'] = args.save_prefix
if args.model_name is not None:
    state['model_name'] = args.model_name
if args.epsilon is not None:
    state['epsilon'] = args.epsilon
state['target'] = args.target
if 'defense' not in state:
    state['defense'] = False

new_state = state.copy()
new_state['batch_size'] = 1
new_state['test_bs'] = 1
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

_, dataloader, nlabels, mean, std = DataLoader.imagenet(new_state)

model = get_model(state["model_name"], mean, std, state['defense'])

model.to(device)
model.eval()

F = Function(model, state['batch_size'], state['margin'], nlabels, state['target'])
Attack = CorrAttack_Flip(F, state, device)
count_success = 0
count_total = 0
if not os.path.exists(state['save_path']):
    os.mkdir(state['save_path'])

if state['target']:
    target_classes = np.load('target_class.npy')

for i, (images, labels) in enumerate(dataloader):
    images = images.to(device)
    labels = int(labels)
    logits = model(images)
    correct = torch.argmax(logits, dim=1) == labels
    if correct:
        torch.cuda.empty_cache()
        if state['target']:
            labels = target_classes[i]

        with torch.no_grad():
            adv, success = Attack.attack(images[0], labels)

        count_success += int(success)
        count_total += int(correct)
        print("image: {} eval_count: {} success: {} average_count: {} success_rate: {}".format(i, F.current_counts, success, F.get_average(iter=state['query_limit']),  float(count_success) / float(count_total)))

F.new_counter()
success_rate = float(count_success) / float(count_total)

if args.target:
    save_prefix = "{}_{}".format(state['save_prefix'], "Target")
else:
    save_prefix = "{}_{}".format(state['save_prefix'], "Un-target")
np.save(os.path.join(state['save_path'], '{}_{}_Epsilon_{}.npy'.format(save_prefix, state['model_name'], state['epsilon'])), np.array(F.counts))
print("success rate {}".format(success_rate))
print("average eval count {}".format(F.get_average(iter=state['query_limit'])))
