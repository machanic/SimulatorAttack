import torch
from torch import nn
from torch.nn import functional as F

class Environment(object):
    def __init__(self):
        self.previous_loss = 0

    def loss_fn(self, model, x, label):
        logit = model(x)
        return F.cross_entropy(logit, label, reduction='none')


    # reward map is obtained by modifying bounding box separately, the increment loss value is treated as reward
    def get_reward(self, model, image, label):
        current_loss = self.loss_fn(model, image, label)  # loss 越大越好, shape = (batch_size,)
        reward = current_loss - self.previous_loss
        self.previous_loss = current_loss.detach().clone()
        return reward

