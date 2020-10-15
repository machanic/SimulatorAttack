"""PyTorch Carlini and Wagner L2 attack algorithm.

Based on paper by Carlini & Wagner, https://arxiv.org/abs/1608.04644 and a reference implementation at
https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
"""
import copy
import numpy as np
import torch.nn.functional as F
import torch
from meta_attack.attacks.helpers import *

from torch import nn
class GradientGenerator(object):

    def __init__(self, update_pixels, targeted=False, classes=10, debug=False):
        self.debug = debug
        self.targeted = targeted  # false
        self.num_classes = classes
        self.confidence = 0  # FIXME need to find a good value for this, 0 value used in paper not doing much...

        self.use_log = True
        self.batch_size = update_pixels
        self.use_importance = True
        self.constant = 0.5
        self.model_1 = None


    def _loss(self, output, target, dist, constant):
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            if self.use_log:
                loss1 = torch.clamp(torch.log(other + 1e-30) - torch.log(real + 1e-30), min=0.)
            else:
                loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            if self.use_log:
                loss1 = torch.clamp(torch.log(real + 1e-30) - torch.log(other + 1e-30), min=0.)
            else:
                loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = constant * loss1
        loss = loss1
        loss2 = dist
        return loss, loss1, loss2

    def run(self, model, img, target, indice):
        ori_img = img.clone()
        grad = torch.zeros(self.batch_size, dtype=torch.float32)
        modifier = torch.zeros_like(img, dtype=torch.float32)
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if torch.cuda.is_available():
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = target_onehot.detach()
        target_var.requires_grad = False
        img_var = img.repeat(self.batch_size * 2 + 1, 1, 1, 1)

        for i in range(self.batch_size):
            img_var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
            img_var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0000
        with torch.no_grad():
            output = F.softmax(model(img_var), dim=1).detach()
            dist = l2_dist(img_var, ori_img, keepdim=True).squeeze(2).squeeze(2)
            loss, loss1, loss2 = self._loss(output.data, target_var, dist, self.constant)
            for i in range(self.batch_size):
                grad[i] = (loss[i * 2 + 1] - loss[i * 2 + 2]) / 0.0002
            modifier.reshape(-1)[indice] = grad.cuda()
        return modifier.detach().cpu().numpy()[0], indice


    # save GPU memory, but slower
    # 放弃治疗，失败。GPU 显存仍然溢出
    # def run_efficient_memory(self, model, img, target, indice):
    #     # ori_img = img
    #     grad = torch.zeros(self.batch_size, dtype=torch.float32)
    #     modifier = torch.zeros_like(img, dtype=torch.float32)
    #     if self.model_1 is None:
    #         self.model_1 = copy.deepcopy(model).to(1)
    #     model_1 = self.model_1
    #     models= [model, model_1]
    #     target_onehot = torch.zeros(target.size() + (self.num_classes,))
    #     if torch.cuda.is_available():
    #         target_onehot = target_onehot.cuda()
    #     target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    #     target_var = target_onehot.detach()
    #     target_var.requires_grad = False
    #
    #     img_repeat = img.clone().repeat(self.batch_size, 1, 1, 1)
    #
    #     for i in range(self.batch_size):
    #         img_repeat[i].reshape(-1)[indice[i]] += 0.0001  # 1,3,5,...  -> 1,2,3,4,
    #     with torch.no_grad():
    #         output_original = F.softmax(model(img), dim=1)
    #         # dist_original = l2_dist(img, ori_img, keepdim=True).squeeze(2).squeeze(2)
    #         loss_orig, loss1_orig, loss2_orig = self._loss(output_original, target_var, None, self.constant)
    #         split_size = self.batch_size // 5
    #         img_repeat_chunks = torch.split(img_repeat, split_size, dim=0)
    #         # dist = l2_dist(img_repeat, ori_img, keepdim=True).squeeze(2).squeeze(2)
    #         index = 0
    #         for i, img_repeat_each in enumerate(img_repeat_chunks):
    #             model_ = models[i%len(models)]
    #             if i % 2 != 0:
    #                 device_id = 1
    #             else:
    #                 device_id = 0
    #             img_repeat_each = img_repeat_each.to(device_id)
    #             output_each = F.softmax(model_(img_repeat_each), dim=1).detach().to(0)
    #             loss_each, loss1, loss2 = self._loss(output_each, target_var, None, self.constant)
    #             loss_each = loss_each.detach()
    #             for i in range(loss_each.size(0)):
    #                 grad[index + i] = (loss_each[i] -  loss_orig[0]) / 0.0002
    #             index += loss_each.size(0)
    #             torch.cuda.empty_cache()
    #         modifier.reshape(-1)[indice] = grad.cuda()
    #     del img_repeat
    #
    #
    #     # pdb.set_trace()
    #     return modifier.cpu().numpy()[0], indice

