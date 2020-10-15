import torch
import torch.nn as nn
from MGA_attack.utils import *
import numpy as np

class MI_FGSM_ENS(object):

    def __init__(self, models, weights=None, epsilon=0.1, stepsize=0.01, iters=10, mu=1,
                 random_start=True, loss_fn=nn.CrossEntropyLoss(), pnorm=np.inf,
                 clip_min=0, clip_max=1,targeted=False, position='loss'):
        '''
        :param models:
        :param weights:
        :param position: ensemble position, logits, probabilities, loss
        '''
        self.models = []
        for model in models:
            self.models.append(model.cuda())

        if weights is None:
            num_ensemble = len(self.models)
            self.weights = [1./num_ensemble]*num_ensemble
        else:
            self.weights = weights

        self.epsilon = epsilon
        self.stepsize = stepsize
        self.iters = iters
        self.mu = mu
        self.random_start = random_start

        self.loss_fn = loss_fn
        self.pnorm = pnorm
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

        self.position = position

    def perturb(self, x, y):
        x = x.clone()
        x, y = x.cuda(), y.cuda()
        # Call MI_FGSM Attack
        return self.__mi_fgsm_attack(x, y)

    def ensemble_logits(self, x, y):
        # ensemble in logits with same weight
        logits = 0
        for model, w in zip(self.models, self.weights):
            logits += model(x) * w

        loss = self.loss_fn(logits, y)  # Calculate the loss
        # target or untarget
        if self.targeted:
            loss = -loss
        return loss

    def ensemble_loss(self, x, y):
        # ensemble in logits with same weight
        ensemble_loss = 0
        for model, w in zip(self.models, self.weights):
            logits = model(x)

            loss = self.loss_fn(logits, y)  # Calculate the loss
            # target or untarget
            if self.targeted:
                loss = -loss

            ensemble_loss += loss *w

        return ensemble_loss

    def __mi_fgsm_attack(self, x, y):

        if self.random_start:
            noise = np.random.uniform(-self.epsilon, self.epsilon, x.size()).astype('float32')
            noise = torch.from_numpy(noise).cuda()
        else:
            noise = 0

        # perturbation
        delta = torch.zeros_like(x) + noise
        delta = torch.clamp(x + delta, self.clip_min, self.clip_max) - x
        delta.requires_grad = True

        g = 0
        for i in range(self.iters):
            x_nes = x + delta
            x_nes = torch.clamp(x_nes, 0, 1)
            for model in self.models:
                model.zero_grad()

            if self.position == 'logits':
                loss = self.ensemble_logits(x_nes, y)
            if self.position == 'loss':
                loss = self.ensemble_loss(x_nes, y)
            loss.backward()  # 所有的logits加到一起反向传播
            grad_data = delta.grad

            # this is the wrong code, but it works better on mnist  todo verify it on imagenet
            # g = self.mu * g + grad_data / torch.batch_norm(grad_data, p=1)
            # this is the stadrad MI-FGSM
            g = self.mu * g + normalize(grad_data, p=1)

            if self.pnorm == np.inf:

                delta.data += self.stepsize * g.sign()
                # clamp accm perturbation to [-epsilon, epsilon]
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            # pnorm = 2
            else:
                delta.data += self.stepsize * normalize(g, p=2)
                delta.data = clamp_by_2norm(delta.data, self.epsilon)

            delta.data = torch.clamp(
                x + delta.data, self.clip_min, self.clip_max) - x

            delta.grad.data.zero_()
            # clear cache,
            torch.cuda.empty_cache()

        return torch.clamp(x + delta.data, self.clip_min, self.clip_max)

