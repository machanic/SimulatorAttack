# -*- coding: utf-8 -*-

import torch
from torch import nn

import adversarial_defense.post_averaging.PA_defense as padef
from config import CLASS_NUM
from dataset.standard_model import StandardModel


class PostAveragedNetwork(nn.Module):
    def __init__(self, model, K, R, num_classes, device='cuda'):
        super(PostAveragedNetwork, self).__init__()
        self._model = model.to(device)
        self.num_classes = num_classes
        self._K = K
        self._r = [R / 3, 2 * R / 3, R]
        self._sample_method = 'random'
        self._vote_method = 'avg_softmax'
        self._device = device

    @property
    def model(self):
        return self._model

    def classify(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # gather neighbor samples
        x_squad = padef.formSquad_resnet(self._sample_method, self._model, x, self._K, self._r, device=self._device)
        # forward with a batch of neighbors
        logits, _ = padef.integratedForward(self._model, x_squad, batchSize=100, nClasses=self.num_classes, device=self._device,
                                            voteMethod=self._vote_method)
        return torch.as_tensor(logits)

    def forward(self, x):
        logits_list = []
        for img in x:
            logits = self.classify(img)
            logits_list.append(logits)
        return torch.cat(logits_list, dim=0).cuda()

    def to(self, device):
        self._model = self._model.to(device)
        self._device = device

    def eval(self):
        self._model = self._model.eval()


def post_averged_model(dataset, arch):
    model = StandardModel(dataset, arch, no_grad=False)
    R = 30 if arch in ["resnet-101","resnet-110","resnet-152"] else 6
    return PostAveragedNetwork(model, K=15, R=R,num_classes=CLASS_NUM[dataset])
