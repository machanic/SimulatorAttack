import torch
import numpy as np
import os
import json
from dataset.standard_model import StandardModel
from torch.nn import functional as F

class Tracker:
    def __init__(self, foldername):
        self.counter = 0
        self.results = []
        self.curt_best = float("inf")
        self.foldername = foldername
        try:
            os.makedirs(foldername, exist_ok=True)
        except OSError:
            print("Creation of the directory %s failed" % foldername)
        else:
            print("Successfully created the directory %s " % foldername)

    def dump_trace(self):
        trace_path = self.foldername + '/result' + str(len(self.results))
        final_results_str = json.dumps(self.results)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def track(self, result):
        if result < self.curt_best:
            self.curt_best = result
        self.results.append(self.curt_best)
        if len(self.results) % 100 == 0:
            self.dump_trace()


class Attack:
    def __init__(self, dataset, arch, dims=32*32*3):
        self.model = StandardModel(dataset, arch, no_grad=True)
        self.model.cuda()
        self.model.eval()
        self.dims    = dims                   #problem dimensions
        self.lb      =  np.zeros(dims)         #lower bound for each dimensions
        self.ub      =  np.ones(dims)         #upper bound for each dimensions
        self.tracker = Tracker('Attack')      #defined in functions.py

        # tunable hyper-parameters in LA-MCTS
        self.Cp = 10
        self.leaf_size = 8
        self.kernel_type = "poly"
        self.ninits = 40
        self.gamma_type = "auto"
        print("initialize levy at dims:", self.dims)

    def xent_loss(self, logit, label, target=None):
        if target is not None:
            return -F.cross_entropy(logit, target)
        else:
            return F.cross_entropy(logit, label)

    def __call__(self, x, true_labels, target_labels):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        logits = self.model(torch.from_numpy(x.reshape(3,32,32)).cuda())
        loss = self.xent_loss(logits, true_labels, target_labels)
        self.tracker.track(loss.item())
        return loss.item()
