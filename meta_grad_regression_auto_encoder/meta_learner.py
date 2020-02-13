from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import IN_CHANNELS
from meta_grad_regression_auto_encoder.network.autoencoder import AutoEncoder
from dataset.meta_img_grad_dataset import MetaImgOnlineGradTaskDataset, MetaImgOfflineGradTaskDataset
from optimizer.radam import RAdam


class MetaLearner(object):
    def __init__(self, dataset, meta_batch_size, inner_batch_size,
                 inner_step_size, epoch, num_inner_updates, protocol, tot_num_tasks):
        self.dataset = dataset
        self.meta_batch_size = meta_batch_size
        self.inner_step_size = inner_step_size
        self.epoch = epoch
        self.tot_num_tasks = tot_num_tasks
        self.num_inner_updates = num_inner_updates
        self.network = AutoEncoder(IN_CHANNELS[self.dataset]).cuda()
        trn_dataset = MetaImgOfflineGradTaskDataset(tot_num_tasks, dataset, inner_batch_size, protocol)
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.mse_loss = nn.MSELoss(reduction="mean").cuda()
        self.optimizer = Adam(self.network.parameters(), lr=inner_step_size, betas=(0, 0.999))

    def inner_train_step(self, model, task_images, task_grads):
        """
        Inner training step procedure.
        """
        x, y = task_images, task_grads
        ypred = model(x)
        loss = self.mse_loss(ypred, y)
        # if log: print loss.data[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def meta_train_step(self, images, grads, current_step_size):
        """
        Meta training step procedure.
        """
        weights_original = deepcopy(self.network.state_dict())
        new_weights = []
        for task_idx, (task_images, task_grads) in enumerate(zip(images, grads)): # each task_images is shape of (B,C,H,W)
            task_images, task_grads= task_images.cuda(), task_grads.cuda()
            for inner_iter in range(self.num_inner_updates):
                self.inner_train_step(self.network, task_images, task_grads)
            new_weights.append(deepcopy(self.network.state_dict()))
            self.network.load_state_dict({name: weights_original[name] for name in weights_original})

        # update to fsweight
        ws = len(new_weights)
        fweights = {name: sum(d[name] for d in new_weights) / float(ws) for name in new_weights[0]}  # mean value
        # the outer update step of reptile
        self.network.load_state_dict({name: weights_original[name] + ((fweights[name] - weights_original[name]) * current_step_size)
                               for name in weights_original})

    def adjust_learning_rate(self, itr, meta_lr, lr_decay_itr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if lr_decay_itr > 0:
            if int(itr % lr_decay_itr) == 0 and itr > 0:
                meta_lr = meta_lr / (10 ** int(itr / lr_decay_itr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = meta_lr

    def train(self, meta_step_size,  resume_epoch, model_path):
        """
        Meta training.
        """
        for epoch in range(resume_epoch, self.epoch):
            for i, (images, grads) in enumerate(self.train_loader):  # images shape = (T,B,C,H,W)、
                itr = epoch * len(self.train_loader) + i
                # self.adjust_learning_rate(itr, meta_step_size, lr_decay_itr)
                frac_done = float(itr) / (self.epoch * len(self.train_loader))
                current_step_size = meta_step_size * (1. - frac_done)
                # images, grads = images.cuda(),grads.cuda()
                self.meta_train_step(images, grads, current_step_size)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, model_path)

    #
    # def evaluate(self, dataset, model,
    #              criterion, optimizer, num_shots,
    #              num_classes, inner_batch_size, inner_iters):
    #     """
    #     Evaluation. Trains on eval training set and evaluates on a small number of test images.
    #     """
    #
    #     weights_original = deepcopy(model.state_dict())
    #     train_set, test_set = dset.split_train_test(
    #         dset.sample_mini_dataset(dataset, num_classes, num_shots + 1))
    #
    #     for batch in dset.mini_batches(train_set, inner_batch_size, inner_iters, False):
    #         self.inner_train_step(model, criterion, optimizer, batch)
    #
    #     inputs, labels = zip(*test_set)
    #     preds, loss = self.predict(model, inputs, labels, criterion)
    #     preds = preds.cpu().numpy()
    #     num_correct = sum([float(pred == sample[1]) for pred, sample in zip(preds, test_set)])
    #     model.load_state_dict({name: weights_original[name] for name in weights_original})
    #     return num_correct, loss
    #
    #
    # def test(self, dataset,
    #          model, criterion, optimizer,
    #          num_shots, num_classes, eval_inner_batch_size, eval_inner_iters, num_samples):
    #     """
    #     Runs evaluation multiple times and returns the number of correct predictions.
    #     """
    #
    #     total = 0.
    #     for _ in range(num_samples):
    #         ncorrect, loss = self.evaluate(dataset, model, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size,
    #                                   eval_inner_iters)
    #         total += ncorrect
    #     return total / (num_samples * num_classes)
    #
    #
    # def evaluate_embedding(self, dataset,
    #                        model, criterion, optimizer,
    #                        num_shots, num_classes, inner_batch_size, inner_iters):
    #     """
    #     Evaluation. Trains on eval training set and evaluates on a small number of test images.
    #     """
    #
    #     weights_original = deepcopy(model.state_dict())
    #
    #     train_sets = []
    #     test_sets = []
    #
    #     for _ in range(4):
    #         train_set, test_set = dset.split_train_test(
    #             dset.sample_mini_dataset(dataset, num_classes, num_shots + 1))
    #         train_sets.append(train_set)
    #         test_sets.append(test_set)
    #
    #     for train_set in train_sets:
    #         for batch in dset.mini_batches(train_set, inner_batch_size, inner_iters / 4, False):
    #             self.inner_train_step(model, criterion, optimizer, batch)
    #
    #     num_correct = 0.
    #     for train_set, test_set in zip(train_sets, test_sets):
    #         inputs_test, labels_test = zip(*test_set)
    #         inputs_train, labels_train = zip(*train_set)
    #
    #         xt = self.to_tensor(np.array(inputs_train))
    #         embt = model.embedding(xt).data.cpu().numpy()
    #         embtg = embt.reshape(-1, embt.shape[-2] / num_classes, embt.shape[-1])
    #         avgs = np.mean(embtg, axis=1)
    #
    #         xte = self.to_tensor(np.array(inputs_test))
    #         embte = model.embedding(xte).data.cpu().numpy()
    #
    #         pred = []
    #         for emb in embte:
    #             d = np.linalg.norm(np.repeat(np.array([emb]), avgs.shape[0], axis=0) - avgs, axis=1)
    #             pred.append(np.argmin(d))
    #         preds = np.array(pred)
    #
    #         num_correct += sum([float(pred == sample[1]) for pred, sample in zip(preds, test_set)])
    #
    #     model.load_state_dict({name: weights_original[name] for name in weights_original})
    #     return num_correct


