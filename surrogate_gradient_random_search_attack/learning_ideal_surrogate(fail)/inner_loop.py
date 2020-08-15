from collections import OrderedDict
from torch import nn
import torch
import copy
from torch.nn import functional as F


class InnerLoop(nn.Module):
    '''
    This module performs the inner loop
    The forward method updates weights with gradient steps on training data, 
    then computes and returns a meta-gradient w.r.t. validation data
    '''
    def __init__(self, network, num_updates, step_size, meta_batch_size, loss_type):
        super(InnerLoop, self).__init__()
        self.network = copy.deepcopy(network)
        self.num_updates = num_updates
        self.step_size = step_size
        self.meta_batch_size = meta_batch_size
        if loss_type=="cw":
            self.loss_fn = self.cw_loss
        elif loss_type == "xent":
            self.loss_fn = self.xent_loss
        else:
            raise Exception("No such loss function of {}".format(loss_type))
        self.mse_loss = nn.MSELoss(reduction='mean')

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        self.network.copy_weights(net)

    def net_forward(self, x, weights=None):
        return self.network.net_forward(x, weights)


    def xent_loss(self, logit, label, target=None):
        if target is not None:
            return -F.cross_entropy(logit, target, reduction='none')
        else:
            return F.cross_entropy(logit, label, reduction='none')

    def cw_loss(self, logit, label, target=None):
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), target]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    def forward_pass(self, images, gt_gradients, gt_logits, true_labels, target_labels, weights=None):
        ''' Run data through net, return loss and output '''
        images = images.cuda()
        images.requires_grad_()
        gt_gradients = gt_gradients.cuda()
        if target_labels is not None:
            target_labels = target_labels.cuda()
        true_labels = true_labels.cuda()
        output = self.net_forward(images, weights)
        loss_knowledge_distillation = self.mse_loss(output, gt_logits)
        loss_temp = self.loss_fn(output, true_labels, target_labels)
        pred_gradient = torch.autograd.grad(loss_temp.mean(), images, create_graph=True)[0]
        loss_grad_regression = self.mse_loss(pred_gradient, gt_gradients)
        return loss_grad_regression + loss_knowledge_distillation

    def forward(self, task_train_images, task_test_images,
                task_train_gt_gradients, task_test_gt_gradients,
                task_train_logits, task_test_logits,
                task_train_true_labels, task_train_target_labels,
                task_test_true_labels, task_test_target_labels):
        fast_weights = OrderedDict((name, param) for (name, param) in self.network.named_parameters())

        for i in range(self.num_updates):
            if i==0:
                train_loss = self.forward_pass(task_train_images, task_train_gt_gradients, task_train_logits,
                                       task_train_true_labels, task_train_target_labels)
                grads = torch.autograd.grad(train_loss, self.parameters(), create_graph=True)
            else:
                train_loss = self.forward_pass(task_test_images, task_test_gt_gradients, task_test_logits,
                                      task_test_true_labels, task_test_target_labels, fast_weights)
                grads = torch.autograd.grad(train_loss, fast_weights.values())
            fast_weights = OrderedDict((name, param - self.step_size * grad)
                                     for ((name, param), grad) in zip(fast_weights.items(), grads))
        # Compute the meta gradient and return it
        test_loss = self.forward_pass(task_test_images, task_test_gt_gradients, task_test_logits,
                                      task_test_true_labels, task_test_target_labels, fast_weights)
        if self.num_updates == 1:
            loss = (train_loss + test_loss) / self.meta_batch_size   # normalize loss
        else:
            loss = test_loss / self.meta_batch_size
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
        return meta_grads

