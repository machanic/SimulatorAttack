from collections import OrderedDict
from torch import nn
import torch
import copy

class InnerLoop(nn.Module):
    '''
    This module performs the inner loop of MAML
    The forward method updates weights with gradient steps on training data, 
    then computes and returns a meta-gradient w.r.t. validation data
    '''
    def __init__(self, network, num_updates, step_size, meta_batch_size, use_softmax, loss_fn=None):
        super(InnerLoop, self).__init__()
        self.network = copy.deepcopy(network)
        # Number of updates to be taken
        self.num_updates = num_updates
        # Step size for the updates
        self.step_size = step_size
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=1)
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.MSELoss()
        self.meta_batch_size = meta_batch_size

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        self.network.copy_weights(net)

    def net_forward(self, x, weights=None):
        return self.network.net_forward(x, weights)

    def forward_pass(self, images, target, weights=None):
        ''' Run data through net, return loss and output '''
        images = images.cuda()
        target = target.cuda()
        if self.use_softmax:
            images = self.softmax(images)
            target = self.softmax(target)
        # Run the batch through the net, compute loss
        out = self.net_forward(images, weights)
        loss = self.loss_fn(out, target)
        return loss, out


    def forward(self, task_q1_images,  task_q2_images, task_q1_logits, task_q2_logits):
        in_support, in_query, target_support, target_query = task_q1_images.detach(), task_q2_images.detach(),\
                                                             task_q1_logits.detach(), task_q2_logits.detach()
        fast_weights = OrderedDict((name, param) for (name, param) in self.network.named_parameters())
        for i in range(self.num_updates):
            if i==0:
                loss, _ = self.forward_pass(in_support, target_support)
                grads = torch.autograd.grad(loss, self.parameters() )
            else:
                loss, _ = self.forward_pass(in_support, target_support, fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
        # Compute the meta gradient and return it
        loss, _ = self.forward_pass(in_query, target_query, fast_weights)
        loss = loss / self.meta_batch_size   # normalize loss
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
        return meta_grads, fast_weights

