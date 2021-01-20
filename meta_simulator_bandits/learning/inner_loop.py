from collections import OrderedDict
from torch import nn
import torch
import copy


class InnerLoop(nn.Module):
    '''
    This module performs the inner loop
    The forward method updates weights with gradient steps on training data, 
    then computes and returns a meta-gradient w.r.t. validation data
    '''
    def __init__(self, network, num_updates, step_size, meta_batch_size, loss_type, use_softmax):
        super(InnerLoop, self).__init__()
        self.network = copy.deepcopy(network)
        # Number of updates to be taken
        self.num_updates = num_updates
        # Step size for the updates
        self.step_size = step_size
        self.meta_batch_size = meta_batch_size
        self.pair_wise_distance = nn.PairwiseDistance(p=2)
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.softmax = nn.Softmax(dim=1)
        self.use_softmax = use_softmax
        self.loss_type = loss_type    # pair_mse, mse

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        self.network.copy_weights(net)

    def net_forward(self, x, weights=None):
        return self.network.net_forward(x, weights)

    def forward_pass(self, imgs_1, imgs_2, target_1, target_2, weights=None):
        ''' Run data through net, return loss and output '''
        imgs_1 = imgs_1.cuda()
        imgs_2 = imgs_2.cuda()
        out_1 = self.net_forward(imgs_1, weights)
        out_2 = self.net_forward(imgs_2, weights)
        if self.use_softmax:
            out_1 = self.softmax(out_1)
            out_2 = self.softmax(out_2)
            target_1 = self.softmax(target_1)
            target_2 = self.softmax(target_2)
        diff_loss1 = self.mse_loss(out_1, target_1)
        diff_loss2 = self.mse_loss(out_2, target_2)
        if self.loss_type == "pair_mse":
            predict_distance = self.pair_wise_distance(out_1, out_2)  # shape = （batch_size,)
            normal_factor = torch.mean(predict_distance)
            predict_distance = predict_distance / normal_factor
            target_distance = self.pair_wise_distance(target_1, target_2)  # shape = （batch_size,)
            target_distance = target_distance / torch.mean(target_distance)
            distance_loss = self.mse_loss(predict_distance, target_distance)
            loss = distance_loss + 0.1 * diff_loss1 + 0.1 * diff_loss2
        else:
            loss = diff_loss1 + diff_loss2
        return loss

    def evaluate_accuracy(self, query_images, query_targets, weights):
        # query_images shape = (B,C,H,W) query_targets = (B, #class)
        query_images = query_images.cuda()
        query_targets = query_targets.cuda()
        query_target_labels = torch.max(query_targets, dim=1)[1]
        query_output = self.net_forward(query_images, weights)
        query_predict = query_output.max(1)[1]
        accuracy = query_target_labels.eq(query_predict.long()).sum() / float(query_targets.size(0))
        if self.use_softmax:
            query_output =self.softmax(query_output)
            query_targets = self.softmax(query_targets)
        mse_error = self.mse_loss(query_output, query_targets)
        accuracy = accuracy.detach().cpu()
        mse_error = mse_error.detach().cpu()
        return accuracy, mse_error
    
    def forward(self, task_support_images_1, task_support_images_2, task_query_images_1, task_query_images_2,
                      task_support_target_1, task_support_target_2, task_query_target_1, task_query_target_2):
        fast_weights = OrderedDict((name, param) for (name, param) in self.network.named_parameters())
        for i in range(self.num_updates):
            if i==0:
                loss = self.forward_pass(task_support_images_1, task_support_images_2, task_support_target_1, task_support_target_2)
                grads = torch.autograd.grad(loss, self.parameters())
            else:
                loss = self.forward_pass(task_support_images_1, task_support_images_2, task_support_target_1, task_support_target_2,
                                         fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = OrderedDict((name, param - self.step_size * grad)
                                     for ((name, param), grad) in zip(fast_weights.items(), grads))
        # Compute the meta gradient and return it
        loss = self.forward_pass(task_query_images_1, task_query_images_2, task_query_target_1, task_query_target_2, fast_weights)
        loss = loss / self.meta_batch_size   # normalize loss
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
        # query_images = torch.cat((task_query_images_1, task_query_images_2), dim=0)  # B, C,H,W
        # query_targets = torch.cat((task_query_target_1, task_query_target_2), dim=0)  # B, #class
        # with torch.no_grad():
        #     accuracy, mse_error = self.evaluate_accuracy(query_images, query_targets, fast_weights)
        return meta_grads #, accuracy, mse_error

