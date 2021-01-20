import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
import pandas as pd
from meta_attack.meta_training.learner import Learner
from copy import deepcopy

class Meta(nn.Module):
    def __init__(self, args, config):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = Learner(config)

    def loss_function(self,logits,target):
        loss = F.mse_loss(logits,target)
        return loss

    def forward(self, batch_data):
        """
            :param support_x:   [b, setsz, c_, h, w]
            :param support_y:   [b, setsz]
            :param x_qry:   [b, querysz, c_, h, w]
            :param y_qry:   [b, querysz]
            :batch_data : [support_x, support_y, support_label, query_x, query_y, query_label]
            :return:
        """
        task_num = len(batch_data)  # each is [support_x, support_y, support_label, query_x, query_y, query_label]
        _, _, c_, h, w = batch_data[0][0].size()  # shape of support_x
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 2)]
        querysz = 1

        def compute_diff(weights):
            '''
            this method is to computer the difference between received weights
            and the self.net.parameters(), it will return the difference
            '''
            dis = []
            for i, each in enumerate(self.net.parameters()):
                dis.append((each - weights[i]).clone().detach())
            return dis

        def computer_meta_weight(weights):
            '''
            meta optim for reptile
            this method will update the self.net.parameters according to the
            reveived weights, which is the updating directions. The updata learning
            rate is self.update.lr
            '''
            dic = self.net.state_dict()
            keys = list(dic.keys())
            for i,each in enumerate(weights[0]):
                diff = torch.zeros_like(each)
                for j in range(task_num):
                    diff += weights[j][i]  # j is task index
                diff /= task_num
                dic[keys[i]] -=  diff
            self.net.load_state_dict(dic)

        task_diff_weights = []  # task diff weights is the list to store all weights diiffs in all tasks
        for i in range(task_num):
            # each entry of batch_data [x_spt, y_spt, x_qry, y_qry]
            x_spt, y_spt, label_spt, x_qry, y_qry, label_qry = [x.reshape([-1,c_,h,w]).cuda()
                                                                if i % 3 <=1 else x.reshape([-1]).cuda()
                                                                for i,x in enumerate(batch_data[i])]
            fast_weights = []
            for each in self.net.parameters():
                pp = each.clone().detach()
                pp.requires_grad_()
                fast_weights.append(pp)
            '''
            the optimizer for each sub-task
            '''
            cur_task_optim = optim.Adam(fast_weights, lr=self.update_lr)

            logits = self.net(x_spt, fast_weights, bn_training=True)
            loss = self.loss_function(logits, y_spt)
            cur_task_optim.zero_grad()
            loss.backward()
            cur_task_optim.step()
            # this is the loss and accuracy before first update
            with torch.no_grad():
                logits_q = self.net(x_spt, self.net.parameters(), bn_training=True)
                loss_q = self.loss_function(logits_q, y_spt)
                losses_q[0] += loss_q
                correct = loss_q.sum()
                corrects[0] = corrects[0] + correct
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                loss_q = self.loss_function(logits_q, y_qry)
                losses_q[1] += loss_q
                correct = loss_q.sum()
                corrects[1] = corrects[1] + correct
            for k in range(1, self.update_step):
                logits = self.net(x_spt, fast_weights, bn_training=True)
                loss = self.loss_function(logits, y_spt)
                cur_task_optim.zero_grad()
                loss.backward()
                cur_task_optim.step()

                with torch.no_grad():
                    logits_q = self.net(x_spt, fast_weights, bn_training=True)
                    loss_q = self.loss_function(logits_q, y_spt)

                    correct = loss_q.sum()
                    corrects[k + 1] = corrects[k + 1] + correct
            current_task_diff = compute_diff(fast_weights)
            task_diff_weights.append(current_task_diff)

        computer_meta_weight(task_diff_weights)
        logits_q = self.net(x_spt, bn_training=True)
        loss_q = self.loss_function(logits_q, y_spt)
        correct = loss_q.sum()
        corrects[k + 2] = correct
        for i in range(len(corrects)):
            corrects[i] = corrects[i].item()
        accs = np.array(corrects) / (querysz * task_num)
        return accs

    def finetunning(self, test_data, model, step_size=0.3):

        # method attack to measure the attack success rate
        def attack(x,y,model,step_size):
            orilabel = model(x).argmax(dim=1)
            x_fool = x + torch.sign(y)*step_size
            x_fool = torch.clamp(x_fool,0,1)
            fool_label = model(x_fool).argmax(dim=1)
            acc = (orilabel==fool_label).detach().cpu().numpy().sum()/len(orilabel)
            return acc
        _, __, c_, h, w = test_data[0].size()
        x_spt, y_spt, label_spt, x_qry, y_qry, label_qry = [x.reshape([-1,c_,h,w]).cuda()
                                                            if i % 3 <=1 else x.reshape([-1]).cuda().long()
                                                            for i, x in enumerate(test_data)]
        assert len(x_spt.shape) == 4
        random_point_number = 256
        weight_mask = np.zeros((784))
        picked_points = np.random.choice(728, random_point_number)
        weight_mask[picked_points] = 1
        weight_mask = weight_mask.reshape((28, 28))
        weight_mask = torch.tensor(weight_mask).float().cuda()
        corrects = [[0, 0] for _ in range(self.update_step_test + 1)]
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        net.eval()
        cur_task_optim = optim.Adam(net.parameters(), lr=self.update_lr)
        logits_q = net(x_qry, bn_training=True)
        correct = [attack(x_qry, logits_q, model, step_size),
                   attack(x_qry, y_qry, model, step_size)]
        corrects[0] = correct
        cur_task_optim.zero_grad()
        logits = net(x_spt)
        loss = self.loss_function(logits, y_spt)
        loss.backward()
        cur_task_optim.step()
        logits_q = net(x_qry, bn_training=True)
        correct = [attack(x_qry, logits_q, model, step_size),
                   attack(x_qry, y_qry, model, step_size)]
        corrects[1] = correct
        for k in range(1, self.update_step_test):
            logits = net(x_spt, bn_training=True)
            loss = self.loss_function(logits, y_spt)
            cur_task_optim.zero_grad()
            loss.backward()
            cur_task_optim.step()
            logits_q = net(x_qry,bn_training=True)
            correct = [attack(x_qry,logits_q,model,step_size),attack(x_qry,y_qry,model,step_size)]
            corrects[k + 1] = correct
        del net
        accs = np.array(corrects)
        return accs