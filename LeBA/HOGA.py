
import torch
from torch.nn import functional as F
from torch import nn


class TrainSurrogateModel():
    # HOGA: Class Method to train surrogate model
    def __init__(self, FL_rate):
        self.train_num = 0
        self.lamda_dict = {}
        self.d_loss_record = {}
        self.s_loss_record = {}
        self.d_loss_sum = 0
        self.s_loss_sum = 0
        self.FL_rate = FL_rate

    def get_lamda(self, filenames):
        # Adaptive gamma in paper, Here is to get adaptive lamda
        for filename in filenames:
            if filename in self.d_loss_record and filename in self.s_loss_record:
                if self.train_num > 50:
                    lamda2 = self.s_loss_sum / self.d_loss_sum  # Use history s_loss sum and d_loss sum, compute lamda2
                    self.lamda_dict[filename] = self.lamda_dict[
                                                    filename] * 0.9 + lamda2 * 0.1  # Update lamda with lamda2 using momentum
                else:
                    self.lamda_dict[filename] = 3.0
            else:
                self.lamda_dict[filename] = 3.0

    def __call__(self, filenames, imgs, model2, labels, diff, query_score, query_loss, last_loss, optimizer):
        # Call HOGA, train model2
        '''Args:
            diff: Current query perturbation
            query_score: Current query score with (imgs+diff)
            query_loss: Current query loss with (imgs+diff)
            last_loss: History query loss with (imgs)
            model2: surrogate model
            optimizer: optimizer for model2
        '''
        self.get_lamda(filenames)
        lamda = torch.tensor([self.lamda_dict[filename] for filename in filenames]).cuda()
        self.train_num += 1
        d_loss = query_loss - last_loss  # Get Query delta loss
        adv_imgs = imgs.detach().clone()
        adv_imgs.requires_grad = True
        out = model2(adv_imgs)
        # print(out.shape,labels.shape)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        prob = F.softmax(out, dim=1)
        s_score = prob.gather(1, labels.view(-1, 1))
        loss = nn.CrossEntropyLoss(reduction='none')(out, labels)  # Note that using cross entropy loss to train surrogate model here
        grad = torch.autograd.grad(loss.sum(), adv_imgs, create_graph=True)  # Create High Order Compute Graph
        grad = grad[0]
        s_loss = (diff.detach() * grad).view([imgs.shape[0], -1]).sum(
            dim=1)  # diff*s_grad: surrogate model loss with diff.
        forward_loss = nn.MSELoss()(s_score,
                                    query_score.detach())  # Forward Loss: approximate forward-pass score number
        backward_loss = nn.MSELoss()(s_loss / lamda, d_loss.detach())
        # Backward Loss: Minimize difference between surrogate model loss and query loss. equal to high-order gradient approximation.
        loss2 = backward_loss + forward_loss * self.FL_rate

        model2.zero_grad()
        loss2.backward()
        optimizer.step()
        model2.zero_grad()
        optimizer.zero_grad()
        del adv_imgs

        for i in range(len(filenames)):
            filename = filenames[i]
            if filename not in self.d_loss_record:
                self.d_loss_record[filename] = []
            if filename not in self.s_loss_record:
                self.s_loss_record[filename] = []
            self.d_loss_record[filename].append(d_loss[i].detach().cpu())
            self.s_loss_record[filename].append(s_loss[i].detach().cpu())
            self.d_loss_sum += d_loss[i].detach().cpu().abs()
            self.s_loss_sum += s_loss[i].detach().cpu().abs()


