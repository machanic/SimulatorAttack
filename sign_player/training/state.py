import numpy as np
import torch

class State(object):
    def __init__(self, size, norm_lp, epsilon):
        self.image = torch.zeros(size, dtype=torch.float32).cuda()
        self.p = norm_lp
        self.epsilon = epsilon

    def reset(self, x):
        self.image = x.cuda()

    def step(self, action_map):
        action_map = action_map.float().cuda()
        self.image = self.lp_step(self.image, action_map, self.epsilon, self.p)
        return self.image

    def lp_step(self, x, g, lr, p):
        """
        performs lp step of x in the direction of g, where the norm is computed
        across all the dimensions except the first one (assuming it's the batch_size)
        :param x: batch_size x dim x .. tensor (or numpy)
        :param g: batch_size x dim x .. tensor (or numpy)
        :param lr: learning rate (step size)
        :param p: 'inf' or '2'
        :return:
        """
        if p == 'linf':
            return self.linf_step(x, g, lr)
        elif p == 'l2':
            return self.l2_step(x, g, lr)
        else:
            raise Exception('Invalid p value')

    def normalize(self, t):
        """
        Return the norm of a tensor (or numpy) along all the dimensions except the first one
        :param t:
        :return:
        """
        _shape = t.shape
        batch_size = _shape[0]
        num_dims = len(_shape[1:])
        if torch.is_tensor(t):
            norm_t = torch.sqrt(t.pow(2).sum(dim=[_ for _ in range(1, len(_shape))])).view([batch_size] + [1] * num_dims)
            norm_t += (norm_t == 0).float() * np.finfo(np.float64).eps
            return norm_t
        else:
            _norm = np.linalg.norm(
                t.reshape([batch_size, -1]), axis=1,
            ).reshape([batch_size] + [1] * num_dims)
            return _norm + (_norm == 0) * np.finfo(np.float64).eps

    def l2_step(self, x, g, lr):
        """
        performs l2 step of x in the direction of g, where the norm is computed
        across all the dimensions except the first one (assuming it's the batch_size)
        :param x: batch_size x dim x .. tensor (or numpy)
        :param g: batch_size x dim x .. tensor (or numpy)
        :param lr: learning rate (step size)
        :return:
        """
        # print(x.device)
        # print(g.device)
        # print(norm(g).device)
        return x + lr * g / self.normalize(g)

    def linf_step(self, x, g, lr):
        """
        performs linfinity step of x in the direction of g
        :param x: batch_size x dim x .. tensor (or numpy)
        :param g: batch_size x dim x .. tensor (or numpy)
        :param lr: learning rate (step size)
        :return:
        """
        if torch.is_tensor(x):
            return x + lr * torch.sign(g)
        else:
            return x + lr * np.sign(g)
