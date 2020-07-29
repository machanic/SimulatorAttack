import cv2
import numpy as np
import random
import torch
from torch.nn import functional as F

class State(object):
    def __init__(self, size, norm_lp, epsilon):
        self.image = torch.zeros(size, dtype=torch.float32).cuda()
        self.p = norm_lp
        self.epsilon = epsilon
        self.proj_maker = self.l2_proj if norm_lp == 'l2' else self.linf_proj  # 调用proj_maker返回的是一个函数

    def reset(self, x):
        self.image = x.cuda()
        self.proj_step = self.proj_maker(x.cuda(), self.epsilon)

    def l2_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            delta = new_x - orig
            out_of_bounds_mask = (self.normalize(delta) > eps).float()
            x = (orig + eps * delta / self.normalize(delta)) * out_of_bounds_mask
            x += new_x * (1 - out_of_bounds_mask)
            return x
        return proj

    def linf_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            return orig + torch.clamp(new_x - orig, -eps, eps)
        return proj

    # 每个位置产生与不产生action
    def step(self, action_map):
        action_map = action_map.float().unsqueeze(1) # N,1,H,W
        # action_map resize to input image
        action_map = F.interpolate(action_map, size=(self.image.size(-2),self.image.size(-1)),mode='nearest',align_corners=True)
        action_map = np.squeeze(action_map.detach().cpu().numpy().astype(np.int32))

        for batch_idx in range(action_map.shape[0]):
            connect_arr = cv2.connectedComponents(action_map[batch_idx], connectivity=8, ltype=cv2.CV_32S)  # mask shape = 1 x H x W
            component_num = connect_arr[0]
            label_matrix = connect_arr[1]
            effective_count = 0  # 有效的box number
            actions = []
            for component_label in range(1, component_num):
                action = np.zeros(shape=(3, action_map.shape[-2], action_map[-1]))  # 3,H,W
                current = (label_matrix == component_label) & (action_map[batch_idx] != 0)
                current_int = current.astype(np.int32)
                if np.any(current_int):
                    effective_count += 1
                    i_arr, j_arr = np.nonzero(current_int)
                    action[:, i_arr, j_arr] = np.random.choice([-1, 1], size=[3, 1, 1])
                actions.append(action)
            actions = np.stack(actions)
            image = self.lp_step(self.image, action_map, self.epsilon, self.p)  # TODO 目前只支持Linf attack
        # N,B,C,H,W
        batch_action_map = np.zeros(shape=(action_map.shape[0], effective_count, 3, action_map.shape[-2], action_map.shape[-1]))
        for batch_idx in range(action_map.shape[0]):
            for component_label in range(1, component_num):
                current = (label_matrix == component_label) & (action_map != 0)
                current = current.astype(np.int32)
                individual_action_map[:,channel,:,:][label_matrix==component_label] = np.random.choice([-1, 1],
                                                                                                      size=[3, 1, 1])

            action_map[(label_matrix==component_label) & (action_map!=0)] = random.choice([-1, 1])

        # returns the image with the shape of (N,B,C,H,W), where B is the total number of positive bounding boxes

        action_map = torch.from_numpy(action_map).float().cuda().unsqueeze(1) # N,1,H,W
        image = self.lp_step(self.image, action_map, self.epsilon, self.p)  #  TODO 目前只支持Linf attack
        self.image = self.proj_step(image)
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
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

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
