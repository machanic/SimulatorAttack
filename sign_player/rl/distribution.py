from abc import ABCMeta
from abc import abstractmethod
from cached_property import cached_property

import torch
import numpy as np
from torch.nn import functional as F

def sample_discrete_actions(batch_probs):
    """Sample a batch of actions from a batch of action probabilities.

    Args:
        batch_probs (ndarray): batch of action probabilities BxA
    Returns:
        ndarray consisting of sampled action indices
    """
    batch_probs = batch_probs.detach().cpu().numpy()
    return np.argmax(
        np.log(batch_probs) + np.random.gumbel(size=batch_probs.shape),
        axis=1).astype(np.int32, copy=False)

class Distribution(object, metaclass=ABCMeta):
    """Batch of distributions of data."""

    @property
    @abstractmethod
    def entropy(self):
        """Entropy of distributions.

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self):
        """Sample from distributions.

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def prob(self, x):
        """Compute p(x).

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, x):
        """Compute log p(x).

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @abstractmethod
    def copy(self, x):
        """Copy a distribion unchained from the computation graph.

        Returns:
            Distribution
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def most_probable(self):
        """Most probable data points.

        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def kl(self, distrib):
        """Compute KL divergence D_KL(P|Q).

        Args:
            distrib (Distribution): Distribution Q.
        Returns:
            chainer.Variable
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def params(self):
        """Learnable parameters of this distribution.

        Returns:
            tuple of chainer.Variable
        """
        raise NotImplementedError()

    def sample_with_log_prob(self):
        """Do `sample` and `log_prob` at the same time.

        This can be more efficient than calling `sample` and `log_prob`
        separately.

        Returns:
            chainer.Variable: Samples.
            chainer.Variable: Log probability of the samples.
        """
        y = self.sample()
        return y, self.log_prob(y)

class CategoricalDistribution(Distribution):
    """Distribution of categorical data."""

    @cached_property
    def entropy(self):
        with torch.enable_grad():
            return - (self.all_prob * self.all_log_prob).sum(dim=1)

    @cached_property
    def most_probable(self):
        action = torch.argmax(self.all_prob, dim=1).long()
        action = action - 1
        return action

    def sample(self):
        action = torch.from_numpy(sample_discrete_actions(self.all_prob))
        action = action - 1
        return action

    def prob(self, x):
        return torch.stack([self.all_prob[i, x[i]] for i in range(self.all_prob.size(0))])

    def log_prob(self, x):
        return torch.stack([self.all_log_prob[i, x[i]] for i in range(self.all_log_prob.size(0))])

    @abstractmethod
    def all_prob(self):
        raise NotImplementedError()

    @abstractmethod
    def all_log_prob(self):
        raise NotImplementedError()

    def kl(self, distrib):
        return (self.all_prob * (self.all_log_prob - distrib.all_log_prob)).sum(dim=1)


class SoftmaxDistribution(CategoricalDistribution):
    """Softmax distribution.

    Args:
        logits (ndarray): Logits for softmax
            distribution.
        beta (float): inverse of the temperature parameter of softmax
            distribution
        min_prob (float): minimum probability across all labels
    """

    def __init__(self, logits, beta=1.0, min_prob=0.0):
        self.logits = logits
        self.beta = beta
        self.min_prob = min_prob
        self.n = logits.shape[1]
        assert self.min_prob * self.n <= 1.0

    @property
    def params(self):
        return (self.logits,)

    @cached_property
    def all_prob(self):
        with torch.enable_grad():
            if self.min_prob > 0:
                return (F.softmax(self.beta * self.logits, dim=1)
                        * (1 - self.min_prob * self.n)) + self.min_prob
            else:
                return F.softmax(self.beta * self.logits, dim=1)

    @cached_property
    def all_log_prob(self):
        with torch.enable_grad():
            if self.min_prob > 0:
                return torch.log(self.all_prob)
            else:
                return F.log_softmax(self.beta * self.logits,dim=1)

    def copy(self, x):
        return SoftmaxDistribution(self.logits.copy(),
                                   beta=self.beta, min_prob=self.min_prob)

    @cached_property
    def myentropy(self):
        with torch.enable_grad():
            return -torch.sum(self.all_prob * self.all_log_prob, dim=1, keepdim=False)  # B,C,H,W

    def mylog_prob(self, action):
        action = action.clone()
        action = action + 1 # convert (-1,0,1) to (0,1,2) range
        action = action.cuda().long() # B,C,H,W
        n_batch, n_actions, in_channels, h, w = self.all_log_prob.size()
        selected_p = torch.gather(self.all_log_prob,1,action.unsqueeze(1))  # out[i][j][k] = input[i][index[i][j][k]][k]
        return selected_p.view(n_batch, in_channels, h, w)

        # p_trans = self.all_log_prob.permute(0, 2, 3, 4, 1).contiguous()  # B,C,H,W,Action
        # p_trans = p_trans.view(-1, n_actions) # B*C*H*W, Action
        # action_reshape = action.view(1, -1)[0]  # B,C,H,W -> 1,B*C*H*W
        # selected_p = torch.stack([p_trans[i, action_reshape[i].item()] for i in range(p_trans.size(0))])
        # return selected_p.view(n_batch, in_channels, h, w)

    def __repr__(self):
        return 'SoftmaxDistribution(beta={}, min_prob={}) logits:{} probs:{} entropy:{}'.format(  # NOQA
            self.beta, self.min_prob, self.logits.detach().cpu().numpy(),
            self.all_prob.detach().cpu().numpy(), self.entropy.detach().cpu().numpy())

    def __getitem__(self, i):
        return SoftmaxDistribution(self.logits[i],
                                   beta=self.beta, min_prob=self.min_prob)

