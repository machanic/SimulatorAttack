from abc import ABCMeta
from abc import abstractmethod
from torch import nn

from square_attack_RL.rl.distribution import SoftmaxDistribution, CategoricalDistribution


class Policy(object, metaclass=ABCMeta):
    """Abstract policy."""

    @abstractmethod
    def forward(self, state):
        """Evaluate a policy.
        Returns:
            Distribution of actions
        """
        raise NotImplementedError()

class SoftmaxPolicy(nn.Module, Policy):
    """Softmax policy that uses Boltzmann distributions.

    Args:
        model (chainer.Link):
            Link that is callable and outputs action values.
        beta (float):
            Parameter of Boltzmann distributions.
    """

    def __init__(self, model, beta=1.0, min_prob=0.0):
        super(SoftmaxPolicy, self).__init__()
        self.beta = beta
        self.min_prob = min_prob
        self.model = model

    def forward(self, x):
        h = self.model(x)
        return SoftmaxDistribution(
            h, beta=self.beta, min_prob=self.min_prob)