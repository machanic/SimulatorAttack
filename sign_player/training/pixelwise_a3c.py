from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library

from sign_player.rl import agent
from sign_player.rl.a3c_model import A3CModel
from sign_player.rl.misc import async_
from sign_player.rl.misc.copy_param import copy_grad, copy_param
from sign_player.rl.recurrent import state_kept, Recurrent
from sign_player.rl.misc.batch_states import batch_states

standard_library.install_aliases()  # NOQA
import copy
import glog as log
import torch

class PixelWiseA3C(agent.AttributeSavingMixin, agent.AsyncAgent):
    """A3C: Asynchronous Advantage Actor-Critic.
        Args:
            model (A3CModel): Model to train
            optimizer (torch.optim.optimizer.Optimizer): optimizer used to train the model
            t_max (int): The model is updated after every t_max local steps
            gamma (float): Discount factor [0,1]
            beta (float): Weight coefficient for the entropy regularizaiton term.
            process_idx (int): Index of the process.
            phi (callable): Feature extractor function
            pi_loss_coef (float): Weight coefficient for the loss of the policy
            v_loss_coef (float): Weight coefficient for the loss of the value
                function
            act_deterministically (bool): If set true, choose most probable actions
                in act method.
            batch_states (callable): method which makes a batch of observations.
                default is `chainerrl.misc.batch_states.batch_states`
        """
    process_idx = None
    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer, t_max, gamma, beta=1e-2,
                 phi=lambda x: x, pi_loss_coef=1.0, v_loss_coef=0.5,
                 keep_loss_scale_same=False, normalize_grad_by_t_max=False,
                 use_average_reward=False, average_reward_tau=1e-2,
                 act_deterministically=False, average_entropy_decay=0.999,
                 average_value_decay=0.999, batch_states=batch_states):
        assert isinstance(model, A3CModel)
        # Globally shared model
        self.shared_model = model
        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)
        async_.assert_params_not_shared(self.shared_model, self.model)
        self.optimizer = optimizer  # 优化的是self.shared_model

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same
        self.normalize_grad_by_t_max = normalize_grad_by_t_max
        self.use_average_reward = use_average_reward
        self.average_reward_tau = average_reward_tau
        self.act_deterministically = act_deterministically
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.batch_states = batch_states

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.average_reward = 0
        # A3C won't use a explorer, but this arrtibute is referenced by run_dqn
        self.explorer = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0
        self.shared_model.cuda()

    def sync_parameters(self):
        self.model.load_state_dict(self.shared_model.state_dict())

    @property
    def shared_attributes(self):
        return ('shared_model', 'optimizer')

    def update(self, statevar):
        assert self.t_start < self.t
        if statevar is None:
            R = 0
        else:
            with state_kept(self.model):
                _, vout = self.model.pi_and_v(statevar)
            R = vout.float().view(-1)
        pi_loss = 0
        v_loss = 0
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            if self.use_average_reward:
                R -= self.average_reward
            v = self.past_values[i]
            advantage = R - v  # shape = (B,)
            if self.use_average_reward:
                self.average_reward += self.average_reward_tau * advantage
            # Accumulate gradients of policy
            log_prob = self.past_action_log_prob[i]  # shape = B,C,H,W
            entropy = self.past_action_entropy[i]  #  shape = B,C,H,W

            # Log probability is increased proportionally to advantage
            pi_loss -= log_prob * advantage.view(-1,1,1,1).float()
            # Entropy is maximized
            pi_loss -= self.beta * entropy.view_as(pi_loss)
            # Accumulate gradients of value function
            v_loss += torch.mul(v - R, v - R) / 2
        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef
        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef
        # Normalize the loss of sequences truncated by terminal states
        if self.keep_loss_scale_same and \
                self.t - self.t_start < self.t_max:
            factor = self.t_max / (self.t - self.t_start)
            pi_loss *= factor
            v_loss *= factor
        if self.normalize_grad_by_t_max:
            pi_loss /= self.t - self.t_start
            v_loss /= self.t - self.t_start
        if self.process_idx == 0:
            log.info("pi_loss:{} v_loss:{}".format(pi_loss.mean().item(), v_loss.mean().item()))
        total_loss = torch.mean(pi_loss + v_loss.view(-1,1,1,1))
        # Compute gradients using thread-specific model
        self.model.zero_grad()
        total_loss.backward()
        self.shared_model.zero_grad()
        copy_grad(target_model=self.shared_model, source_model=self.model)
        # Update the globally shared model
        self.optimizer.step()
        self.sync_parameters()
        if isinstance(self.model, Recurrent):
            self.model.unchain_backward()
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.t_start = self.t

    def act_and_train(self, state, reward):
        statevar = state.cuda()
        self.past_rewards[self.t - 1] = reward.cuda()
        if self.t - self.t_start == self.t_max:
            self.update(statevar)
        self.past_states[self.t] = statevar
        pout, vout = self.model.pi_and_v(statevar)
        action = pout.sample()  # Do not backprop through sampled actions # shape =(batch_size,C,H,W)
        self.past_action_log_prob[self.t] = pout.mylog_prob(action)  # 根据action 选择那个action的log 概率
        self.past_action_entropy[self.t] = pout.myentropy  # B,C,H,W
        self.past_values[self.t] = vout.view(-1)
        self.t += 1
        return action

    def act(self, obs):
        with torch.no_grad():
            statevar = obs.cuda()
            pout, _ = self.model.pi_and_v(statevar)
            if self.act_deterministically:
                return pout.most_probable.cpu()
            else:
                return pout.sample().cpu()

    def stop_episode_and_train(self, state, reward, done=False):
        self.past_rewards[self.t - 1] = reward.cuda()  # reward = (batch_size,)
        if done:
            self.update(None)
        else:
            statevar = state.cuda()
            self.update(statevar)
        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    def stop_episode(self):
        if isinstance(self.model, Recurrent):
            self.model.reset_state()

    # def load(self, dirname): #TODO
    #     super().load(dirname)
    #     copy_param(target_model=self.shared_model,
    #                           source_model=self.model)

    def get_statistics(self):
        return [
            ('average_value', self.average_value),
            ('average_entropy', self.average_entropy),
        ]