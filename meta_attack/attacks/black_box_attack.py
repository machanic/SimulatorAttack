import copy
import sys

import numpy as np
from torch import optim
from torch.nn import functional as F

from config import CLASS_NUM
from meta_attack.attacks.gradient_generator import GradientGenerator
from meta_attack.attacks.helpers import *
import glog as log

def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch,
                    beta1, beta2, proj):
    # for i in range(batch_size):
    #   grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002

    # grad = torch.from_numpy(grad).cuda()
    # pdb.set_trace()

    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    epoch = adam_epoch[indice]
    corr = (torch.sqrt(1 - torch.pow(beta2, epoch))) / (1 - torch.pow(beta1, epoch))
    # if self.cuda:
    corr = corr.cuda()
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * corr * mt / (torch.sqrt(vt) + 1e-8)
    if proj:
        old_val = torch.max(torch.min(old_val, up[indice]), down[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1.


class MetaAttack(object):
    def __init__(self, args, norm, epsilon, targeted=False, search_steps=None, max_steps=None, use_log=True, cuda=True, debug=False):
        self.debug = debug
        self.norm = norm
        self.epsilon = epsilon
        self.targeted = targeted  # false
        self.num_classes = CLASS_NUM[args.dataset]
        self.confidence = 0  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = 0.5  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 5
        self.repeat = self.binary_search_steps >= 10

        self.abort_early = True
        self.clip_min = -1
        self.clip_max = 1
        self.cuda = cuda
        self.init_rand = False  # an experiment, does a random starting point help?
        self.use_log = use_log
        self.use_tanh = args.use_tanh
        if self.norm == "linf":
            assert not self.use_tanh, "The linf norm must set use_tanh to False."
        self.batch_size = 784
        self.update_pixels = args.update_pixels
        self.GRAD_STORE =0
        self.guided = False
        self.simba_pixel =args.simba_update_pixels
        # self.every_iter = int(float(self.update_pixels) / float(self.simba_pixel) ) * args.finetune_interval
        self.every_iter =  args.finetune_interval
        self.finetune_interval = args.finetune_interval
        self.max_queries = args.max_queries
        self.max_steps = max_steps or 1000
        self.max_iter = self.max_steps * self.every_iter

        self.use_importance = True
        self.LEARNING_RATE = args.learning_rate
        # self.LEARNING_RATE = 1e-2
        self.beta1 = 0
        self.beta2 = 0
        self.dataset = args.dataset

        self.reset_adam_after_found = False
        self.num_channels = 3
        self.small_x = args.init_size
        self.small_y = args.init_size
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size

        self.mt = torch.zeros(var_size, dtype=torch.float32)
        self.vt = torch.zeros(var_size, dtype=torch.float32)
        self.modifier_up = torch.zeros(var_size, dtype=torch.float32)
        self.modifier_down = torch.zeros(var_size, dtype=torch.float32)
        self.grad = torch.zeros(self.batch_size, dtype=torch.float32)
        self.hess = torch.zeros(self.batch_size, dtype=torch.float32)
        self.adam_epoch = torch.ones(var_size, dtype=torch.float32)

        self.solver_name = 'adam'

        if self.solver_name == 'adam':
            self.solver = coordinate_ADAM
            # self.solver = check_optimizer
        elif self.solver_name != 'fake_zero':
            log.info('unknown solver', self.solver_name)
            self.solver = coordinate_ADAM
        log.info('Using {} sovler'.format(self.solver_name))

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output).item()
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, output, target, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            if self.use_log:
                loss1 = torch.clamp(torch.log(other + 1e-30) - torch.log(real + 1e-30), min=0.)
            else:
                # if targeted, optimize for making the other class most likely
                loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            if self.use_log:
                loss1 = torch.clamp(torch.log(real + 1e-30) - torch.log(other + 1e-30), min=0.)
            else:
                # if non-targeted, optimize for making this class least likely.
                loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = scale_const * loss1
        # loss2 = dist.squeeze(1)
        loss2 = loss1.clone()
        loss = loss1 + loss2
        return loss, loss1, loss2

    def normalize(self, t, p=2):
        assert len(t.shape) == 4
        if p == 2:
            norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        elif p == 1:
            norm_vec = t.abs().sum(dim=[1, 2, 3]).view(-1, 1, 1, 1)
        else:
            raise NotImplementedError('Unknown norm p={}'.format(p))
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def l2_dist_within_epsilon(self, image, adv_image, epsilon):
        delta = adv_image - image
        inside_l2_bound = (self.normalize(delta) <= epsilon).view(-1).byte()  # norm返回shape(N,1,1,1) > epsilon
        return inside_l2_bound.all().item()  # 如果全是1.返回True,表示没有一个像素超出范围外

    def _optimize(self, model, meta_model, step, input_var, modifier_var, target_var, scale_const_var, target,
                  indice, input_orig=None):
        query = 0
        if self.use_tanh:
            input_adv = (tanh_rescale(modifier_var + input_var) + 1)/ 2  # 0~1
        else:
            input_adv = modifier_var + input_var
        output = F.softmax(model(input_adv), dim=1) # query model for 1 time
        query += 1
        if input_orig is None:
            dist = l2_dist(input_adv, input_var, keepdim=True).squeeze(2).squeeze(2)
        else:
            dist = l2_dist(input_adv, input_orig, keepdim=True).squeeze(2).squeeze(2)
        loss, loss1, loss2 = self._loss(output.data, target_var, scale_const_var)
        meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.01)
        input_adv_copy = copy.deepcopy(input_adv.detach())
        if self.guided and step == 0:
            meta_output = meta_model(input_adv.detach())
            indice = torch.abs(meta_output.data).cpu().numpy().reshape(-1).argsort()[-500:]
        if (step + 1) % self.every_iter == 0 and step < self.max_iter:
            zoo_gradients = []
            generate_grad = GradientGenerator(update_pixels=self.update_pixels,
                                              targeted=self.targeted, classes=CLASS_NUM[self.dataset])
            zoo_grad, select_indice = generate_grad.run(model, input_adv_copy, target, indice)  # query for batch_size times
            query += self.update_pixels
            zoo_gradients.append(zoo_grad)
            zoo_gradients = np.array(zoo_gradients, np.float32)
            zoo_gradients = torch.from_numpy(zoo_gradients).cuda()

            std = zoo_gradients.cpu().numpy().std(axis=(1, 2, 3))
            std = std.reshape((-1, 1, 1, 1)) + 1e-23
            zoo_gradients = zoo_gradients / torch.from_numpy(std).cuda()
            assert not torch.isnan(zoo_gradients.sum())
            for i in range(20):
                meta_optimizer.zero_grad()
                meta_grads = meta_model(input_adv_copy)
                # meta_grads = meta_grads * torch.sign(torch.abs(zoo_gradients))
                # meta_loss = F.mse_loss(meta_grads, zoo_gradients)
                meta_loss = F.mse_loss(meta_grads.reshape(-1)[select_indice], zoo_gradients.reshape(-1)[select_indice])
                meta_loss.backward()
                meta_optimizer.step()
        meta_output = meta_model(input_adv.detach())
        indice = torch.abs(meta_output.data).cpu().numpy().reshape(-1).argsort()[-self.update_pixels:]
        indice2 = indice
        if (step + 1) % self.every_iter == 0 and step < self.max_iter:
            grad = zoo_gradients.reshape(-1)[indice2]
        else:
            grad = meta_output.reshape(-1)[indice2]
        # 修改的是modifier_var来达到修改对抗样本的目的
        self.solver(loss, indice2, grad, self.hess, self.batch_size, self.mt, self.vt, modifier_var,
                    self.modifier_up, self.modifier_down, self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2,
                    not self.use_tanh)  # use_tanh模式，就不使用modifier_up和modifier_down
        loss_np = loss[0].item()
        loss1 = loss1[0].item()
        loss2 = loss2[0].item()
        dist_np = dist[0].data.cpu().numpy()
        output_np = output[0].unsqueeze(0).data.cpu().numpy()
        input_adv_np = input_adv[0].unsqueeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()  # back to BHWC for numpy consumption
        return loss_np, loss1, loss2, dist_np, output_np, input_adv_np, indice, query

    def run(self, model, meta_model, input, target):
        total_queries = 0
        success_queries = 0
        batch_size, c, h, w = input.size()
        var_size = c * h * w
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        if not self.use_tanh:  # 此时modifier_up和modifier_down起作用，可以设置为linf的攻击的eps半径
            if self.norm == "linf":
                self.modifier_up = torch.clamp(input.view(-1) + self.epsilon, min=0, max=1) - input.view(-1)
                self.modifier_down = torch.clamp(input.view(-1) - self.epsilon, min=0, max=1) - input.view(-1)
            else:
                self.modifier_up = 1 - input.reshape(-1)  # 像素范围是0到1之间，这是修改量的范围
                self.modifier_down = 0 - input.reshape(-1)
        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = input.permute(0, 2, 3, 1).cpu().numpy()  # put channel as the last dimension: BHWC
        # setup input (image) variable, clamp/scale as necessary
        if self.use_tanh:  # l2模式使用, linf norm使用no_tanh
            input_var = torch_arctanh(input * 2 - 1).detach()  # arctanh接受的是-1到1
            input_var.requires_grad = False
            input_orig = (tanh_rescale(input_var) + 1) / 2
        else:
            input_var = input.detach()
            input_var.requires_grad = False
            input_orig = input_var.clone()
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.cuda:
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = target_onehot.detach()
        target_var.requires_grad = False
        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        self.mt = torch.zeros(var_size, dtype=torch.float32)
        self.vt = torch.zeros(var_size, dtype=torch.float32)
        self.adam_epoch = torch.ones(var_size, dtype=torch.float32)
        stage = 0
        if self.init_rand:
            modifier = torch.normal(mean=modifier, std=0.001)
        if self.cuda:
            modifier = modifier.cuda()
            self.modifier_up = self.modifier_up.cuda()
            self.modifier_down = self.modifier_down.cuda()
            self.mt = self.mt.cuda()
            self.vt = self.vt.cuda()
            self.grad = self.grad.cuda()
            self.hess = self.hess.cuda()
        modifier_var = modifier
        first_step = 0
        for search_step in range(self.binary_search_steps):
            if self.debug:
                log.info('Const:')
                for i, x in enumerate(scale_const):
                    log.info("{} {}".format(i, x))
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.cuda:
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = scale_const_tensor.detach()
            scale_const_var.requires_grad = False
            indice = np.zeros(250)
            last_loss1 = 1.0
            for step in range(self.max_steps):
                loss, loss1, loss2, dist, output, adv_img, indice, query = self._optimize(
                    model, meta_model, step,
                    input_var, modifier_var, target_var,
                    scale_const_var, target, indice, input_orig)
                total_queries += query
                if loss1 == 0.0 and last_loss1 != 0 and stage == 0:
                    if self.reset_adam_after_found:
                        self.mt = torch.zeros(var_size, dtype=torch.float32)
                        self.vt = torch.zeros(var_size, dtype=torch.float32)
                        self.adam_epoch = torch.ones(var_size, dtype=torch.float32)
                    stage = 1
                last_loss1 = loss1
                if step % 100 == 0 or step == self.max_steps - 1:
                    log.info(
                        'Step: {0:>4}, loss: {1:6.4f}, loss1: {2:5f}, loss2: {3:5f}, dist: {4:8.5f}, modifier mean: {5:.5e}'.format(
                            step, loss, loss1, loss2, dist.mean(), modifier_var.mean()))
                if self.abort_early and first_step != 0:
                    log.info('Aborting early...')
                    break
                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = dist[i]
                    if self.debug:
                        if step % 100 == 0:
                            log.info('{0:>2} dist: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    if di < best_l2[i] and self._compare(output_logits, target_label.item()):
                        if self.debug:
                            log.info('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                    if di < o_best_l2[i] and self._compare(output_logits, target_label.item()):
                        if self.debug:
                            log.info('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label

                        adv_img_tensor = torch.from_numpy(adv_img[i]).unsqueeze(0).permute(0,3, 1,2) # BHWC->BCHW
                        input_orig_tensor = input_orig[i].unsqueeze(0).cpu()   # BCHW
                        assert adv_img_tensor.size() == input_orig_tensor.size()
                        if self.norm == "l2" and self.l2_dist_within_epsilon(input_orig_tensor,
                                                                             adv_img_tensor, self.epsilon):
                            success_queries = total_queries
                            o_best_attack[i] = adv_img[i]  # 找到一张图
                            first_step = step  # 找到对抗样本的迭代次数
                        elif self.norm == "linf":
                            success_queries = total_queries
                            o_best_attack[i] = adv_img[i]  # 找到一张图
                            first_step = step  # 找到对抗样本的迭代次数

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                if self._compare(best_score[i], target[i]) and best_score[i] != -1:
                    # successful, do binary search and divide const by two
                    upper_bound[i] = min(upper_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if self.debug:
                        log.info('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                            i, scale_const[i]))
                else:
                    # failure, multiply by 10 if no solution found
                    # or do binary search with the known upper bound
                    lower_bound[i] = max(lower_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        scale_const[i] *= 10
                    if self.debug:
                        log.info('{0:>2} failed attack, raising const to {1:.3f}'.format(
                            i, scale_const[i]))
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                else:
                    batch_failure += 1
            log.info('Num failures: {0:2d}, num successes: {1:2d}.'.format(batch_failure, batch_success))

        if first_step > self.max_iter:
            first_step = self.max_iter
        o_best_l2 = np.sqrt(np.array(o_best_l2))[0].item()
        return o_best_attack, o_best_l2, scale_const, first_step, success_queries