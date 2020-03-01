import math
import time

import numpy as np
import scipy
import scipy.misc
from numba import jit
import torch
from torch.nn import functional as F
import torch
from torch.autograd import grad
import glog as log

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # indice = np.array(range(0, 3*299*299), dtype = np.int32)
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002  # grad的shape = (batch_size,), 所以已经是随机挑了个坐标的，随机挑选坐标的秘密再losses
    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

@jit(nopython=True)
def coordinate_Newton(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
        hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (0.0001 * 0.0001)

    # negative hessian cannot provide second order information, just do a gradient descent
    hess[hess < 0] = 1.0
    # hessian too small, could be numerical problems
    hess[hess < 0.1] = 0.1
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * grad / hess
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    m[indice] = old_val

@jit(nopython=True)
def coordinate_Newton_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
        hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (0.0001 * 0.0001)
    # positive hessian, using newton's method
    hess_indice = (hess >= 0)
    # print(hess_indice)
    # negative hessian, using ADAM
    adam_indice = (hess < 0)
    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    # Newton's Method
    m = real_modifier.reshape(-1)
    old_val = m[indice[hess_indice]]
    old_val -= lr * grad[hess_indice] / hess[hess_indice]
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice[hess_indice]]), down[indice[hess_indice]])
    m[indice[hess_indice]] = old_val
    # ADMM
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch[adam_indice]))) / (1 - np.power(beta1, epoch[adam_indice]))
    old_val = m[indice[adam_indice]]
    old_val -= lr * corr * mt[adam_indice] / (np.sqrt(vt[adam_indice]) + 1e-8)
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice[adam_indice]]), down[indice[adam_indice]])
    m[indice[adam_indice]] = old_val
    adam_epoch[indice] = epoch + 1

class ZOOAttack(object):
    def __init__(self, model, num_channels, img_size, args):
        self.image_size = model.input_size[-1]
        self.num_channels = num_channels
        self.model = model.cuda().eval()
        self.targeted = args.targeted
        self.BATCH_SIZE = args.batch_size  # Number of gradient evaluations to run simultaneously.
        self.LEARNING_RATE = args.lr
        self.MAX_ITERATIONS = 200
        self.print_every = args.print_every
        # self.early_stop_iters = args.early_stop_iters if args.early_stop_iters != 0 else args.max_iterations // 10
        self.ABORT_EARLY = args.abort_early
        self.BINARY_SEARCH_STEPS = args.binary_steps
        self.CONFIDENCE = args.confidence
        self.use_log = args.use_log
        self.initial_const = args.init_const
        self.start_iter = args.start_iter
        self.resize_init_size = args.init_size

        self.use_importance = not args.uniform
        if args.resize:
            self.small_x = args.init_size
            self.small_y = args.init_size
        else:
            self.small_x = img_size
            self.small_y = img_size
        self.use_tanh = args.use_tanh
        self.resize = args.resize
        self.repeat = args.binary_steps >= 10
        shape = (None, num_channels, img_size, img_size)
        self.single_shape = (num_channels, img_size, img_size)
        small_single_shape = (num_channels, self.small_x, self.small_y)
        self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        self.used_var_list = np.zeros(var_size, dtype=np.int32)
        self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size
        # upper and lower bounds for the modifier
        self.modifier_up = np.zeros(var_size, dtype=np.float32)
        self.modifier_down = np.zeros(var_size, dtype=np.float32)
        if self.use_tanh:
            self.l2dist = lambda newimg, timg: torch.sum((newimg - (F.tanh(timg) + 0.5 * 1.99999) / 2).pow(2), (1, 2, 3))  # (F.tanh(timg) + 0.5 * 1.99999) / 2 range (0,1)
        else:
            self.l2dist = lambda newimg, timg: torch.sum((newimg - timg).pow(2), (1,2,3))
        self.loss2 = self.l2dist
        self.epsilone = args.epsilone
        # random permutation for coordinate update
        self.perm = np.random.permutation(var_size)
        self.perm_index = 0
        # ADAM status
        self.mt = np.zeros(var_size, dtype=np.float32)
        self.vt = np.zeros(var_size, dtype=np.float32)
        self.beta1 = args.adam_beta1
        self.beta2 = args.adam_beta2
        self.reset_adam_after_found = args.reset_adam
        self.adam_epoch = np.ones(var_size, dtype=np.int32)
        self.stage = 0
        # variables used during optimization process
        self.grad = np.zeros(self.BATCH_SIZE, dtype=np.float32)
        self.hess = np.zeros(self.BATCH_SIZE, dtype=np.float32)
        solver = args.solver.lower()
        self.solver_name = solver
        if solver == "adam":
            self.solver = coordinate_ADAM
        elif solver == "newton":
            self.solver = coordinate_Newton
        elif solver == "adam_newton":
            self.solver = coordinate_Newton_ADAM
        elif solver != "fake_zero":
            print("unknown solver", solver)
            self.solver = coordinate_ADAM
        log.info("Using", solver, "solver")

    def max_pooling(self, image, size):
        img_pool = np.copy(image)
        img_x = image.shape[0]
        img_y = image.shape[1]
        for i in range(0, img_x, size):
            for j in range(0, img_y, size):
                img_pool[i:i + size, j:j + size] = np.max(image[i:i + size, j:j + size])
        return img_pool

    def get_new_prob(self, prev_modifier, gen_double=False):
        prev_modifier = np.squeeze(prev_modifier)
        old_shape = prev_modifier.shape
        if gen_double:
            new_shape = (old_shape[0] * 2, old_shape[1] * 2, old_shape[2])
        else:
            new_shape = old_shape
        prob = np.empty(shape=new_shape, dtype=np.float32)
        for i in range(prev_modifier.shape[2]):
            image = np.abs(prev_modifier[:, :, i])
            image_pool = self.max_pooling(image, old_shape[0] // 8)
            if gen_double:
                prob[:, :, i] = scipy.misc.imresize(image_pool, 2.0, 'nearest', mode='F')
            else:
                prob[:, :, i] = image_pool
        prob /= np.sum(prob)
        return prob

    def resize_op(self, resize_input, resize_size_x, resize_size_y):
        resize_input = torch.from_numpy(resize_input)
        return F.interpolate(resize_input, (resize_size_y, resize_size_x), mode='bilinear',align_corners=True).detach().numpy()

    def resize_img(self, small_x, small_y, reset_only = False):
        self.small_x = small_x
        self.small_y = small_y
        small_single_shape = (self.num_channels, self.small_x, self.small_y)
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        # ADAM status
        self.mt = np.zeros(var_size, dtype=np.float32)
        self.vt = np.zeros(var_size, dtype=np.float32)
        self.adam_epoch = np.ones(var_size, dtype=np.int32)
        # update sample probability
        if reset_only:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
            self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size
        else:
            prev_modifier = np.copy(self.real_modifier)
            self.real_modifier = self.resize_op(self.real_modifier, self.small_x, self.small_y)
            self.sample_prob = self.get_new_prob(prev_modifier, True)
            self.sample_prob = self.sample_prob.reshape(var_size)


    def loss(self, newimg, timg, true_label, target_label, const):
        with torch.no_grad():
            logits = self.model(newimg)  # B, # num_class
        if self.use_log:
            logits = F.log_softmax(logits, dim=1)
        # real_val =  self.real(logits, tlab)
        # other_val = self.other(logits, tlab)  # shape = (batch_size,)
        # loss1_val = self.loss1(real_val, other_val)  # shape = (batch_size,)
        if true_label.size(0)!=logits.size(0):
            assert logits.size(0) % true_label.size(0) == 0
            repeat_num = logits.size(0) // true_label.size(0)
            true_label = true_label.repeat(repeat_num)
            if target_label is not None:
                target_label = target_label.repeat(repeat_num)
        loss1_val = self.cw_loss(logits, true_label, target_label)
        loss2_val = self.loss2(newimg, timg)  # returned shape = (batch_size,) ; clean_x shape = (1,C,H,W) newimg = (B,C,H,W)
        assert loss1_val.size() == loss2_val.size()
        return const * loss1_val + loss2_val, loss1_val, loss2_val, logits


    def cw_loss(self, logit, label, target):
        # logit = F.log_softmax(logit, dim=1)
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), target]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return torch.clamp(second_max_logit - target_logit + self.CONFIDENCE,min=0.0)
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return torch.clamp(gt_logit - second_max_logit + self.CONFIDENCE, min=0.0)

    def get_newimg(self, timg, modifier):
        if self.resize:
            scaled_modifier = F.interpolate(modifier, self.image_size, mode="bilinear",align_corners=True)

        else:
            scaled_modifier = modifier
        if self.use_tanh:
            newimg = (F.tanh(scaled_modifier + timg) + 0.5 * 1.99999) / 2
        else:
            newimg = scaled_modifier + timg
        return newimg


    def fake_blackbox_optimizier(self, timg, true_label, target_label, const):
        newimg = self.get_newimg(timg, torch.from_numpy(self.real_modifier).cuda())
        losses, loss1, loss2, scores = self.loss(newimg, timg, true_label, target_label, const)
        true_grads = torch.autograd.grad(losses, self.real_modifier)
        grad = true_grads[0].detach().cpu().numpy().reshape(-1)
        epoch = self.adam_epoch[0]
        mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        vt = self.beta2 * self.vt + (1 - self.beta2) * np.square(grad)
        corr = (math.sqrt(1 - self.beta2 ** epoch)) / (1 - self.beta1 ** epoch)
        m = self.real_modifier.reshape(-1)
        m -= self.LEARNING_RATE * corr * (mt / (np.sqrt(vt) + 1e-8))
        self.mt = mt
        self.vt = vt
        if not self.use_tanh:
            m_proj = np.maximum(np.minimum(m, self.modifier_up), self.modifier_down)
            np.copyto(m, m_proj)
        self.adam_epoch[0] = epoch + 1
        return losses[0].item(), loss2[0].item(), loss1[0].item(), loss2[0].item(), scores[0], newimg[0].unsqueeze(0)


    def blackbox_optimizer(self, timg, true_label, target_label, const):
        # build new inputs, based on current variable value
        var = np.repeat(self.real_modifier, self.BATCH_SIZE * 2 + 1, axis=0)
        var_size = self.real_modifier.size
        if self.use_importance:
            var_indice = np.random.choice(self.var_list.size, self.BATCH_SIZE, replace=False, p=self.sample_prob)
        else:
            var_indice = np.random.choice(self.var_list.size, self.BATCH_SIZE, replace=False) # randomly pick a coordinate to calculate
        indice = self.var_list[var_indice]
        for i in range(self.BATCH_SIZE):
            var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
            var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0001
        modifier = torch.from_numpy(var).cuda().float()
        newimg = self.get_newimg(timg, modifier)
        losses, loss1, loss2, scores = self.loss(newimg, timg, true_label, target_label, const)
        self.solver(losses.detach().cpu().numpy(), indice, self.grad, self.hess, self.BATCH_SIZE, self.mt, self.vt,
                    self.real_modifier, self.modifier_up, self.modifier_down,
                    self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)

        if self.real_modifier.shape[0] > self.resize_init_size:  # FIXME 为何是[0],感觉是bug
            self.sample_prob = self.get_new_prob(self.real_modifier)
            self.sample_prob = self.sample_prob.reshape(var_size)
        return losses[0].item(), loss2[0].item(), loss1[0].item(), loss2[0].item(), scores[0], newimg[0].unsqueeze(0)

    def compare(self, x, true_label, target_label):
        if target_label is None:
            y = true_label[0].item()
        else:
            assert self.targeted
            y = target_label[0].item()
        temp_x = np.copy(x)
        if not isinstance(x, (float, int, np.int64)):
            if self.targeted:
                temp_x[y] -= self.CONFIDENCE
            else:
                temp_x[y] += self.CONFIDENCE
            temp_x = np.argmax(temp_x)
        if self.targeted:
            return temp_x == y
        else:
            return temp_x != y

    def attack(self, img, true_label, target_label):
        batch_size = img.size(0)
        query = torch.zeros(batch_size).cuda()
        with torch.no_grad():
            logit = self.model(img)
        pred = logit.argmax(dim=1)
        correct = pred.eq(true_label).float()  # shape = (batch_size,)
        not_done = correct.clone()  # shape = (batch_size,)

        prob = F.softmax(logit, dim=1)
        success = (1 - not_done) * correct
        success_query = success * query
        not_done_prob = prob[torch.arange(batch_size), true_label] * not_done
        adv_loss = self.cw_loss(logit, true_label, target_label)
        not_done_loss = adv_loss * not_done

        if self.use_tanh:  # tf版本像素范围-0.5到0.5
            img = atanh((img - 0.5) * 1.99999)  # atanh形参数值范围(-1,1)
        else:
            img_flatten = img.view(-1)
            self.modifier_up =  (torch.ones_like(img_flatten) - img_flatten).detach().cpu().numpy()
            self.modifier_down = (torch.zeros_like(img_flatten) - img_flatten).detach().cpu().numpy()
        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        CONST = self.initial_const
        upper_bound = 1e10
        self.real_modifier.fill(0.0)  # FIXME
        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestattack = img
        eval_costs = 0
        adv_images_founded = False
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            bestl2 = 1e10
            bestscore = -1
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound
            timg = img
            const = CONST
            prev = 1e6
            train_timer = 0.0
            last_loss1 = 1.0
            if self.resize:
                self.resize_img(self.resize_init_size, self.resize_init_size, True)
            else:
                self.real_modifier.fill(0.0)
            self.mt.fill(0.0)
            self.vt.fill(0.0)
            self.adam_epoch.fill(1)
            self.stage = 0
            for iteration in range(self.start_iter, self.MAX_ITERATIONS):
                if self.resize:
                    if iteration == 2000:
                        self.resize_img(64,64)
                    if iteration == 10000:
                        self.resize_img(128, 128)
                if iteration % (self.print_every) == 0:
                    newimg = self.get_newimg(timg, torch.from_numpy(self.real_modifier).cuda())
                    losses, loss1, loss2, scores = self.loss(newimg, timg, true_label, target_label, const)
                    log.info(
                        "[STATS][L2] iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}".format(
                            iteration, eval_costs, train_timer, self.real_modifier.shape, losses[0].item(), loss1[0].item(), loss2[0].item()))
                attack_begin_time = time.time()
                if self.solver_name == "fake_zero":
                    l, l2, loss1, loss2, score, nimg = self.fake_blackbox_optimizier(timg, true_label, target_label, const)
                else:
                    l, l2, loss1, loss2, score, nimg = self.blackbox_optimizer(timg, true_label, target_label, const)
                score = score.detach().cpu().numpy()
                if self.solver_name == "fake_zero":
                    eval_costs += np.prod(self.real_modifier.shape)  # 假的就当作把全部像素都估计一遍梯度
                else:
                    eval_costs += self.BATCH_SIZE * 2  #  FIXME 他原本的代码写错了，没乘以2
                if loss1 == 0.0 and last_loss1 != 0.0 and self.stage==0:
                    if self.reset_adam_after_found:
                        self.mt.fill(0.0)
                        self.vt.fill(0.0)
                        self.adam_epoch.fill(0.0)
                    self.stage = 1
                last_loss1 = loss1

                if l2 < bestl2 and self.compare(score, true_label, target_label):
                    bestl2 = l2
                    bestscore = np.argmax(score)
                if l2 < o_bestl2 and self.compare(score, true_label, target_label):
                    if o_bestl2 == 1e10:
                        log.info(
                            "[STATS][FirstAttack] iter:{}, const:{}, cost:{}, time:{:.3f}, size:{}, loss:{:.5g}, loss1:{:.5g}, loss2:{:.5g}, l2:{:.5g}".format(
                                iteration, CONST, eval_costs, train_timer, self.real_modifier.shape, l, loss1,
                                loss2, l2))
                    # 攻击成功了，记录query
                    o_bestl2 = l2
                    o_bestattack = nimg
                    # begin statistics
                    query.fill_(eval_costs)
                    with torch.no_grad():
                        adv_logit = self.model(o_bestattack)
                    adv_prob = F.softmax(adv_logit, dim=1)
                    adv_pred = adv_logit.argmax(dim=1)
                    if self.targeted:
                        not_done = not_done * (1 - adv_pred.eq(target_label)).float()
                    else:
                        not_done = not_done * adv_pred.eq(true_label).float()
                    success = (1 - not_done) * correct
                    success_query = success * query
                    adv_loss = self.cw_loss(adv_logit, true_label, target_label)
                    not_done_loss = adv_loss * not_done
                    not_done_prob = adv_prob[torch.arange(batch_size), true_label] * not_done

                    if self.ABORT_EARLY:
                        if loss2 <= self.epsilone and not not_done.byte().any():
                            adv_images_founded = True
                            log.info("Early stopping attack successfully and total pixels' distortion is {:.3f}".format(loss2))
                            break
                        # prev = l
                train_timer += time.time() - attack_begin_time
            if self.compare(bestscore, true_label, target_label) and bestscore!=-1:
                upper_bound = min(upper_bound, CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound) / 2
                else:
                    lower_bound = max(lower_bound, CONST)
                    if upper_bound < 1e9:
                        CONST = (lower_bound + upper_bound) / 2
                    else:
                        CONST *= 10
            if adv_images_founded:
                break
        stats_info = {"query": query.detach().float().cpu(), "correct": correct.detach().float().cpu(),
                      "not_done": not_done.detach().float().cpu(),
                      "success": success.detach().float().cpu(), "success_query": success_query.detach().float().cpu(),
                      "not_done_prob": not_done_prob.detach().float().cpu(),
                      "not_done_loss":not_done_loss.detach().float().cpu() }

        return o_bestattack, stats_info
