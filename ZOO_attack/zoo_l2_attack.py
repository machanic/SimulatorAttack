import time

from config import IN_CHANNELS, IMAGE_SIZE, CLASS_NUM
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from numba import jit
import glog as log
import scipy

BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # if we stop improving, abort gradient descent early
LEARNING_RATE = 2e-3     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 0.5      # the initial constant c to pick as a first guess
@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # indice = np.array(range(0, 3*299*299), dtype = np.int32)
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
    # true_grads = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
    # true_grads, losses, l2s, scores, nimgs = self.sess.run([self.grad_op, self.loss, self.l2dist, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
    # grad = true_grads[0].reshape(-1)[indice]
    # print(grad, true_grads[0].reshape(-1)[indice])
    # self.real_modifier.reshape(-1)[indice] -= self.LEARNING_RATE * grad
    # self.real_modifier -= self.LEARNING_RATE * true_grads[0]
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
    # print(grad)
    # print(old_val - m[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

@jit(nopython=True)
def coordinate_Newton(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    # def sign(x):
    #     return np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
        hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (0.0001 * 0.0001)
    # print("New epoch:")
    # print('grad', grad)
    # print('hess', hess)
    # hess[hess < 0] = 1.0
    # hess[np.abs(hess) < 0.1] = sign(hess[np.abs(hess) < 0.1]) * 0.1
    # negative hessian cannot provide second order information, just do a gradient descent
    hess[hess < 0] = 1.0
    # hessian too small, could be numerical problems
    hess[hess < 0.1] = 0.1
    # print(hess)
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * grad / hess
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    # print('delta', old_val - m[indice])
    m[indice] = old_val
    # print(m[indice])

@jit(nopython=True)
def coordinate_Newton_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
        hess[i] = (losses[i*2+1] - 2 * cur_loss + losses[i*2+2]) / (0.0001 * 0.0001)
    # print("New epoch:")
    # print(grad)
    # print(hess)
    # positive hessian, using newton's method
    hess_indice = (hess >= 0)
    # print(hess_indice)
    # negative hessian, using ADAM
    adam_indice = (hess < 0)
    # print(adam_indice)
    # print(sum(hess_indice), sum(adam_indice))
    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    # hess[np.abs(hess) < 0.1] = sign(hess[np.abs(hess) < 0.1]) * 0.1
    # print(adam_indice)
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
    # old_val -= lr * grad[adam_indice]
    # set it back to [-0.5, +0.5] region
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice[adam_indice]]), down[indice[adam_indice]])
    m[indice[adam_indice]] = old_val
    adam_epoch[indice] = epoch + 1
    # print(m[indice])

class ZOOL2Attack(object):
    def __init__(self, dataset, model, batch_size, confidence = CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS, print_every=100,
                 early_stop_iters=0,
                 abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST,
                 use_log=False, use_tanh=True, use_resize=False, adam_beta1=0.9, adam_beta2=0.999,
                 reset_adam_after_found=False,
                 solver="adam", save_ckpts="", load_checkpoint="", start_iter=0,
                 init_size=32, use_importance=True):
        """
            The L_2 optimized attack.
            This attack is the most efficient and should be used as the primary
            attack to evaluate potential defenses.
            Returns adversarial examples for the supplied model.

            confidence: Confidence of adversarial examples: higher produces examples
              that are farther away, but more strongly classified as adversarial.
            batch_size: Number of gradient evaluations to run simultaneously.
            targeted: True if we should perform a targetted attack, False otherwise.
            learning_rate: The learning rate for the attack algorithm. Smaller values
              produce better results but are slower to converge.
            binary_search_steps: The number of times we perform binary search to
              find the optimal tradeoff-constant between distance and confidence.
            max_iterations: The maximum number of iterations. Larger values are more
              accurate; setting too small will require a large learning rate and will
              produce poor results.
            abort_early: If true, allows early aborts if gradient descent gets stuck.
            initial_const: The initial tradeoff-constant to use to tune the relative
              importance of distance and confidence. If binary_search_steps is large,
              the initial constant is not important.
        """
        image_size, in_channels, num_classes = IMAGE_SIZE[dataset][0],IN_CHANNELS[dataset], CLASS_NUM[dataset]
        self.in_channels = in_channels
        self.model = model
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iterations // 10
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.start_iter = start_iter
        self.batch_size= batch_size
        self.resize_init_size = init_size
        self.use_importance = use_importance
        if use_resize:
            self.small_x = self.resize_init_size
            self.small_y = self.resize_init_size
        else:
            self.small_x = image_size
            self.small_y = image_size
        self.use_tanh = use_tanh
        self.use_resize = use_resize
        self.repeat = binary_search_steps >= 10
        small_single_shape = (self.small_x, self.small_y, in_channels)
        if self.use_resize:
            self.scaled_modifier = nn.UpsamplingBilinear2d(size=(image_size, image_size))
        else:
            self.scaled_modifier = lambda x: x
        if load_checkpoint:
            # if checkpoint is incorrect reshape will fail
            print("Using checkpint", load_checkpoint)
            self.real_modifier = torch.from_numpy(np.load(load_checkpoint).reshape((1,) + small_single_shape)).cuda()
        else:
            self.real_modifier = torch.zeros((1,) + small_single_shape, dtype=torch.float32).cuda()

        self.tlab = torch.zeros(num_classes).long().cuda()
        self.const = torch.tensor(0.0).float().cuda()
        if use_tanh:
            self.newimg = lambda x, timg: F.tanh(self.scaled_modifier(x) + timg) / 2.0 # convert pixel values to 0~1 range
            self.l2dist = lambda x, timg: torch.sum((self.newimg(x, timg) - F.tanh(timg) / 2).pow(2), (1, 2, 3))
        else:
            self.newimg = lambda x, timg: self.scaled_modifier(x) + timg
            self.l2dist = lambda x, timg: torch.sum((self.newimg(x, timg) - timg).pow(2), (1,2,3))

        if self.targeted:
            if use_log:
                self.loss1 = lambda real_val, other_val: self.const * torch.max(torch.zeros_like(other_val),
                                                          torch.log(other_val + 1e-30) - torch.log(real_val + 1e-30))
            else:
                self.loss1 = lambda real_val, other_val: self.const * torch.max(torch.zeros_like(other_val),
                                                              other_val - real_val + self.confidence)
        else:
            if use_log:
                self.loss1 = lambda real_val, other_val: self.const * torch.max(torch.zeros_like(other_val),
                                                        torch.log(real_val + 1e-30) - torch.log(other_val + 1e-30))
            else:
                self.loss1 = lambda real_val, other_val: self.const * torch.max(torch.zeros_like(other_val),
                                                              real_val - other_val + self.confidence)
        self.loss2 = self.l2dist

        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * in_channels
        self.use_var_len = var_size
        self.var_list = torch.from_numpy(np.array(range(0, self.use_var_len), dtype = np.int32))
        self.used_var_list = torch.zeros(var_size, dtype=torch.int)
        self.sample_prob = torch.ones(var_size, dtype=torch.float32) / var_size
        # upper and lower bounds for the modifier
        self.modifier_up = torch.zeros(var_size, dtype=torch.float32)
        self.modifier_down = torch.zeros(var_size, dtype=torch.float32)
        # random permutation for coordinate update
        self.perm = torch.from_numpy(np.random.permutation(var_size))
        self.perm_index = 0
        # ADAM status
        self.mt = torch.zeros(var_size, dtype=torch.float32)
        self.vt = torch.zeros(var_size, dtype=torch.float32)
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.reset_adam_after_found = reset_adam_after_found
        self.adam_epoch = torch.ones(var_size, dtype=torch.int32)
        self.stage = 0
        # variables used during optimization process
        self.grad = torch.zeros(batch_size, dtype=torch.float32)
        self.hess = torch.zeros(batch_size, dtype=torch.float32)
        # for testing
        self.grad_op = lambda loss, modifier: torch.autograd.grad(loss, modifier)
        # compile numba function
        # self.coordinate_ADAM_numba = jit(coordinate_ADAM, nopython = True)
        # self.coordinate_ADAM_numba.recompile()
        # print(self.coordinate_ADAM_numba.inspect_llvm())
        # np.set_printoptions(threshold=np.nan)
        # set solver
        solver = solver.lower()
        self.solver_name = solver
        if solver == "adam":
            self.solver = coordinate_ADAM
        elif solver == "newton":
            self.solver = coordinate_Newton
        elif solver == "adam_newton":
            self.solver = coordinate_Newton_ADAM
        elif solver != "fake_zero":
            log.info("unknown solver", solver)
            self.solver = coordinate_ADAM
        log.info("Using", solver, "solver")



    def loss(self, adv_x, clean_x, tlab):
        newimg = self.newimg(adv_x, clean_x)
        with torch.no_grad():
            logits = self.model(newimg)
        real_val =  self.real(logits, tlab)
        other_val = self.other(logits, tlab)
        loss1_val = self.loss1(real_val, other_val)
        loss2_val = self.loss2(adv_x, clean_x).mean()
        assert loss1_val.size() == loss2_val.size()
        assert adv_x.size(0) == clean_x.size(0) == 1
        return loss1_val + loss2_val, loss1_val, loss2_val, logits


    def real(self, logits, tlab):
        out = logits[torch.arange(logits.size(0)), tlab.long()]
        return out

    def other(self, logits, tlab):
        _, argsort = logits.sort(dim=1, descending=True)
        gt_is_max = argsort[:, 0].eq(tlab).long()
        second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
        second_max_logit = logits[torch.arange(logits.size(0)), second_max_index]  # shape = batch_size
        return second_max_logit

    def max_pooling(self, image, size):
        img_pool = np.copy(image)
        img_x = image.shape[0]
        img_y = image.shape[1]
        for i in range(0, img_x, size):
            for j in range(0, img_y, size):
                img_pool[i:i + size, j:j + size] = np.max(image[i:i + size, j:j + size])
        return img_pool

    def get_new_prob(self, prev_modifier, gen_double = False):
        prev_modifier = torch.squeeze(prev_modifier)
        old_shape = prev_modifier.shape
        if gen_double:
            new_shape = (old_shape[0] * 2, old_shape[1] * 2, old_shape[2])
        else:
            new_shape = old_shape
        prob = np.empty(shape=new_shape, dtype=np.float32)
        for i in range(prev_modifier.shape[2]):
            image = np.abs(prev_modifier[:, :, i].detach().cpu().numpy())
            image_pool = self.max_pooling(image, old_shape[0] // 8)
            if gen_double:
                prob[:, :, i] = scipy.misc.imresize(image_pool, 2.0, 'nearest', mode='F')
            else:
                prob[:, :, i] = image_pool
        prob /= np.sum(prob)
        return prob

    def resize_img(self, small_x, small_y, reset_only=False):
        self.small_x = small_x
        self.small_y = small_y
        small_single_shape = (self.small_x, self.small_y, self.in_channels)
        if reset_only:
            self.real_modifier = np.zeros((1,) + small_single_shape, dtype=np.float32)
        else:
            # run the resize_op once to get the scaled image
            prev_modifier = np.copy(self.real_modifier)
            # FIXME
            self.real_modifier = self.sess.run(self.resize_op, feed_dict={self.resize_size_x: self.small_x,
                                                                          self.resize_size_y: self.small_y,
                                                                          self.resize_input: self.real_modifier})
        # prepare the list of all valid variables
        var_size = self.small_x * self.small_y * self.in_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype=np.int32)
        # ADAM status
        self.mt = np.zeros(var_size, dtype=np.float32)
        self.vt = np.zeros(var_size, dtype=np.float32)
        self.adam_epoch = np.ones(var_size, dtype=np.int32)
        # update sample probability
        if reset_only:
            self.sample_prob = np.ones(var_size, dtype=np.float32) / var_size
        else:
            self.sample_prob = self.get_new_prob(prev_modifier, True)
            self.sample_prob = self.sample_prob.reshape(var_size)


    def blackbox_optimizer(self, iteration, timg, tlab):
        var = self.real_modifier.repeat(self.batch_size * 2 + 1,1,1,1)
        var_size = self.real_modifier.nelement()
        if self.use_importance:
            var_indice = np.random.choice(self.var_list.nelement(), self.batch_size, replace=False, p = self.sample_prob)
        else:
            var_indice = np.random.choice(self.var_list.nelement(), self.batch_size, replace=False)
        indice = self.var_list[var_indice]
        for i in range(self.batch_size):
            var[i * 2 + 1].view(-1)[indice[i]] +=  0.0001
            var[i * 2 + 2].view(-1)[indice[i]] -= 0.0001
        modifier = var
        scale_modifier = self.scaled_modifier(modifier)
        newimg = self.newimg(scale_modifier, timg)
        losses, loss1, loss2, scores = self.loss(newimg, timg, tlab)
        nimgs = newimg
        self.solver(losses, indice, self.grad, self.hess, self.batch_size, self.mt, self.vt, self.real_modifier, self.modifier_up,
                    self.modifier_down, self.learning_rate, self.adam_epoch, self.beta1, self.beta2, not self.use_tanh)
        # adjust sample probability, sample around the points with large gradient
        if self.real_modifier.shape[0] > self.resize_init_size:
            self.sample_prob = self.get_new_prob(self.real_modifier)
            self.sample_prob = self.sample_prob.view(var_size)
        return losses[0].item(), loss1[0].item(), loss2[0].item(), scores[0].item(), nimgs[0]

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = x.clone()
                if self.targeted:
                    x[y] -= self.confidence
                else:
                    x[y] += self.confidence
                x = torch.argmax(x, 0)
            if self.targeted:
                return x.eq(y)
            else:
                return x.eq(y)
        if imgs.dim() == 4:
            imgs = imgs[0]
        # convert to tanh-space
        if self.use_tanh:
            imgs = torch.atan(imgs * 1.999999)
        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        CONST = self.initial_const
        upper_bound = 1e10
        # convert img to float32 to avoid numba error
        imgs = imgs.float()
        # set the upper and lower bounds for the modifier
        if not self.use_tanh:
            self.modifier_up = torch.ones_like(imgs.view(-1)) * 0.5 - imgs.view(-1)
            self.modifier_down = torch.ones_like(imgs.view(-1)) * (-0.5) - imgs.view(-1)
        # clear the modifier
        if not self.load_checkpoint:
            if self.use_resize:
                self.resize_img(self.resize_init_size, self.resize_init_size, True)
            else:
                self.real_modifier.fill_(0.0)
        # the best l2, score, and image attack
        o_best_const = CONST
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = imgs
        for outer_step in range(self.binary_search_steps):
            bestl2 = 1e10
            bestscore = -1
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.binary_search_steps - 1:
                CONST = upper_bound
            self.const = CONST
            prev = 1e6
            train_timer = 0.0
            last_loss1 = 1.0
            if not self.load_checkpoint:
                if self.use_resize:
                    self.resize_img(self.resize_init_size, self.resize_init_size, True)
                else:
                    self.real_modifier.fill_(0.0)
            # reset ADAM status
            self.mt.fill_(0.0)
            self.vt.fill_(0.0)
            self.adam_epoch.fill_(1)
            self.stage = 0
            multiplier = 1
            eval_costs = 0
            if self.solver_name != "fake_zero":
                multiplier = 24
            for iteration in range(self.start_iter, self.max_iterations):
                if self.use_resize:
                    if iteration == 2000:
                        self.resize_img(64, 64)
                    if iteration == 10000:
                        self.resize_img(128, 128)
                attack_begin_time = time.time()
                if self.solver_name == "fake_zero":
                    l, l2, loss1, loss2, score, nimg = self.fake_blackbox_optimizer()
                else:
                    l, l2, loss1, loss2, score, nimg = self.blackbox_optimizer(iteration)
                if self.solver_name == "fake_zero":
                    eval_costs += np.prod(self.real_modifier.size())
                else:
                    eval_costs += self.batch_size
                # reset ADAM states when a valid example has been found
                if loss1 == 0.0 and last_loss1 != 0.0 and self.stage == 0:
                    # we have reached the fine tunning point
                    # reset ADAM to avoid overshoot
                    if self.reset_adam_after_found:
                        self.mt.fill_(0.0)
                        self.vt.fill_(0.0)
                        self.adam_epoch.fill_(1)
                    self.stage = 1
                last_loss1 = loss1
                # check if we should abort search if we're getting nowhere.
                # if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                if self.abort_early and iteration % self.early_stop_iters == 0:
                    if l > prev * .9999:
                        log.info("Early stopping because there is no improvement")
                        break
                    prev = l
                # adjust the best result found so far
                # the best attack should have the target class with the largest value,
                # and has smallest l2 distance
                if l2 < bestl2 and compare(score, torch.argmax(labs)):
                    bestl2 = l2
                    bestscore = torch.argmax(score)
                if l2 < o_bestl2 and compare(score, torch.argmax(labs)):
                    if o_bestl2 == 1e10:
                        log.info("[STATS][L3](First valid attack found!) iter = {}, cost = {}, time = {:.3f}, size = {}, loss = {:.5g}, loss1 = {:.5g}, loss2 = {:.5g}, l2 = {:.5g}".format(iteration, eval_costs, train_timer, self.real_modifier.shape, l, loss1, loss2, l2))
                    o_bestl2 = l2
                    o_bestscore = torch.argmax(score)
                    o_bestattack = nimg
                    o_best_const = CONST
                train_timer += time.time() - attack_begin_time
            # adjust the constant as needed
            if compare(bestscore, torch.argmax(labs)) and bestscore != -1:
                upper_bound = min(upper_bound, CONST)
                if upper_bound < 1e9:
                    CONST = (lower_bound + upper_bound) / 2
                else:
                    lower_bound = max(lower_bound, CONST)
                    if upper_bound < 1e9:
                        CONST = (lower_bound + upper_bound) / 2
                    else:
                        CONST *= 10
        return o_bestattack, o_best_const

