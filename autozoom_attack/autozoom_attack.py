import time
from abc import ABCMeta, abstractmethod

import glog as log
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from config import IN_CHANNELS, CLASS_NUM


# settings for ADAM solver
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999


def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, lr, adam_epoch, beta1, beta2, proj):
    for i in range(batch_size):
        grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
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
    old_val -= lr * corr * mt / (np.sqrt(vt)  + 1e-8 )

    m[indice] = old_val
    adam_epoch[indice] = epoch + 1



# 原作者的image是-0.5到+0.5之间
def SGD(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, lr, adam_epoch, beta1, beta2, proj, beta, z, q=1):
    for i in range(q):
        grad[i] = q * (losses[i+1] - losses[0]) * z[i] / beta

    # argument indice should be removed for the next version
    # the entire modifier is updated for every epoch and thus indice is not required
    avg_grad = np.mean(grad, axis=0)
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * np.sign(avg_grad)
    m[indice] = old_val


# 原作者的image是-0.5到+0.5之间
def ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, lr, adam_epoch, beta1, beta2, proj, beta, z, q=1):
    for i in range(q):
        grad[i] = q * (losses[i+1] - losses[0]) * z[i] / beta

    # argument indice should be removed for the next version
    # the entire modifier is updated for every epoch and thus indice is not required
    avg_grad = np.mean(grad, axis=0)
    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * avg_grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (avg_grad * avg_grad)
    vt_arr[indice] = vt

    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * corr * mt / (np.sqrt(vt)  + 1e-8 )
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

class BlackboxAttack(metaclass=ABCMeta):
    def __init__(self,  model, dataset, args):
        # data information
        self.model = model
        self.model.eval().cuda()
        self.num_channels = IN_CHANNELS[dataset]
        self.image_size = model.input_size[-1]
        self.num_labels = CLASS_NUM[dataset]
        # attack settings
        self.MAX_ITER = args["max_iterations"]
        self.PRINT_EVERY = args["print_every"]
        self.SWITCH_ITER = args["switch_iterations"]
        self.INIT_CONST = args["init_const"]
        self.ATTACK_TYPE = args["attack_type"]
        self.USE_TANH = args["use_tanh"]
        self.BATCH_SIZE = args["batch_size"]  # number of pixels' gradients evaluations to run simultaneously.
        self.LEARNING_RATE = args["lr"]
        self.CONFIDENCE = args["confidence"]
        self.ABORT_EARLY = True  # # if we stop improving, abort gradient descent early
        self.modifier_size = args["img_resize"]
        self.epsilone = args["epsilone"]
        self.image_shape = (self.num_channels, self.image_size, self.image_size)
        self.modifier_shape = (self.num_channels, self.modifier_size, self.modifier_size)
        # self.early_stop_iters = args["early_stop_iters"] if args["early_stop_iters"] != 0 else self.MAX_ITER // 10
        self.real_modifier = np.zeros((1,) + self.modifier_shape, dtype=np.float32)
        self.define_loss_func()

        self.var_size = self.modifier_size * self.modifier_size * self.num_channels
        self.use_var_len = self.var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        # ADAM status
        self.mt = np.zeros(self.var_size, dtype=np.float32)
        self.vt = np.zeros(self.var_size, dtype=np.float32)
        self.beta1 = ADAM_BETA1
        self.beta2 = ADAM_BETA2
        self.adam_epoch = np.ones(self.var_size, dtype=np.int32)
        self.stage = 0

    def define_loss_func(self):
        if self.USE_TANH:
            self.l2dist = lambda newimg, timg: torch.sum((newimg - (F.tanh(timg) + 0.5 * 1.99999) / 2).pow(2), (1, 2, 3))  # (F.tanh(timg) + 0.5 * 1.99999) / 2 range (0,1)
        else:
            self.l2dist = lambda newimg, timg: torch.sum((newimg - timg).pow(2), (1,2,3))
        # if self.ATTACK_TYPE == "targeted":
        #     self.loss1 = lambda real_val, other_val: torch.max(torch.zeros_like(other_val),
        #                                                                     torch.log(other_val + 1e-30) - torch.log(
        #                                                                         real_val + 1e-30))
        # elif self.ATTACK_TYPE == "untargeted":
        #     self.loss1 = lambda real_val, other_val: torch.max(torch.zeros_like(other_val),
        #                                                                     torch.log(real_val + 1e-30) - torch.log(
        #                                                                         other_val + 1e-30))
        self.loss2 = self.l2dist

    def get_newimg(self, timg, modifier):
        # modifier shape = (2B+1)
        self.set_img_modifier(modifier)  # set self.img_modifier
        img_modifier = self.img_modifier
        if self.USE_TANH:
            self.newimg = (F.tanh(img_modifier + timg) + 0.5 * 1.99999) / 2   # range (0,1)
        else:
            self.modifier_up = 1.0 - timg # convert from 0.5 - timg to 1 - timg, because our pixel range is (0,1)
            self.modifier_down = 0.0 - timg # convert from -0.5 to 0.0
            cond1 = torch.gt(img_modifier, self.modifier_up).float()
            cond2 = torch.le(img_modifier, self.modifier_up).float()
            cond3 = torch.gt(img_modifier, self.modifier_down).float()
            cond4 = torch.le(img_modifier, self.modifier_down).float()
            self.img_modifier = torch.mul(cond1, self.modifier_up) + torch.mul(torch.mul(cond2, cond3),
                                 self.img_modifier) + torch.mul(cond4, self.modifier_down)
            self.newimg = self.img_modifier + timg
        return self.newimg

    def loss(self, newimg, timg, true_label, target_label, const):
        with torch.no_grad():
            logits = self.model(newimg)  # B, # num_class
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

    # def real(self, logits, tlab):
    #     out = logits[torch.arange(logits.size(0)), tlab.long()]
    #     return out
    #
    # def other(self, logits, tlab):
    #     _, argsort = logits.sort(dim=1, descending=True)
    #     gt_is_max = argsort[:, 0].eq(tlab).long()
    #     second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
    #     second_max_logit = logits[torch.arange(logits.size(0)), second_max_index]  # shape = batch_size
    #     return second_max_logit


    def cw_loss(self, logit, label, target):
        logit = F.log_softmax(logit, dim=1)
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

    def print_info(self, loss, loss1, loss2, eval_costs):
        log.info(
            "[Iter] iter:{}, const:{}, cost:{}, time:{:.3f}, size:{}, loss:{:.5g}, loss1:{:.5g}, loss2:{:.10g}".format(
                self.current_iter, self.current_const, eval_costs, self.train_timer, self.real_modifier.shape,
                loss, loss1, loss2))

    @abstractmethod
    def set_img_modifier(self, modifier):
        pass

    @abstractmethod
    def get_eval_costs(self):
        pass

    @abstractmethod
    def blackbox_optimizer(self, iteration, timg, true_label, target_label, const):
        pass
    @abstractmethod
    def post_success_setting(self):
        pass


    def compare(self, x, true_label, target_label):
        if target_label is None:
            y = true_label[0].item()
        else:
            assert self.ATTACK_TYPE == "targeted"
            y = target_label[0].item()
        temp_x = np.copy(x)
        if not isinstance(x, (float, int, np.int64)):
            if self.ATTACK_TYPE == "targeted":
                temp_x[y] -= self.CONFIDENCE
                temp_x = np.argmax(temp_x)
            else:
                for i in range(len(temp_x)):
                    if i != y:
                        temp_x[i] -= self.CONFIDENCE
                temp_x = np.argmax(temp_x)
        if self.ATTACK_TYPE == "targeted":
            return temp_x == y
        else:
            return temp_x != y


    def attack(self, img, true_label, target_label):
        """
        Perform the L_2 attack on the given images for the given targets.
        # img : (B,C,H,W) 4D tensor
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """

        batch_size = img.size(0)
        query = torch.zeros(batch_size).cuda()
        with torch.no_grad():
            logit = self.model(img)
        pred = logit.argmax(dim=1)
        correct = pred.eq(true_label).float()  # shape = (batch_size,)
        not_done = correct.clone()  # shape = (batch_size,)
        # convert to tanh-space
        if self.USE_TANH:  # tf版本像素范围-0.5到0.5
            img = atanh((img - 0.5) * 1.99999)  # atanh形参数值范围(-1,1)
        # remove the extra batch dimension
        assert batch_size == 1
        # if img.dim() == 4:
        #     img = img[0]  # C,H,W
        # set the lower and upper bounds accordingly
        lower_bound = 0.0
        CONST = self.INIT_CONST
        self.current_const = CONST
        upper_bound = 1e10

        # set the upper and lower bounds for the modifier
        if not self.USE_TANH:
            img_flatten = img.view(-1)
            self.modifier_up =  torch.ones_like(img_flatten) - img_flatten
            self.modifier_down = torch.zeros_like(img_flatten) - img_flatten

        # the over all best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestattack = img
        last_loss1 = 1e10
        # inner best l2 and scores
        bestl2 = 1e10
        bestscore = -1
        timg = img
        const = CONST
        prev = 1e6
        self.train_timer = 0.0
        last_loss2 = 1e10

        # reset ADAM status
        self.mt.fill(0.0)
        self.vt.fill(0.0)
        self.adam_epoch.fill(1)
        self.real_modifier.fill(0)  # clear the modifier
        self.stage = 0
        eval_costs = 0
        o_bestscore = -1
        # np.random.seed(1234)
        attack_begin_time = time.time()
        for iteration in range(self.MAX_ITER):

            # perform the attack
            l, l2, loss1, loss2, score, nimg = self.blackbox_optimizer(iteration, timg, true_label, target_label, const)
            eval_costs += self.get_eval_costs()
            if iteration % self.PRINT_EVERY == 0:
                self.current_iter = iteration
                self.print_info(l, loss1, loss2, self.get_eval_costs())

            # reset ADAM states when a valid example has been found
            if loss1 == 0.0 and last_loss1 != 0.0 and self.stage == 0:
                # we have reached the fine tunning point
                # reset ADAM to avoid overshoot
                log.info("##### Reset ADAM #####")
                self.mt.fill(0.0)
                self.vt.fill(0.0)
                self.adam_epoch.fill(1)
                self.stage = 1
            last_loss1 = loss1
            score = score.detach().cpu().numpy()
            if l2 < bestl2 and self.compare(score, true_label, target_label):
                bestl2 = l2
                bestscore = np.argmax(score)

            if l2 < o_bestl2 and self.compare(score, true_label, target_label):
                # print a message if it is the first attack found
                if o_bestl2 == 1e10:
                    # print("save modifier")
                    log.info(
                        "[STATS][FirstAttack] iter:{}, const:{}, cost:{}, time:{:.3f}, size:{}, loss:{:.5g}, loss1:{:.5g}, loss2:{:.5g}, l2:{:.5g}".format(
                            iteration, CONST, eval_costs, self.train_timer, self.real_modifier.shape, l, loss1,
                            loss2, l2))
                    self.post_success_setting()
                    lower_bound = 0.0
                o_bestl2 = l2
                o_bestattack = nimg
                # begin statistics
                query.fill_(eval_costs)
                with torch.no_grad():
                    adv_logit = self.model(o_bestattack)
                    adv_prob = F.softmax(adv_logit, dim=1)
                adv_pred = adv_logit.argmax(dim=1)
                if self.ATTACK_TYPE == "targeted":
                    not_done = not_done * (1 - adv_pred.eq(target_label)).float()
                else:
                    not_done = not_done * adv_pred.eq(true_label).float()
                success = (1 - not_done) * correct
                success_query = success * query
                not_done_prob = adv_prob[torch.arange(batch_size), true_label] * not_done
                # end statistics
                if self.ABORT_EARLY:
                    if loss2 < self.epsilone and not not_done.byte().any():
                        log.info(
                            "Early stopping attack successfully and total pixels' distortion is {:.3f}".format(loss2))
                        break
            self.train_timer += time.time() - attack_begin_time

            # switch constant when reaching switch iterations
            if iteration % self.SWITCH_ITER == 0 and iteration != 0:
                if self.compare(bestscore, true_label, target_label) and bestscore != -1:
                    # success, divide const by two
                    log.info("iter:{} old constant:{}".format(iteration, CONST))
                    upper_bound = min(upper_bound, CONST)
                    if upper_bound < 1e9:
                        CONST = (lower_bound + upper_bound) / 2
                    log.info("iter:{} new constant:{}".format(iteration, CONST))
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    log.info("iter:{} old constant:{}".format(iteration, CONST))
                    lower_bound = max(lower_bound, CONST)
                    if upper_bound < 1e9:
                        CONST = (lower_bound + upper_bound) / 2
                    elif CONST < 1e8:
                        CONST *= 10
                    else:
                        log.info("CONST > 1e8, no change")

                    log.info("iter:{} new constant:{}".format(iteration, CONST))

                bestl2 = 1e10
                bestscore = -1
                self.current_const = CONST

                if self.stage == 1:
                    self.mt.fill(0.0)
                    self.vt.fill(0.0)
                    self.adam_epoch.fill(1)

                self.current_const = CONST
                # update constant
                timg = img
                const = CONST

        # return the best solution found
        stats_info = {"query": query, "correct":correct, "not_done": not_done,
                      "success": success, "success_query":success_query, "not_done_prob":not_done_prob,
                      "adv_logit":adv_logit.detach()}
        return o_bestattack, stats_info


class ZOO(BlackboxAttack):

    def __init__(self, model, dataset, args):
        super(ZOO, self).__init__(model, dataset, args)
        self.grad = np.zeros(self.BATCH_SIZE, dtype=np.float32)
        self.hess = np.zeros(self.BATCH_SIZE, dtype=np.float32)
        self.solver = coordinate_ADAM

    def set_img_modifier(self, modifier):
        # the shape of modifier may have decreased to smaller resolution, which can be enlarged by F.upsample_bilinear or decoder.
        if self.modifier_size == self.image_size:
            self.img_modifier =  modifier  # not resizing image or using autoencoder
        else:
            # resizing image
            self.img_modifier = F.interpolate(modifier, [self.image_size, self.image_size], mode="bilinear",
                                              align_corners=True)

    def get_eval_costs(self):
        return self.BATCH_SIZE * 2

    def blackbox_optimizer(self, iteration, timg, true_label, target_label, const):
        # argument iteration is for debugging
        # build new inputs, based on current variable value
        var = np.repeat(self.real_modifier, self.BATCH_SIZE * 2 + 1, axis=0)
        # randomly select a coordinate for each image to estimate gradient
        var_indice = np.random.choice(self.var_list.size, self.BATCH_SIZE, replace=False)
        indice = self.var_list[var_indice] # var_indice shape = (B,)  而self.var_list是[0,...,HxWxC], 相当于一个图片的分辨率，其中选择batch_size,为每个图片随机选择一个坐标，估计梯度
        for i in range(self.BATCH_SIZE):
            var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
            var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0001
        modifier = torch.from_numpy(var).cuda().float()
        newimg = self.get_newimg(timg, modifier)  # newimg shape = (2B+1,C,H,W) small shift (fake) images to estimate gradient
        losses, loss1, loss2, scores = self.loss(newimg, timg, true_label, target_label, const)
        # 下面这一步修改了real_modifier,然后这个变量增加在timg上,从而修改图像
        self.solver(losses.detach().cpu().numpy(), indice, self.grad, self.hess, self.BATCH_SIZE, self.mt, self.vt, self.real_modifier,
                    self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.USE_TANH)
        # return l, l2, loss1, loss2, score, nimg
        return losses[0].item(), loss2[0].item(), loss1[0].item(), loss2[0].item(), scores[0], newimg[0].unsqueeze(0)

    def post_success_setting(self):
        pass


class ZOO_AE(ZOO):
    def __init__(self, model, dataset, args, decoder):
        self.decoder = decoder
        super(ZOO_AE, self).__init__(model, dataset, args)

    def set_img_modifier(self, modifier):
        assert modifier.size()[1:] == self.modifier_shape
        if self.decoder.output_shape[1] == self.image_size:
            self.img_modifier = self.decoder(modifier)
        else:
            self.decoder_output = self.decoder(modifier)
            self.img_modifier = F.interpolate(self.decoder_output, [self.image_size, self.image_size], mode="bilinear",
                                              align_corners=True)

    def post_success_setting(self):
        pass


class AutoZOOM_BiLIN(BlackboxAttack):

    def __init__(self, model, dataset, args):
        super(AutoZOOM_BiLIN, self).__init__(model, dataset, args)
        self.hess = np.zeros(self.BATCH_SIZE, dtype=np.float32)
        self.solver = ADAM
        self.num_rand_vec = 1
        self.post_success_num_rand_vec = args["num_rand_vec"]
        self.grad = np.zeros((self.num_rand_vec, self.var_size), dtype=np.float32)

    def set_img_modifier(self, modifier):
        assert modifier.size()[1:] == self.modifier_shape
        if self.modifier_size == self.image_size:
            # not resizing image or using autoencoder
            self.img_modifier = modifier
        else:
            self.img_modifier = F.interpolate(modifier, [self.image_size, self.image_size],mode="bilinear",align_corners=True)

    def get_eval_costs(self):
        return self.num_rand_vec + 1

    def blackbox_optimizer(self, iteration, timg, true_label, target_label, const):
        # argument iteration is for debugging
        var_size = self.real_modifier.size
        indice = list(range(var_size))
        self.beta = 1.0 / (var_size)  # 使用beta相乘，会使值变得过小，从而loss2越变越大，暂时不知道园有
        var_noise = np.random.normal(loc=0, scale=1000, size=(self.num_rand_vec, var_size))
        # var_mean = np.mean(var_noise, axis=1, keepdims=True)
        # var_std = np.std(var_noise, axis=1, keepdims=True)
        noise_norm = np.apply_along_axis(np.linalg.norm, 1, var_noise, keepdims=True)
        var_noise = var_noise / noise_norm
        var = np.concatenate((self.real_modifier,
                              self.real_modifier + self.beta * var_noise.reshape(self.num_rand_vec, self.num_channels,
                                                                                 self.modifier_size, self.modifier_size,
                                                                                 )), axis=0)
        modifier = torch.from_numpy(var).cuda().float() # 此时修改modifier内容不会影响var或者self.real_modifier的内容
        newimg = self.get_newimg(timg, modifier)
        losses, loss1, loss2, scores = self.loss(newimg, timg, true_label, target_label, const)
        print("losses :{}, loss1:{}, loss2:{}".format(losses, loss1, loss2))
        # if (iteration + 1) % 10 == 0:
        #     time.sleep(5)
        self.solver(losses.detach().cpu().numpy(), indice, self.grad, self.hess, self.BATCH_SIZE, self.mt, self.vt, self.real_modifier,
                    self.LEARNING_RATE, self.adam_epoch, self.beta1, self.beta2, not self.USE_TANH, self.beta,
                    var_noise, self.num_rand_vec)
        return losses[0].item(), loss2[0].item(), loss1[0].item(), loss2[0].item(), scores[0], newimg[0].unsqueeze(0)

    def post_success_setting(self):
        self.num_rand_vec = self.post_success_num_rand_vec
        self.grad = np.zeros((self.num_rand_vec, self.var_size), dtype = np.float32)
        log.info("Set random vector number to :{}".format(self.num_rand_vec))

class AutoZOOM_AE(AutoZOOM_BiLIN):

    def __init__(self, model, dataset, args, decoder):
        super(AutoZOOM_AE, self).__init__(model, dataset, args)
        self.decoder = decoder
        self.num_rand_vec = 1
        self.post_success_num_rand_vec = args["num_rand_vec"]
        self.grad = np.zeros((self.num_rand_vec, self.var_size), dtype=np.float32)
        self.hess = np.zeros(self.BATCH_SIZE, dtype=np.float32)
        self.solver = ADAM

    def set_img_modifier(self, modifier):
        assert modifier.size()[1:] == self.modifier_shape
        if self.decoder.output_shape[1] == self.image_size:
            self.img_modifier = self.decoder(modifier)
        else:
            self.decoder_output = self.decoder(modifier)
            self.img_modifier = F.interpolate(self.decoder_output, [self.image_size, self.image_size], mode="bilinear",
                                              align_corners=True)

    def post_success_setting(self):
        self.num_rand_vec = self.post_success_num_rand_vec
        self.grad = np.zeros((self.num_rand_vec, self.var_size), dtype = np.float32)
        log.info("Set random vector number to :{}".format(self.num_rand_vec))

    def get_eval_costs(self):
        return self.num_rand_vec + 1