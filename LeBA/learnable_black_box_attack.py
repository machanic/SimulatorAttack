from collections import defaultdict, OrderedDict
import glog as log
import os

import json
import torch
from torch.nn import functional as F

from LeBA.HOGA import TrainSurrogateModel
from config import PY_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
import numpy as np
from torch.distributions import Categorical
from torch import nn
from torch import optim

class QueryModel(object):
    '''  Query Model Class
    Args: defense_method (str): defense name,
          model (nn.module): basic victim model
    '''
    def __init__(self, model=None):
        self.model = model

    def get_query(self, out, labels):
        #return query results: score, cw loss and cross_entropy_loss
        with torch.no_grad():
            prob = F.softmax(out,dim=1)
            loss = nn.CrossEntropyLoss(reduction='none')(out, labels)
        score = prob.gather(1, labels.reshape([-1,1]))
        correct = prob.argmax(dim=1)==labels
        top2 = prob.topk(2)
        delta_score = torch.log(top2.values[:,0])-torch.log(top2.values[:,1])
        return score, delta_score, loss, correct

    def query(self, imgs, model, labels):
        # Query for no defense case
        with torch.no_grad():
            out = model(imgs)
            return self.get_query(out, labels)

    def __call__(self, imgs,  labels):
        return self.query(imgs, self.model, labels)

class LeBA(object):
    def __init__(self, dataset, model, surrogate_model, surrogate_arch, mode, pretrain_weight, epsilon, ba_num, ba_interval,
                 batch_size, targeted, target_type, pixel_epsilon, norm, lr, FL_rate, lower_bound=0.0, upper_bound=1.0,
                 max_queries=10000):
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        self.pixel_epsilon = pixel_epsilon
        self.norm = norm
        self.max_queries = max_queries
        self.epsilon = epsilon
        self.ba_num = ba_num
        self.ba_interval = ba_interval
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.targeted = targeted
        self.target_type = target_type
        self.surrogate_model = surrogate_model
        self.optimizer = optim.SGD(surrogate_model.parameters(), lr=lr, momentum=0.9)
        self.dataset_loader = DataLoaderMaker.get_imageid_test_attacked_data(dataset, batch_size)

        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.counts_all = torch.zeros_like(self.query_all)  # number of images
        self.correct_all = torch.zeros_like(self.query_all)
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.l2_distortion_all = torch.zeros_like(self.query_all).float()
        self.train_model_s = TrainSurrogateModel(FL_rate)
        self.query = QueryModel(model)
        self.query2 = QueryModel(surrogate_model).query
        self.surrogate_model_path = "{}/train_pytorch_model/LeBA/{}_{}.pth.tar".format(PY_ROOT, dataset, surrogate_arch)
        os.makedirs(os.path.dirname(self.surrogate_model_path),exist_ok=True)
        self.if_train = False
        self.with_TIMI = True
        self.with_s_prior = True
        if mode == 'train':  # LeBA
            self.if_train = True
            self.minibatch = 8
        elif mode == 'test':  # LeBA test mode
            self.minibatch = 8
            if pretrain_weight is None:
                self.pretrain_weight = "this_weight"
        elif mode == 'SimBA++':  # SimBA++
            self.minibatch =  8
            self.pretrain_weight = ''
        elif mode == 'SimBA+':
            self.minibatch = 8
            self.pretrain_weight = ''
            self.with_TIMI = False
        elif mode == 'SimBA':
            self.minibatch = 16
            self.with_TIMI = False
            self.with_s_prior = False
        if batch_size != 0:
            self.minibatch = batch_size


    def get_data(self, data_iter, num):
        # Get Data from data_loader
        # Args:
        #	num: get data number.
        filenames = []
        imgs = []
        labels = []
        for i in range(num):
            try:
                image_id, image, label = next(data_iter)
                imgs.append(image.cuda())
                labels.append(label.cuda())
                filename = str(image_id)  #  这个image_id是唯一的（1000个图里面唯一）
                filenames.append(filename)
                # data_end=False
            except:
                log.info("Data Iterater finished")
                break
        return imgs, labels, filenames

    def gkern(self, kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        import scipy.stats as st
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel.astype(np.float32)

    def gauss_conv(self, img, k_size):
        kernel = self.gkern(k_size, 3).astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        stack_kernel = torch.Tensor(stack_kernel).cuda()
        out = F.conv2d(img, stack_kernel, padding=(k_size - 1) // 2, groups=3)
        return out

    def distance(self, imgs1, imgs2=None, norm=2):
        # Compute L2 or L_inf distance between imgs1 and imgs2
        if imgs1.dim() == 3:
            imgs1 = imgs1.unsqueeze(0)
            imgs2 = imgs2.unsqueeze(0)
        img_num = imgs1.size(0)
        if imgs2 is None:
            if norm == 2:
                distance = imgs1.view([img_num, -1]).norm(2, dim=1)
                return distance
        if norm == 2:
            try:
                distance = (imgs1.view([img_num, -1]) - imgs2.view([img_num, -1])).norm(2, dim=1)
            except Exception:
                log.info("{} {} {}".format(img_num, imgs1.shape, imgs2.shape))
        elif norm == 'linf':
            distance = (imgs1.view([img_num, -1]) - imgs2.view([img_num, -1])).norm(float('inf'), dim=1)
        return distance

    def update_img(self, imgs, raw_imgs, diff, max_distance):
        # update imgs: clip(imgs+diff), clip new_imgs to constrain the noise within max_distace
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
            raw_imgs = raw_imgs.unsqueeze(0)
            diff = diff.unsqueeze(0)
        diff_norm = self.distance(torch.clamp(imgs + diff, 0, 1), raw_imgs)
        factor = (max_distance / diff_norm).clamp(0, 1.0).view(-1, 1, 1, 1)
        adv_diff = (torch.clamp(imgs + diff, 0, 1) - raw_imgs) * factor
        adv_imgs = torch.clamp(raw_imgs + adv_diff, 0, 1)
        return adv_imgs

    def normalize(self, input_):
        return input_ / input_.view([input_.shape[0], -1]).pow(2).mean(-1).sqrt().view(-1, 1, 1, 1).clamp(1e-12, 1e6)

    def update_slice(self, value, slice1, slice2, target):
        temp = value[slice1]
        temp[slice2] = target
        value[slice1] = temp

    def get_gauss_diff(self, shape, select, k_size, epsilon=1.0):
        diff = torch.zeros([shape[0], shape[1], shape[2] + k_size - 1, shape[3] + k_size - 1])  # .to(device)
        diff_kernel = torch.zeros([shape[0], k_size, k_size])
        for i in range(shape[0]):
            gauss_kernel = torch.tensor(self.gkern(k_size, 3)) * epsilon  # .to(device)
            diff_kernel[i] = gauss_kernel + torch.randn(gauss_kernel.shape) * gauss_kernel * 0.1
            diff[i, select[i, 0], select[i, 1]:select[i, 1] + k_size, select[i, 2]:select[i, 2] + k_size] += \
            diff_kernel[i]
        if k_size != 1:
            diff = diff[:, :, k_size // 2:-(k_size // 2), k_size // 2:-(k_size // 2)]
        return diff, diff_kernel

    def get_diff_gauss(self, selects, shape, reference, k_size):
        # Return Gaussian diff
        diff, diff_kernel = self.get_gauss_diff(shape, selects[:, 0, :], k_size)
        diff = diff.cuda()
        for i in range(diff.shape[0]):
            diff[i] = diff[i] / diff[i].max()
            diff[i] *= reference[i]
            diff_kernel[i] = diff_kernel[i] / diff_kernel[i].max()
            diff_kernel[i] *= reference[i]
        return diff, diff_kernel

    def sample_byprob(self, probs, shape):
        # Sample one pixel per image according to probs
        with torch.no_grad():
            m = Categorical(probs)
            select = m.sample()
            c = select // (shape[2] * shape[3])
            w = select % shape[3]
            h = (select - c * shape[2] * shape[3]) // shape[3]
            select = torch.stack([c, h, w]).transpose(1, 0).long()
        return select

    def select_points(self, mode='by_prob', probs=None, select_num=1):
        # Args: mode: 'by_prob': select pixel by prob map
        #		 	or 'max': select top k prob pixel
        # Sample Multi pixels.
        shape = probs.shape
        if mode == 'by_prob':
            probs = probs.reshape([probs.shape[0], -1])
            selects = []
            for n in range(select_num):
                select = self.sample_byprob(probs, shape)
                selects.append(select)
            selects = torch.stack(selects).permute(1, 0, 2)
        elif mode == 'max':
            probs = probs.reshape([probs.shape[0], -1])
            a, select = torch.topk(probs, select_num, dim=-1)
            c = select / (shape[2] * shape[3])
            w = select % shape[3]
            h = (select - c * shape[2] * shape[3]) // shape[3]
            selects = torch.stack([c, h, w]).permute([1, 2, 0]).long()
        return selects



    def get_trans_advimg(self, imgs, surrogate_model, labels, raw_imgs, ba_num):
        # TIMI for following iterations in LeBA, similar to attack_black function, but it won't query victim model during iteration
        # Args: ba_num: iteration num in TIMI
        adv_img = imgs.detach().clone()
        adv_img.requires_grad = True
        diff = 0
        momentum = 0.9
        # epsilon = self.epsilon / 16.37 # FIXME
        epsilon = 1.0 # FIXME
        max_distance = self.epsilon
        img_num = imgs.shape[0]

        def proj(img, diff):
            return self.update_img(img, raw_imgs, diff, max_distance)

        for i in range(ba_num):
            out = surrogate_model(adv_img)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            loss = nn.CrossEntropyLoss()(out, labels)
            loss.backward()
            grad = adv_img.grad
            grad = self.gauss_conv(grad, 9)
            diff_norm = (diff * momentum + grad).view(img_num, -1).norm(2, dim=1).clamp(1e-8, 1e8).view(
                img_num, 1, 1, 1)
            diff = epsilon * (diff * momentum + grad) / diff_norm
            adv_img = proj(adv_img.data, diff)
            adv_img.grad.zero_()
            surrogate_model.zero_grad()
        adv_img.requires_grad = False
        return adv_img.detach()

    def index_(self, list1, index):
        new_list = []
        for i in range(index.shape[0]):
            if index[i].data is True:
                new_list.append(list1[i])
        return new_list

    def before_query_iter(self, imgs, labels, model2, with_TIMI, with_s_prior):
        # First iteration in LeBA
        raw_imgs = imgs.clone()
        # First query victim model.
        # Get last_score, last_query(cw_loss:delta log score for simbda) and last_loss(cross entropy loss for TIMI),
        # correct:correctly classified: Not correct = Success
        last_score, last_query, last_loss, correct = self.query(imgs,  labels)
        _, a, b, correct_s = self.query2(imgs, model2, labels)

        log.info("Init correct rate, model {}, model_s {}".format(correct.float().mean().item(), correct_s.float().mean().item()))
        img_num = imgs.shape[0]
        counts = torch.ones([img_num]).cuda()
        end_type = torch.zeros([img_num]).cuda()
        prior_prob = torch.ones(imgs.shape).cuda()
        if correct.sum().item() > 0:
            # RUN TIMI, and update counts, correct, last_query status
            if with_s_prior:
                best_advimg, adv_img = self.attack_black(imgs, labels, model2, counts,
                                                    correct, last_query)
                # Update prior prob according to  accumulative gradient in TIMI, accumulative gradient is more stable.
                prior_prob = (best_advimg - raw_imgs).abs().clamp(1e-6, 1e6)  # 修改： best_advimg to adv_img
                prior_norm = prior_prob.view(img_num, -1).norm(2, dim=1).clamp(1e-12, 1e12).view(img_num, 1, 1, 1)
                prior_prob = prior_prob / prior_norm
            if with_TIMI and with_s_prior:
                imgs = best_advimg
        last_score, last_query, last_loss, correct = self.query(imgs,  labels)
        counts += correct.float()
        end_type[~correct] = 1
        return imgs, counts, last_score, last_query, last_loss, correct, prior_prob, end_type

    def attack_black(self, images, labels, surrogate_model, counts, correct, last_query):
        ''' Black-box attack TIMI, run in the first iteration in LeBA Attack
            Args: counts, correct, last_query: Init records
        '''
        raw_imgs = images
        adv_img = images.clone()
        adv_img.requires_grad = True
        diff = 0
        momentum = 0.9
        # epsilon = self.epsilon / 16.37 # FIXME
        epsilon = 1.0 # FIXME
        max_distance = self.epsilon
        img_num = images.shape[0]
        best_advimg = images.clone()

        def proj(imgs, diff, index):
            return self.update_img(imgs, raw_imgs[index], diff, max_distance)

        for it in range(10):
            out = surrogate_model(adv_img)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            loss = nn.CrossEntropyLoss()(out, labels)
            loss.backward()
            grad = adv_img.grad.data
            grad = self.gauss_conv(grad, 9)
            diff_norm = (diff * momentum + grad).view(img_num, -1).norm(2, dim=1).clamp(1e-12, 1e12).reshape(
                [img_num, 1, 1, 1])
            diff = epsilon * (diff * momentum + grad) / diff_norm
            adv_img[correct] = proj(adv_img[correct], diff[correct], correct)
            adv_img.grad.zero_()
            surrogate_model.zero_grad()
            if it > 2 and it % 1 == 0:  # TIMI in first iteration will query model during iterations,
                # it will early stop some query success sample, and won't update some no improve perturbation
                c1 = correct.clone()
                score1, q1, loss1, c1[correct] = self.query(adv_img[correct],  labels[correct])
                counts[correct] += 1
                update_index = (q1 < last_query[correct]).view(-1) | (~c1[correct])
                self.update_slice(last_query, correct, update_index, q1[update_index])
                self.update_slice(best_advimg, correct, update_index, adv_img[correct][update_index])
                correct *= c1
                if correct.sum().item() == 0:
                    break
        adv_img = adv_img.detach()
        adv_img.requires_grad = False
        log.info('black_attack,distance: {}'.format(self.distance(images, best_advimg)))
        return best_advimg, adv_img

    def save_result(self, result_dump_path, args):
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.bool()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.bool()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.bool()].median().item(),
                          "max_query": self.success_query_all[self.success_all.bool()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

    def attack(self):
        '''
        Main function to run LeBA algorithm.
        We use batch for attack, and to accelerate speed, we introduce pipeline attack
        Pipeline attack means if one image has been breached, we add a new image to attack.
        Args:
            target_model: victim model其实没有传入，但是放在query里了
            surrogate_model: surrogate model
            data_loader: iterator return data
            if_train: Flag of if train surrogate model, if 'if_train' off, function degrade to SimBA++
        '''
        surrogate_model = self.surrogate_model
        data_loader = self.dataset_loader
        data_iter = iter(data_loader)
        img_nums = len(data_loader.dataset)
        optimizer = self.optimizer
        if_train = self.if_train
        with_TIMI = self.with_TIMI
        with_s_prior = self.with_s_prior
        minibatch = self.minibatch

        correct_all = torch.ones([img_nums]).bool().cuda()  # record all correct(not success) flag
        counts_all = torch.zeros([img_nums]).cuda()   # Record all query numbers
        end_type_all = torch.zeros([img_nums]).cuda().float()  # for debug
        L2_all = torch.zeros([img_nums]).cuda()  # Record final perturbation amount
        it = 0
        img_id = 0
        indices = torch.zeros([img_nums]).bool().cuda()  # Record indices of all has been attacked images
        indices[:minibatch] = True

        correct = torch.zeros([minibatch]).bool().cuda()  # Minibatch correct(not success) flag
        counts = torch.zeros([minibatch]).cuda() # Record minibatch query numbers
        end_type = torch.zeros([minibatch]).cuda().float()  # for debug
        max_query = self.max_queries  # max query budget
        pixel_epsilon = self.pixel_epsilon  # epsilon for SimBA part
        max_distance = self.epsilon  # Max perturb budget  (L2 distance)
        b_num = 0
        get_new_flag = False

        def proj(imgs, diff, raw_imgs):  # Clip function
            return self.update_img(imgs, raw_imgs, diff, max_distance)

        while True:
            it += 1
            if it % 50 == 1 or get_new_flag:  # Per 50 iteration, add new input data, and save success samples.
                get_new_flag = False
                b_num += 1
                if b_num != 1:
                    L2 = self.distance(imgs, raw_imgs)
                    end_type_all[indices] = end_type
                    L2_all[indices] = L2
                    for i in range(len(imgs)):
                        if correct[i] is False or counts[i] > max_query:  # 攻击成功或者超过一万次query
                            log.info("{}-th image ".format(filenames[i])+' Success: {}'.format(~correct[i])+' counts:{}, L2:{:.4f}, end_type:{}'.format(counts[i], L2[i], end_type[i]))
                            pos = filenames[i].item()
                            # 先汇报被删减的值self.query_all
                            self.query_all[pos] = counts[i].item()
                            self.not_done_all[pos] = correct[i].item()
                            self.success_all[pos] = ~(correct[i]).item()
                            self.success_query_all[pos] =self.success_all[pos] * self.query_all[pos]
                            self.correct_all[pos] = correct[i].item() # 无法区分原本就分类失败的和攻击成功的
                            self.l2_distortion_all[pos] = L2[i].item()
                            correct[i] = False
                correct_all[indices] = correct  # correct_all被用来当指示剩下的mask了，所以不再代表是否成功
                counts_all[indices] = counts

                if img_id == img_nums and correct.sum().item() == 0 and get_new_flag is False:  # Attack finish
                    break
                if correct.sum().item() < minibatch:
                    indices *= correct_all
                    new_imgs, new_labels, new_filenames = self.get_data(data_iter, minibatch - correct.sum().item())  # Get new data to attack
                    get_new = (new_labels != [])  # New attack is available
                    if get_new:
                        new_labels = torch.cat(new_labels)
                        indices[img_id:img_id + new_labels.shape[0]] = True
                        img_id += new_labels.shape[0]
                        new_raw_imgs = torch.cat(new_imgs).clone()
                        # Run TIMI first
                        # Get new_imgs and several update properties
                        new_imgs, counts0, last_score0, last_query0, last_loss0, correct0, prior_prob0, end_type0 = \
                            self.before_query_iter(torch.cat(new_imgs), new_labels, surrogate_model,
                                              with_TIMI, with_s_prior)
                        last_improve0 = torch.zeros([new_imgs.shape[0]]).cuda()

                    if b_num == 1:
                        correct = correct0
                    # Update all the propertities in pipeline
                    last_score = last_score0 if b_num == 1 else torch.cat(
                        [last_score[correct], last_score0]) if get_new else last_score[correct]
                    last_query = last_query0 if b_num == 1 else torch.cat(
                        [last_query[correct], last_query0]) if get_new else last_query[correct]
                    last_loss = last_loss0 if b_num == 1 else torch.cat(
                        [last_loss[correct], last_loss0]) if get_new else last_loss[correct]
                    imgs = new_imgs if b_num == 1 else torch.cat([imgs[correct], new_imgs]) if get_new else imgs[
                        correct]
                    raw_imgs = new_raw_imgs if b_num == 1 else torch.cat(
                        [raw_imgs[correct], new_raw_imgs]) if get_new else raw_imgs[correct]

                    filenames = new_filenames if b_num == 1 else self.index_(filenames, correct) + new_filenames if get_new else self.index_(
                        filenames, correct)
                    labels = new_labels if b_num == 1 else torch.cat([labels[correct], new_labels]) if get_new else \
                    labels[correct]
                    prior_prob = prior_prob0 if b_num == 1 else torch.cat(
                        [prior_prob[correct], prior_prob0]) if get_new else prior_prob[correct]
                    counts = counts0 if b_num == 1 else torch.cat([counts[correct], counts0]).cuda() if get_new else \
                    counts[correct]
                    end_type = end_type0 if b_num == 1 else torch.cat([end_type[correct], end_type0]).cuda() if get_new else end_type[correct]
                    last_improve = last_improve0 if b_num == 1 else torch.cat(
                        [last_improve[correct], last_improve0]).cuda() if get_new else last_improve[correct]
                    correct = correct0 if b_num == 1 else torch.cat([correct[correct], correct0]).cuda() if get_new else correct[correct]
                    log.info("Init last_query: {}".format(last_query))
                    log.info(filenames)

            if it % self.ba_interval == (self.ba_interval - 1) and with_s_prior:
                # Run TIMI
                adv_imgs = self.get_trans_advimg(imgs[correct], surrogate_model, labels[correct], raw_imgs[correct], self.ba_num)
                score3, d_score3, loss3, c3 = self.query(adv_imgs,  labels[correct])
                # Update prior_prob
                prior_prob[correct] = self.normalize((adv_imgs - raw_imgs[correct]).abs().clamp(1e-6,
                                                                                                1e6))  # + torch.rand(imgs[correct].shape).to(device)*0.2
                update_index = (d_score3 < last_query[correct]) | (~c3)  # | ((last_query[correct]==1.0) & (last_improve[correct]>=80))
                # If TIMI attack improve query result(cw_loss: delta log score), update images and properties.
                if update_index.sum().item() > 0:
                    if with_TIMI:
                        self.update_slice(imgs, correct, update_index, adv_imgs[update_index])
                        self.update_slice(last_score, correct, update_index, score3[update_index])
                        self.update_slice(last_query, correct, update_index, d_score3[update_index])
                        self.update_slice(last_loss, correct, update_index, loss3[update_index])
                counts += correct.float()  # update counts record
                correct[correct] *= c3  # update correct flags
                end_type[(end_type == 0) * (~correct)] = 2
                if correct.sum() == 0:
                    get_new_flag = True
                    continue
            if it % 10 == 0:  # log
                L2 = self.distance(imgs, raw_imgs)
                log.info('It {}, Query: {}, d_score: {}, loss1: {},  correct: {}, L2: {}'.format(
                it, counts.mean().item(), last_query.mean().item(), last_loss.mean().item(), correct.float().mean().item(), L2.mean().item()))
                logs_str = "Counts: "
                logs_L2 = "L2: "
                logs_score = "score: "
                logs_loss = "loss: "
                for i in range(imgs.shape[0]):
                    logs_str += "{}, ".format(counts[i])
                    logs_L2 += "{:.3f}, ".format(L2[i])
                    logs_score += "{:.3f}, ".format(last_query[i])
                    logs_loss += "{:.3f}, ".format(last_loss[i])
                log.info(logs_str)
                log.info(logs_L2)
                log.info(logs_score)

            # Run SimBA+:
            reference = torch.ones(imgs.shape[0]) * pixel_epsilon
            if not with_s_prior:
                prior_prob = torch.ones(imgs.shape).cuda()
            selects = self.select_points(mode='by_prob', probs=prior_prob, select_num=1)  # Select point according to prior prob got by TIMI.
            k_size = int((25 + 1) // 2 * 2 + 1)
            diff, diff_kernel = self.get_diff_gauss(selects, imgs.shape, reference, k_size=k_size)  # Add gaussian noise on select pixel.

            c1 = correct.clone()
            adv_imgs = proj(imgs[correct], diff[correct], raw_imgs[correct])
            score1, d_score1, loss1, c1[correct] = self.query(adv_imgs, labels[correct])  # Query model1 with +diff noise
            update_index = (d_score1 < last_query[correct]) | (~c1[correct])
            if if_train:  # Use query information to train surrogate model (HOGA)
                self.train_model_s(self.index_(filenames, correct), imgs[correct], surrogate_model, labels[correct],
                              adv_imgs - imgs[correct], score1, loss1, last_loss[correct], optimizer)

            last_improve[correct] += 1
            # If query result improve update imgs and properties
            self.update_slice(imgs, correct, update_index, adv_imgs[update_index])
            self.update_slice(last_score, correct, update_index, score1[update_index])
            self.update_slice(last_query, correct, update_index, d_score1[update_index])
            self.update_slice(last_loss, correct, update_index, loss1[update_index])
            self.update_slice(last_improve, correct, update_index, 0)
            counts += correct.float()
            # record not correct and not update with +diff indices
            remain = correct.clone()
            self.update_slice(remain, correct, update_index, False)
            correct *= c1
            end_type[(end_type == 0) * (~correct)] = 3
            if correct.sum() == 0:
                get_new_flag = True
                continue
            if remain.sum() > 0:  # For not correct and not update with +diff samples
                c2 = correct.clone()
                adv_imgs = proj(imgs[remain], -diff[remain], raw_imgs[remain])  # Query model1 with -diff noise
                score2, d_score2, loss2, c2[remain] = self.query(adv_imgs,  labels[remain])
                if if_train:  # HOGA
                    self.train_model_s(self.index_(filenames, remain), imgs[remain], surrogate_model, labels[remain],
                                  adv_imgs - imgs[remain], score2, loss2, last_loss[remain], optimizer)
                counts += remain.float()
                update_index2 = (d_score2 < last_query[remain]) | (~c2[remain])

                # If query result improve update imgs and properties
                last_improve[remain] += 1
                self.update_slice(imgs, remain, update_index2, adv_imgs[update_index2])
                self.update_slice(last_score, remain, update_index2, score2[update_index2])
                self.update_slice(last_query, remain, update_index2, d_score2[update_index2])
                self.update_slice(last_loss, remain, update_index2, loss2[update_index2])
                self.update_slice(last_improve, remain, update_index2, 0)
                correct *= c2
                end_type[(end_type == 0) * (~correct)] = 3
            # score, d_score, loss, c = query(imgs,  labels)

            if correct.sum().item() == 0:
                get_new_flag = True
                continue

        if if_train:  # Save train weight of surrogate model
            torch.save(surrogate_model.state_dict(), self.surrogate_model_path)
        return counts_all, correct_all, end_type_all, L2_all