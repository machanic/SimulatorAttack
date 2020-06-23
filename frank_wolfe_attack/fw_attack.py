import json
import torch
from torch import nn

from config import IMAGE_SIZE, IN_CHANNELS, CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker
import glog as log
import numpy as np
from torch.nn import functional as F

class FrankWolfeWhiteBoxAttack(object):

    def __init__(self, args, dataset, targeted, target_type, epsilon, norm, lower_bound=0.0, upper_bound=1.0,
                 max_queries=10000):
        """
            :param epsilon: perturbation limit according to lp-ball
            :param norm: norm for the lp-ball constraint
            :param lower_bound: minimum value data point can take in any coordinate
            :param upper_bound: maximum value data point can take in any coordinate
            :param max_queries: max number of calls to model per data point
            :param max_crit_queries: max number of calls to early stopping criterion  per data poinr
        """
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        self.epsilon = epsilon
        self.norm = norm
        self.max_queries = max_queries

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._proj = None
        self.is_new_batch = False
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)
        self.targeted = targeted
        self.target_type = target_type

        self.data_loader = DataLoaderMaker.get_test_attacked_data(dataset, args.batch_size)
        self.total_images = len(self.data_loader.dataset)

        self.correct_all = torch.zeros(self.total_images)  # number of images
        self.not_done_all = torch.zeros(self.total_images)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros(self.total_images)
        self.not_done_prob_all = torch.zeros(self.total_images)
        self.stop_iter_all = torch.zeros(self.total_images)
        self.att_iter = args.att_iter
        self.ord =  args.norm # linf, l1, l2
        self.clip_min = args.clip_min
        self.clip_max = args.clip_max
        self.lr = args.lr
        self.beta = args.beta
        self.loss_fn = nn.CrossEntropyLoss().cuda()

    def get_grad(self, model, inputs, targets):
        output = model(inputs)
        loss = self.loss_fn(output, targets)
        return torch.autograd.grad(loss, inputs)[0]

    def eval_image(self, model, inputs, true_labels, target_labels):
        output = model(inputs)
        pred = output.max(1)[1]
        if self.targeted:
            loss = self.loss_fn(output, target_labels)
            adv_correct = pred.eq(target_labels).long()
        else:
            loss = self.loss_fn(output, true_labels)
            adv_correct = pred.eq(true_labels).long()
        correct = pred.eq(true_labels).long()
        return loss.item(), output, correct, adv_correct

    def grad_normalization(self, gradients, order):
        if order == "linf":
            signed_grad = torch.sign(gradients)
        elif order in ["l1", "l2"]:
            reduce_indexes = list(range(1, gradients.ndimension()))
            if order == "l1":
                norm = gradients.clone().abs()
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(reduce_ind,keepdim=True)
            elif order == "l2":
                norm = gradients.clone()
                norm = torch.mul(norm, norm)
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(reduce_ind, keepdim=True)
                norm = torch.sqrt(norm)
            signed_grad = gradients / norm
        return signed_grad

    # Norm Ball Projection
    def norm_ball_proj_inner(self, eta, order, eps):
        if order == "linf":
            eta = torch.clamp(eta, -eps, eps)
        elif order in ["l1", "l2"]:
            reduce_indexes = list(range(1, len(eta.shape)))
            if order == 1:
                norm = eta.abs()
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(dim=reduce_ind, keepdim=True)
            elif order == 2:
                norm = torch.mul(eta, eta)
                for reduce_ind in reduce_indexes:
                    norm = norm.sum(dim=reduce_ind, keepdim=True)
                norm = torch.sqrt(norm)
            if norm.item() > eps:
                eta = torch.mul(eta, torch.div(eps, norm))
        return eta

    def attack_batch_images(self, model, batch_index, inputs, true_labels, target_labels):
        x = inputs.clone()
        stop_iter =torch.zeros(inputs.size(0)).cuda()
        m_t = torch.zeros_like(inputs).cuda()
        loss_init, _,  correct, adv_correct = self.eval_image(model, inputs, true_labels, target_labels)
        finished_mask = 1.0 - adv_correct if not self.targeted else adv_correct
        not_done = 1 - finished_mask
        succ_sum = torch.sum(finished_mask).item()
        log.info("Init Loss : % 5.3f, Finished: % 3d ".format(loss_init, succ_sum))
        batch_size = x.size(0)
        selected = torch.arange(batch_index * batch_size,
                                min((batch_index + 1) * batch_size, self.total_images))
        current_lr = self.lr
        for iteration in range(self.att_iter):
            if self.targeted:
                grad = self.get_grad(model, x, target_labels)
            else:
                grad = self.get_grad(model, x, true_labels)
            m_t = m_t * self.beta + grad * (1 - self.beta)
            grad_normalized = self.grad_normalization(m_t, self.ord)
            v_t = - self.epsilon * grad_normalized + inputs
            d_t = v_t - x
            new_x = x + (-1 if not self.targeted else 1) * current_lr * d_t
            new_x = inputs + self.norm_ball_proj_inner(new_x - inputs, self.ord, self.epsilon)
            new_x = torch.clamp(new_x, self.clip_min, self.clip_max)
            mask = finished_mask.view(-1, *[1]*3)
            x = new_x * (1.0 - mask) + x * mask
            stop_iter += 1 * (1. - finished_mask)
            loss, adv_logit, correct, adv_correct = self.eval_image(model, x, true_labels, target_labels)
            tmp = 1.0 - adv_correct if not self.targeted else adv_correct
            finished_mask = finished_mask.byte() | tmp.byte()
            finished_mask = finished_mask.float()
            not_done = 1.0 - finished_mask
            adv_prob = F.softmax(adv_logit, dim=1)
            success = (1 - not_done) * correct
            not_done_prob = adv_prob[torch.arange(inputs.size(0)), true_labels] * not_done
            succ_sum = finished_mask.sum().item()
            if int(succ_sum) == inputs.size(0):
                break

        for key in ['stop_iter', 'correct',  'not_done',
                    'success', 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()

        return x, stop_iter, finished_mask


    def attack_all_images(self, args, arch_name, target_model, result_dump_path):

        for batch_idx, data_tuple in enumerate(self.data_loader):
            if args.dataset == "ImageNet":
                if target_model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            if images.size(-1) != target_model.input_size[-1]:
                images = F.interpolate(images, size=target_model.input_size[-1], mode='bilinear',align_corners=True)
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                                            size=target_labels[
                                                                                invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    logit = target_model(images)
                    target_labels = logit.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None
            self.attack_batch_images(target_model, batch_idx, images.cuda(), true_labels.cuda(),target_labels.cuda())
        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.not_done_all.sum().item() > 0:
            log.info('  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "stop_iter": self.stop_iter_all[self.success_all.byte()].mean().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          "args":vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


class FrankWolfeBlackBoxAttack(FrankWolfeWhiteBoxAttack):

    def __init__(self, args, dataset, targeted, target_type, epsilon, norm,sensing_type,grad_est_batch_size, delta,
                 lower_bound=0.0, upper_bound=1.0, max_queries=10000):
        """
            :param epsilon: perturbation limit according to lp-ball
            :param norm: norm for the lp-ball constraint
            :param lower_bound: minimum value data point can take in any coordinate
            :param upper_bound: maximum value data point can take in any coordinate
            :param max_queries: max number of calls to model per data point
            :param max_crit_queries: max number of calls to early stopping criterion  per data poinr
        """
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        super(FrankWolfeBlackBoxAttack, self).__init__(args, dataset, targeted, target_type, epsilon, norm,
                                                       lower_bound, upper_bound, max_queries)
        self.epsilon = epsilon
        self.norm = norm
        self.max_queries = max_queries
        self.sensing_type = sensing_type
        self.delta = delta
        self.grad_est_batch_size = grad_est_batch_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._proj = None
        self.is_new_batch = False
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)
        self.targeted = targeted
        self.target_type = target_type
        self.single_shape = (IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][0])
        self.data_loader = DataLoaderMaker.get_test_attacked_data(dataset, args.batch_size)
        self.total_images = len(self.data_loader.dataset)

        self.correct_all = torch.zeros(self.total_images)  # number of images
        self.not_done_all = torch.zeros(self.total_images) # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros(self.total_images)
        self.success_query_all = torch.zeros(self.total_images)
        self.not_done_prob_all = torch.zeros(self.total_images)
        self.query_all = torch.zeros(self.total_images)

        self.ord =  args.norm # linf, l1, l2
        self.clip_min = args.clip_min
        self.clip_max = args.clip_max
        self.lr = args.lr
        self.beta = args.beta
        self.loss_fn = nn.CrossEntropyLoss().cuda()

    def get_grad_est(self, model, inputs, labels, num_batches):
        losses = []
        grads = []
        for _ in range(num_batches):
            noise_pos = torch.randn((self.grad_est_batch_size,)+self.single_shape)
            if self.sensing_type == 'sphere':
                reduce_indexes = list(range(1, inputs.dim()))
                noise_norm = torch.mul(noise_pos,noise_pos)
                for reduc_ind in reduce_indexes:
                    noise_norm = noise_norm.sum(reduc_ind,keepdim=True)
                noise_norm = torch.sqrt(noise_norm)
                noise_pos = noise_pos / noise_norm
                d = np.prod(self.single_shape).item()
                noise_pos = noise_pos * (d ** 0.5)
            noise = torch.cat([noise_pos, -noise_pos], dim=0).cuda()
            grad_est_imgs = inputs + self.delta * noise
            grad_est_labs = labels.repeat(self.grad_est_batch_size * 2)
            grad_est_logits = model(grad_est_imgs)
            grad_est_losses = self.loss_fn(grad_est_logits, grad_est_labs)
            grad_est_losses_tiled = grad_est_losses.view(-1,1,1,1)
            grad_estimates = torch.mean(grad_est_losses_tiled * noise, dim=0, keepdim=True)/self.delta
            final_losses =grad_est_losses
            losses.append(final_losses)
            grads.append(grad_estimates)
        return torch.mean(torch.stack(losses)), torch.mean(torch.stack(grads),dim=0)

    def attack_batch_images(self, model, batch_index, inputs, true_labels, target_labels):
        adv_images = inputs.clone()

        loss_init, example_output, correct, adv_correct = self.eval_image(model, inputs, true_labels, target_labels)
        finished_mask = 1.0 - adv_correct if not self.targeted else adv_correct
        succ_sum = torch.sum(finished_mask).item()
        batch_size = inputs.size(0)
        selected = torch.arange(batch_index * batch_size,
                                min((batch_index + 1) * batch_size, self.total_images))
        query = torch.zeros(inputs.size(0))
        if succ_sum == inputs.size(0):
            return adv_images, query, finished_mask
        adv_logits = torch.zeros_like(example_output)
        for i in range(inputs.size(0)):
            data = inputs[i:i + 1]
            true_label = true_labels[i:i + 1]
            target_label = target_labels[i:i+1]
            ori = inputs[i:i + 1].clone()
            x = data
            num_batches = 1
            m_t = torch.zeros_like(data).cuda()
            last_ls = []
            hist_len = 5
            start_decay = 0
            for iteration in range(self.max_queries // (num_batches * self.grad_est_batch_size * 2)):
                query[i] += num_batches * self.grad_est_batch_size * 2
                if query[i] > self.max_queries:
                    query[i] = self.max_queries
                    break
                # Get zeroth-order gradient estimates
                if self.targeted:
                    _, grad = self.get_grad_est(model, x, target_label, num_batches)
                else:
                    _, grad = self.get_grad_est(model, x, true_label, num_batches)
                # momentum
                m_t = m_t * self.beta + grad * (1 - self.beta)
                grad_normalized = self.grad_normalization(m_t, self.ord)
                s_t = - (-1 if not self.targeted else 1) * self.epsilon * grad_normalized + ori
                d_t = s_t - x
                current_lr = self.lr if start_decay == 0 else self.lr / (iteration - start_decay + 1) ** 0.5
                new_x = x + current_lr * d_t
                new_x = torch.clamp(new_x, self.clip_min, self.clip_max)
                x = new_x
                loss, adv_logit, _, adv_correct = self.eval_image(model, x, true_labels, target_labels)
                last_ls.append(loss)
                last_ls = last_ls[-hist_len:]
                if last_ls[-1] > 0.999 * last_ls[0] and len(last_ls) == hist_len:
                    if start_decay == 0:
                        start_decay = iteration - 1
                        print("[log] start decaying lr")
                    last_ls = []
                finished_mask[i] = 1 - adv_correct[0] if not self.targeted else adv_correct[0]
                adv_logits[i] = adv_logit
                if finished_mask[i]:
                    break
            adv_images[i] = new_x
        not_done = 1.0 - finished_mask
        adv_prob = F.softmax(adv_logits, dim=1)
        success = (1 - not_done) * correct
        not_done_prob = adv_prob[torch.arange(inputs.size(0)), true_labels] * not_done
        success_query = success * query
        for key in ['query', 'correct',  'not_done',
                    'success', "success_query", 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()

        return adv_images, query, finished_mask

    def attack_all_images(self, args, arch_name, target_model, result_dump_path):

        for batch_idx, data_tuple in enumerate(self.data_loader):
            if args.dataset == "ImageNet":
                if target_model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            if images.size(-1) != target_model.input_size[-1]:
                images = F.interpolate(images, size=target_model.input_size[-1], mode='bilinear',align_corners=True)
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                                            size=target_labels[
                                                                                invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    logit = target_model(images)
                    target_labels = logit.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None
            self.attack_batch_images(target_model, batch_idx, images.cuda(), true_labels.cuda(),target_labels.cuda())
        log.info('{} is attacked finished ({} images)!'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.not_done_all.sum().item() > 0:
            log.info('  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          "args":vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

