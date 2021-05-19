import sys
from collections import OrderedDict, defaultdict
import os
sys.path.append(os.getcwd())

import glog as log
import torch
from torch.nn import functional as F
import json
import glob
import os.path as osp
import numpy as np
from config import PY_ROOT, MODELS_TEST_STANDARD, CLASS_NUM, IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.defensive_model import DefensiveModel
from dataset.standard_model import StandardModel
import argparse
from types import SimpleNamespace

class ImageIdxToOrigBatchIdx(object):
    def __init__(self, batch_size):
        self.proj_dict = OrderedDict()
        for img_idx in range(batch_size):
            self.proj_dict[img_idx] = img_idx

    def del_by_index_list(self, del_img_idx_list):
        for del_img_idx in del_img_idx_list:
            del self.proj_dict[del_img_idx]
        all_key_value = sorted(list(self.proj_dict.items()), key=lambda e: e[0])
        for seq_idx, (img_idx, batch_idx) in enumerate(all_key_value):
            del self.proj_dict[img_idx]
            self.proj_dict[seq_idx] = batch_idx

    def __getitem__(self, img_idx):
        return self.proj_dict[img_idx]

# This code is used for different models and random direction version
class SwitchAttack(object):
    def __init__(self, dataset, batch_size, targeted, target_type, epsilon, norm, lower_bound=0.0, upper_bound=1.0,
                 max_queries=10000, stats_grad_cosine_similarity=False):
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        self.epsilon = epsilon
        self.norm = norm
        self.max_queries = max_queries
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.targeted = targeted
        self.target_type = target_type
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)
        self.cosine_similarity_all = defaultdict(OrderedDict)    # key is image index, value is {query: cosine_similarity}
        self.stats_grad_cosine_similarity = stats_grad_cosine_similarity

    def delete_tensor_by_index_list(self, del_index_list,  *tensors):
        return_tensors = []
        del_index_list = sorted(del_index_list)
        for tensor in tensors:
            if tensor is None:   # target_label may be None
                return_tensors.append(tensor)
                continue
            concatenate_tensor = []
            for i, each_tensor_element in enumerate(tensor):
                if i not in del_index_list:
                    concatenate_tensor.append(each_tensor_element)
            if len(concatenate_tensor) == 0:
                return [None for _ in tensors]  # delete all
            concatenate_tensor = torch.stack(concatenate_tensor, 0)
            return_tensors.append(concatenate_tensor)
        return return_tensors

    def xent_loss(self, logit, label, target=None):
        if target is not None:
            return -F.cross_entropy(logit, target, reduction='none')
        else:
            return F.cross_entropy(logit, label, reduction='none')

    def cw_loss(self, logit, label, target=None):
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), target]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    def normalize(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def l2_image_step(self, x, g, lr):
        return x + lr * g / self.normalize(g)

    def linf_image_step(self, x, g, lr):
        return x + lr * torch.sign(g)

    def linf_proj_step(self, image, epsilon, adv_image):
        return image + torch.clamp(adv_image - image, -epsilon, epsilon)

    def l2_proj_step(self, image, epsilon, adv_image):
        delta = adv_image - image
        out_of_bounds_mask = (self.normalize(delta) > epsilon).float()
        return out_of_bounds_mask * (image + epsilon * delta / self.normalize(delta)) + (1 - out_of_bounds_mask) * adv_image

    def get_grad(self, model, loss_fn, x, true_labels, target_labels):
        with torch.enable_grad():
            x.requires_grad_()
            logits = model(x)
            loss = loss_fn(logits, true_labels, target_labels).mean()
            gradient = torch.autograd.grad(loss, x, torch.ones_like(loss), retain_graph=False, create_graph=False)[0].detach()
        return gradient

    def get_random_grad(self, x):
        random_grad = torch.randn_like(x)
        random_grad = random_grad / self.normalize(random_grad)
        return random_grad

    def get_cos_similarity(self, grad_a, grad_b):
        grad_a = grad_a.view(grad_a.size(0),-1)
        grad_b = grad_b.view(grad_b.size(0),-1)
        cos_similarity = (grad_a * grad_b).sum(dim=1) / torch.sqrt((grad_a * grad_a).sum(dim=1) * (grad_b * grad_b).sum(dim=1))
        assert cos_similarity.size(0) == grad_a.size(0)
        return cos_similarity

    def attack_images(self, batch_index, images, true_labels, target_labels, target_model, surrogate_model, args):
        image_step = self.l2_image_step if args.norm == 'l2' else self.linf_image_step
        img_idx_to_batch_idx = ImageIdxToOrigBatchIdx(args.batch_size)
        proj_step = self.l2_proj_step if args.norm == 'l2' else self.linf_proj_step
        criterion = self.cw_loss if args.loss == "cw" else self.xent_loss
        adv_images = images.clone()
        query = torch.zeros(images.size(0)).cuda()
        # cos_similarity = torch.zeros(images.size(0), args.max_queries)
        with torch.no_grad():
            logit = target_model(images)
            l = criterion(logit, true_labels, target_labels)
        pred = logit.argmax(dim=1)
        correct = pred.eq(true_labels).float()  # shape = (batch_size,)
        not_done = correct.clone()
        query = query + not_done
        selected = torch.arange(batch_index * args.batch_size, min((batch_index + 1) * args.batch_size, self.total_images))  # 选择这个batch的所有图片的index
        step_index = 0
        while query.min().item() < args.max_queries:
            step_index += 1
            surrogate_gradients = self.get_grad(surrogate_model, criterion, adv_images, true_labels, target_labels)
            if self.stats_grad_cosine_similarity:
                no_grad = target_model.no_grad
                target_model.no_grad = False
                true_grad = self.get_grad(target_model, criterion, adv_images, true_labels, target_labels)
                cosine_similarity = self.get_cos_similarity(surrogate_gradients, true_grad)
                target_model.no_grad = no_grad

            attempt_images = image_step(adv_images, surrogate_gradients, args.image_lr)
            attempt_images = proj_step(images, args.epsilon, attempt_images)
            attempt_images = torch.clamp(attempt_images, 0, 1).detach()
            with torch.no_grad():
                attempt_logits = target_model(attempt_images)
            attempt_positive_loss = criterion(attempt_logits, true_labels, target_labels)

            attempt_images = image_step(adv_images, -surrogate_gradients, args.image_lr)
            attempt_images = proj_step(images, args.epsilon, attempt_images)
            attempt_images = torch.clamp(attempt_images, 0, 1).detach()
            with torch.no_grad():
                attempt_logits = target_model(attempt_images)
            attempt_negative_loss = criterion(attempt_logits, true_labels, target_labels)

            idx_positive_improved = (attempt_positive_loss >= l).float().view(-1,1,1,1)
            idx_negative_improved = (attempt_negative_loss >= l).float().view(-1,1,1,1)

            query = query + not_done
            query = query + (1-idx_positive_improved).view(-1) * not_done

            random_grad = self.get_random_grad(images)

            attempt_images = image_step(adv_images, random_grad, args.image_lr)
            attempt_images = proj_step(images, args.epsilon, attempt_images)
            attempt_images = torch.clamp(attempt_images, 0, 1).detach()
            with torch.no_grad():
                attempt_logits = target_model(attempt_images)
            attempt_positive_loss_random_grad = criterion(attempt_logits, true_labels, target_labels)

            attempt_images = image_step(adv_images, -random_grad, args.image_lr)
            attempt_images = proj_step(images, args.epsilon, attempt_images)
            attempt_images = torch.clamp(attempt_images, 0, 1).detach()
            with torch.no_grad():
                attempt_logits = target_model(attempt_images)
            attempt_negative_loss_random_grad = criterion(attempt_logits, true_labels, target_labels)

            idx_positive_improved_random_grad = (attempt_positive_loss_random_grad >= l).float().view(-1, 1, 1, 1)
            idx_negative_improved_random_grad = (attempt_negative_loss_random_grad >= l).float().view(-1, 1, 1, 1)

            idx_positive_larger_than_negative = (attempt_positive_loss_random_grad >= attempt_negative_loss_random_grad).float().view(-1, 1, 1, 1)
            query = query + (1 - idx_positive_improved).view(-1) * (1 - idx_negative_improved).view(-1) * not_done
            query = query + (1 - idx_positive_improved).view(-1) * (1 - idx_negative_improved).view(-1) * (1-idx_positive_improved_random_grad).view(-1) * not_done

            grad = idx_positive_improved * surrogate_gradients  + \
                   (1 - idx_positive_improved) * idx_negative_improved * (-surrogate_gradients) + \
                  (1 - idx_positive_improved) * (1 - idx_negative_improved) * idx_positive_improved_random_grad * random_grad + \
                  (1 - idx_positive_improved) * (1 - idx_negative_improved) * (1- idx_positive_improved_random_grad) * idx_negative_improved_random_grad * (-random_grad) + \
                   (1 - idx_positive_improved) * (1 - idx_negative_improved) * (1-idx_positive_improved_random_grad) * (1-idx_negative_improved_random_grad) * idx_positive_larger_than_negative*random_grad + \
                   (1 - idx_positive_improved) * (1 - idx_negative_improved) * (1 - idx_positive_improved_random_grad) *\
                   (1 - idx_negative_improved_random_grad) * (1-idx_positive_larger_than_negative) * (-random_grad)
            lr = torch.ones(adv_images.size(0)).cuda().float()
            lr[idx_positive_improved.view(-1).bool()] = args.image_lr
            lr[((1 - idx_positive_improved).view(-1) * idx_negative_improved.view(-1)).bool()] = args.image_lr
            lr[((1 - idx_positive_improved).view(-1) * (1 - idx_negative_improved).view(-1)).bool()] = args.random_lr
            lr = lr.view(-1,1,1,1)

            adv_images = image_step(adv_images, grad, lr)
            adv_images = proj_step(images, args.epsilon, adv_images)
            adv_images = torch.clamp(adv_images, 0, 1).detach()

            with torch.no_grad():
                adv_logit = target_model(adv_images)
                adv_pred = adv_logit.argmax(dim=1)
                adv_prob = F.softmax(adv_logit, dim=1)
                l = criterion(adv_logit, true_labels, target_labels)

            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels).float()
            success = (1 - not_done) * correct
            success_query = success * query
            not_done_prob = adv_prob[torch.arange(adv_images.size(0)), true_labels] * not_done

            log.info('Attacking image {} - {} / {}, step {}, max query {}'.format(
                batch_index * args.batch_size, (batch_index + 1) * args.batch_size,
                self.total_images, step_index + 1, int(query.max().item())
            ))
            step_index += 1
            log.info('        correct: {:.4f}'.format(correct.mean().item()))
            log.info('       not_done: {:.4f}'.format(float(not_done.detach().cpu().sum().item()) / float(args.batch_size)))
            if success.sum().item() > 0:
                log.info('     mean_query: {:.4f}'.format(success_query[success.bool()].mean().item()))
                log.info('   median_query: {:.4f}'.format(success_query[success.bool()].median().item()))
            if not_done.sum().item() > 0:
                log.info('  not_done_prob: {:.4f}'.format(not_done_prob[not_done.bool()].mean().item()))

            not_done_np = not_done.detach().cpu().numpy().astype(np.int32)
            done_img_idx_list = np.where(not_done_np == 0)[0].tolist()
            delete_all = False
            if done_img_idx_list:
                for skip_index in done_img_idx_list:
                    batch_idx = img_idx_to_batch_idx[skip_index]
                    pos = selected[batch_idx].item()
                    for key in ['query', 'correct', 'not_done', 'success', 'success_query', 'not_done_prob']:
                        value_all = getattr(self, key + "_all")
                        value = eval(key)[skip_index].item()
                        value_all[pos] = value
                    if self.stats_grad_cosine_similarity:
                        self.cosine_similarity_all[pos][int(query[skip_index].item())] = cosine_similarity[skip_index].item()
                images, adv_images, query, true_labels, target_labels, correct, not_done, l = \
                    self.delete_tensor_by_index_list(done_img_idx_list, images, adv_images, query,
                                                     true_labels, target_labels, correct, not_done, l)
                img_idx_to_batch_idx.del_by_index_list(done_img_idx_list)
                delete_all = images is None
            if delete_all:
                break

        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_prob']:
            for img_idx, batch_idx in img_idx_to_batch_idx.proj_dict.items():
                pos = selected[batch_idx].item()
                value_all = getattr(self, key + "_all")
                value = eval(key)[img_idx].item()
                value_all[pos] = value  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来

                if self.stats_grad_cosine_similarity:
                    assert cosine_similarity.size(0) == len(img_idx_to_batch_idx.proj_dict)
                    self.cosine_similarity_all[pos][int(query[img_idx].item())] = cosine_similarity[
                        img_idx].item()
        img_idx_to_batch_idx.proj_dict.clear()


    def eg_prior_step(self, x, g, lr):
        real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
        pos = real_x * torch.exp(lr * g)
        neg = (1 - real_x) * torch.exp(-lr * g)
        new_x = pos / (pos + neg)
        return new_x * 2 - 1

    def gd_prior_step(self, x, g, lr):
        return x + lr * g

    def l2_prior_step(self, x, g, lr):
        new_x = x + lr * g / self.norm(g)
        norm_new_x = self.norm(new_x)
        norm_mask = (norm_new_x < 1.0).float()
        return new_x * norm_mask + (1 - norm_mask) * new_x / norm_new_x

    def attack_all_images(self, args, arch_name, target_model, surrogate_model, result_dump_path):
        for batch_idx, data_tuple in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet":
                if target_model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            if images.size(-1) != target_model.input_size[-1]:
                images = F.interpolate(images, size=target_model.input_size[-1], mode='bilinear', align_corners=True)
            images = images.cuda()
            true_labels = true_labels.cuda()
            if self.targeted:
                if self.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                  size=target_labels[invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    logits = target_model(images)
                    target_labels = logits.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None

            self.attack_images(batch_idx, images, true_labels, target_labels, target_model, surrogate_model, args)
        self.not_done_all[(self.query_all > args.max_queries).bool()] = 1
        self.success_all[(self.query_all > args.max_queries).bool()] = 0
        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.info(
                '     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.bool()].mean().item()))
            log.info(
                '   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.bool()].median().item()))
            log.info('     max query: {}'.format(self.success_query_all[self.success_all.bool()].max().item()))
        if self.not_done_all.sum().item() > 0:
            log.info(
                '  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.bool()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.bool()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.bool()].mean().item() if self.success_all.sum() > 0 else 0,
                          "median_query": self.success_query_all[self.success_all.bool()].median().item() if self.success_all.sum() > 0 else 0,
                          "max_query": self.success_query_all[self.success_all.bool()].max().item() if self.success_all.sum() > 0 else 0,
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.bool()].mean().item(),
                          "args": vars(args)}
        if self.stats_grad_cosine_similarity:
            meta_info_dict['grad_cosine_similarities'] = self.cosine_similarity_all
            N = 0
            sum_cosine_similarity = 0.0
            for image_index, cos_dict in self.cosine_similarity_all.items():
                for q, cosine in cos_dict.items():
                    sum_cosine_similarity += abs(cosine)
                    N += 1
            avg_cosine_similarity = sum_cosine_similarity / N
            meta_info_dict["avg_cosine_similarity"] = avg_cosine_similarity
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def get_exp_dir_name(dataset, surrogate_arch, loss, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    method_name = "SWITCH_from_model_to_random_grad"
    if args.cosine_grad:
        method_name += "_grad_cosine_stats"
    assert surrogate_arch is not None

    if args.attack_defense:
        dirname = '{}_on_defensive_model-{}-{}-{}-loss-{}-{}'.format(method_name, surrogate_arch, dataset, loss, norm, target_str)
    else:
        dirname = '{}-{}-{}-{}-loss-{}-{}'.format(method_name, surrogate_arch, dataset, loss, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--random-lr', type=float, default=0.01)
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run attack [linf|l2]')
    parser.add_argument("--loss", type=str, required=True, choices=["xent", "cw"])
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, default=250, help='The mini-batch size')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument("--surrogate_arch", type=str, default="resnet-110", help="The architecture of surrogate model,"
                                                                                 " in original paper it is resnet152")
    parser.add_argument('--json-config', type=str,
                        default=osp.join(os.getcwd(), 'configures/SWITCH_attack_conf.json'),
                        help='a configuration file to be passed in instead of arguments')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random', 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--cosine_grad',action='store_true',help='record the cosine similarity of gradient')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"
    print("using GPU {}".format(args.gpu))
    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = {k: vars(args)[k] for k in vars(args) if vars(args)[k] is not None or k not in defaults}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
        args_dict = defaults
    if args.targeted and args.dataset == "ImageNet":
        args.max_queries = 50000

    if "CIFAR" in args.dataset:
        if args.norm == "l2":
            lr = 0.1
        else:
            if args.targeted:
                lr = 0.003
            else:
                lr = 0.01
    elif args.dataset == "TinyImageNet":
        if args.norm == "l2":
            lr = 0.2
        else:  # linf
            lr = 0.003
    elif args.dataset == "ImageNet":
        if args.norm == "l2":
            lr = 0.46
        else:  # linf
            lr = 0.003
    args.image_lr = lr

    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.surrogate_arch, args.loss, args.norm,
                                                           args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.test_archs:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)

    if args.test_archs:
        archs = []
        if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(PY_ROOT,
                                                                                        args.dataset,  arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
                else:
                    log.info(test_model_path + " does not exists!")
        elif args.dataset == "TinyImageNet":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(
                    root=PY_ROOT, dataset=args.dataset, arch=arch)
                test_model_path = list(glob.glob(test_model_list_path))
                if test_model_path and os.path.exists(test_model_path[0]):
                    archs.append(arch)
        else:
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
                    PY_ROOT,
                    args.dataset, arch)
                test_model_list_path = list(glob.glob(test_model_list_path))
                if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
                    continue
                archs.append(arch)
    else:
        assert args.arch is not None
        archs = [args.arch]
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    surrogate_model = StandardModel(args.dataset, args.surrogate_arch, False)
    surrogate_model.cuda()
    surrogate_model.eval()

    attacker = SwitchAttack(args.dataset,  args.batch_size, args.targeted, args.target_type, args.epsilon,
                            args.norm, 0.0, 1.0, args.max_queries, args.cosine_grad)
    for arch in archs:
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))

        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker.attack_all_images(args, arch,model, surrogate_model, save_result_path)
        model.cpu()
        log.info("Save result of attacking {} done".format(arch))
