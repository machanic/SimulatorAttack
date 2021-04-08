import glob
import sys
import os
sys.path.append(os.getcwd())
import argparse
import json
import os
import os.path as osp
import random
from types import SimpleNamespace

import glog as log
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.modules import Upsample
from config import IN_CHANNELS, CLASS_NUM, PY_ROOT, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel
from dataset.defensive_model import DefensiveModel
from meta_simulator_bandits.attack.meta_model_finetune import MemoryEfficientMetaModelFinetune
from collections import deque, OrderedDict

class FinetuneQueue(object):
    def __init__(self, batch_size, meta_seq_len, img_idx_to_batch_idx):
        self.img_idx_to_batch_idx = img_idx_to_batch_idx
        self.q1_images_for_finetune = {}
        self.q2_images_for_finetune = {}
        self.q1_logits_for_finetune = {}
        self.q2_logits_for_finetune = {}
        for batch_idx in range(batch_size):
            self.q1_images_for_finetune[batch_idx] = deque(maxlen=meta_seq_len)
            self.q2_images_for_finetune[batch_idx] = deque(maxlen=meta_seq_len)
            self.q1_logits_for_finetune[batch_idx] = deque(maxlen=meta_seq_len)
            self.q2_logits_for_finetune[batch_idx] = deque(maxlen=meta_seq_len)

    def append(self, q1_images, q2_images, q1_logits, q2_logits):
        for img_idx, (q1_image, q2_image, q1_logit, q2_logit) in enumerate(zip(q1_images, q2_images, q1_logits, q2_logits)):
            batch_idx = self.img_idx_to_batch_idx[img_idx]
            self.q1_images_for_finetune[batch_idx].append(q1_image.detach().cpu())
            self.q2_images_for_finetune[batch_idx].append(q2_image.detach().cpu())
            self.q1_logits_for_finetune[batch_idx].append(q1_logit.detach().cpu())
            self.q2_logits_for_finetune[batch_idx].append(q2_logit.detach().cpu())

    def stack_history_track(self):
        q1_images = []
        q2_images = []
        q1_logits = []
        q2_logits = []
        for img_idx, batch_idx in sorted(self.img_idx_to_batch_idx.proj_dict.items(), key=lambda e:e[0]):
            q1_images.append(torch.stack(list(self.q1_images_for_finetune[batch_idx])))  # T,C,H,W
            q2_images.append(torch.stack(list(self.q2_images_for_finetune[batch_idx])))  # T,C,H,W
            q1_logits.append(torch.stack(list(self.q1_logits_for_finetune[batch_idx])))  # T, classes
            q2_logits.append(torch.stack(list(self.q2_logits_for_finetune[batch_idx])))  #  T, classes
        q1_images = torch.stack(q1_images).cuda()
        q2_images = torch.stack(q2_images).cuda()
        q1_logits = torch.stack(q1_logits).cuda()
        q2_logits = torch.stack(q2_logits).cuda()
        return q1_images, q2_images, q1_logits, q2_logits

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


# 更简单的方案，1000张图，分成100张一组的10组，每组用一个模型来跑，还可以多卡并行
class MetaSimulatorBanditsAttack(object):
    def __init__(self, args, meta_finetuner):
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)
        self.meta_finetuner = meta_finetuner

    def chunks(self, l, each_slice_len):
        each_slice_len = max(1, each_slice_len)
        return list(l[i:i + each_slice_len] for i in range(0, len(l), each_slice_len))

    def norm(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def eg_prior_step(self, x, g, lr):
        real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
        pos = real_x * torch.exp(lr * g)
        neg = (1 - real_x) * torch.exp(-lr * g)
        new_x = pos / (pos + neg)
        return new_x * 2 - 1

    def gd_prior_step(self, x, g, lr):
        return x + lr * g

    def linf_step(self, x, g, lr):
        return x + lr * torch.sign(g)

    def linf_proj_step(self, image, epsilon, adv_image):
        return image + torch.clamp(adv_image - image, -epsilon, epsilon)

    def l2_proj_step(self, image, epsilon, adv_image):
        delta = adv_image - image
        out_of_bounds_mask = (self.norm(delta) > epsilon).float()
        return out_of_bounds_mask * (image + epsilon * delta / self.norm(delta)) + (1 - out_of_bounds_mask) * adv_image

    def l2_image_step(self, x, g, lr):
        return x + lr * g / self.norm(g)


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
            # tensor = torch.cat([tensor[0:del_index], tensor[del_index + 1:]], 0)
            return_tensors.append(concatenate_tensor)
        return return_tensors

    def attack_all_images(self, args, arch, tmp_dump_path, result_dump_path):
        # subset_pos用于回调函数汇报汇总统计结果
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        # 带有缩减功能的，攻击成功的图片自动删除掉
        for data_idx, data_tuple in enumerate(self.dataset_loader):
            if os.path.exists(tmp_dump_path):
                with open(tmp_dump_path, "r") as file_obj:
                    json_content = json.load(file_obj)
                    resume_batch_idx = int(json_content["batch_idx"])  # resume
                    for key in ['query_all', 'correct_all', 'not_done_all',
                                'success_all', 'success_query_all']:
                        if key in json_content:
                            setattr(self, key, torch.from_numpy(np.asarray(json_content[key])).float())
                    if data_idx < resume_batch_idx:  # resume
                        continue

            if args.dataset == "ImageNet":
                if model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            if images.size(-1) != model.input_size[-1]:
                images = F.interpolate(images, size=model.input_size[-1], mode='bilinear',align_corners=True)
            # skip_batch_index_list = np.nonzero(np.asarray(chunk_skip_indexes[data_idx]))[0].tolist()
            selected = torch.arange(data_idx * args.batch_size,
                                    min((data_idx + 1) * args.batch_size, self.total_images))  # 选择这个batch的所有图片的index
            img_idx_to_batch_idx = ImageIdxToOrigBatchIdx(args.batch_size)

            images, true_labels = images.cuda(), true_labels.cuda()
            first_finetune = True
            finetune_queue = FinetuneQueue(args.batch_size, args.meta_seq_len, img_idx_to_batch_idx)
            prior_size = model.input_size[-1] if not args.tiling else args.tile_size
            assert args.tiling == (args.dataset == "ImageNet")
            if args.tiling:
                upsampler = Upsample(size=(model.input_size[-2], model.input_size[-1]))
            else:
                upsampler = lambda x: x
            with torch.no_grad():
                logit = model(images)
            pred = logit.argmax(dim=1)
            query = torch.zeros(images.size(0)).cuda()
            correct = pred.eq(true_labels).float()  # shape = (batch_size,)
            not_done = correct.clone()  # shape = (batch_size,)

            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                            size=target_labels[
                                                                                invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None
            prior = torch.zeros(images.size(0), IN_CHANNELS[args.dataset], prior_size, prior_size).cuda()
            prior_step = self.gd_prior_step if args.norm == 'l2' else self.eg_prior_step
            image_step = self.l2_image_step if args.norm == 'l2' else self.linf_step
            proj_step = self.l2_proj_step if args.norm == 'l2' else self.linf_proj_step  # 调用proj_maker返回的是一个函数
            criterion = self.cw_loss if args.data_loss == "cw" else self.xent_loss
            adv_images = images.clone()
            for step_index in range(1, args.max_queries + 1):
                # Create noise for exporation, estimate the gradient, and take a PGD step
                dim = prior.nelement() / images.size(0)  # nelement() --> total number of elements
                exp_noise = args.exploration * torch.randn_like(prior) / (dim ** 0.5)  # parameterizes the exploration to be done around the prior
                exp_noise = exp_noise.cuda()
                q1 = upsampler(prior + exp_noise)  # 这就是Finite Difference算法， prior相当于论文里的v，这个prior也会更新，把梯度累积上去
                q2 = upsampler(prior - exp_noise)  # prior 相当于累积的更新量，用这个更新量，再去修改image，就会变得非常准
                # Loss points for finite difference estimator
                q1_images = adv_images + args.fd_eta * q1 / self.norm(q1)
                q2_images = adv_images + args.fd_eta * q2 / self.norm(q2)
                predict_by_target_model = False
                if step_index <= args.warm_up_steps or (step_index - args.warm_up_steps) % args.meta_predict_steps == 0:
                    log.info("predict from target model")
                    predict_by_target_model = True
                    with torch.no_grad():
                        q1_logits = model(q1_images)
                        q2_logits = model(q2_images)

                        q1_logits = q1_logits/torch.norm(q1_logits,p=2,dim=-1,keepdim=True)  # 加入normalize
                        q2_logits = q2_logits/torch.norm(q2_logits,p=2,dim=-1,keepdim=True)

                    finetune_queue.append(q1_images.detach(), q2_images.detach(), q1_logits.detach(), q2_logits.detach())
                    if step_index >= args.warm_up_steps:
                        q1_images_seq, q2_images_seq, q1_logits_seq, q2_logits_seq = finetune_queue.stack_history_track()
                        finetune_times = args.finetune_times if first_finetune else random.randint(3,5)
                        log.info("begin finetune for {} times".format(finetune_times))
                        self.meta_finetuner.finetune(q1_images_seq, q2_images_seq, q1_logits_seq, q2_logits_seq,
                                                     finetune_times, first_finetune, img_idx_to_batch_idx)
                        first_finetune = False
                else:
                    with torch.no_grad():
                        q1_logits, q2_logits = self.meta_finetuner.predict(q1_images, q2_images, img_idx_to_batch_idx)

                        q1_logits = q1_logits / torch.norm(q1_logits, p=2, dim=-1, keepdim=True)
                        q2_logits = q2_logits / torch.norm(q2_logits, p=2, dim=-1, keepdim=True)

                l1 = criterion(q1_logits, true_labels, target_labels)
                l2 = criterion(q2_logits, true_labels, target_labels)
                # Finite differences estimate of directional derivative
                est_deriv = (l1 - l2) / (args.fd_eta * args.exploration)  # 方向导数 , l1和l2是loss
                # 2-query gradient estimate
                est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise  # B, C, H, W,
                # Update the prior with the estimated gradient
                prior = prior_step(prior, est_grad, args.online_lr)  # 注意，修正的是prior,这就是bandit算法的精髓
                grad = upsampler(prior)  # prior相当于梯度
                ## Update the image:
                adv_images = image_step(adv_images, grad * correct.view(-1, 1, 1, 1),  # 注意correct也是删减过的
                                        args.image_lr)  # prior放大后相当于累积的更新量，可以用来更新
                adv_images = proj_step(images, args.epsilon, adv_images)
                adv_images = torch.clamp(adv_images, 0, 1)

                with torch.no_grad():
                    adv_logit = model(adv_images)  #
                adv_pred = adv_logit.argmax(dim=1)
                adv_prob = F.softmax(adv_logit, dim=1)
                adv_loss = criterion(adv_logit, true_labels, target_labels)
                ## Continue query count
                if predict_by_target_model:
                    query = query + 2 * not_done
                if args.targeted:
                    not_done = not_done * (1 - adv_pred.eq(target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
                else:
                    not_done = not_done * adv_pred.eq(true_labels).float()  # 只要是跟原始label相等的，就还需要query，还没有成功
                success = (1 - not_done) * correct
                success_query = success * query
                not_done_loss = adv_loss * not_done
                not_done_prob = adv_prob[torch.arange(adv_images.size(0)), true_labels] * not_done

                log.info('Attacking image {} - {} / {}, step {}'.format(
                    data_idx * args.batch_size, (data_idx + 1) * args.batch_size, self.total_images, step_index
                ))
                log.info('       not_done: {:.4f}'.format(len(np.where(not_done.detach().cpu().numpy().astype(np.int32) == 1)[0]) / float(args.batch_size)))
                log.info('      fd_scalar: {:.9f}'.format((l1 - l2).mean().item()))
                if success.sum().item() > 0:
                    log.info('     mean_query: {:.4f}'.format(success_query[success.bool()].mean().item()))
                    log.info('   median_query: {:.4f}'.format(success_query[success.bool()].median().item()))
                if not_done.sum().item() > 0:
                    log.info('  not_done_loss: {:.4f}'.format(not_done_loss[not_done.bool()].mean().item()))
                    log.info('  not_done_prob: {:.4f}'.format(not_done_prob[not_done.bool()].mean().item()))

                not_done_np = not_done.detach().cpu().numpy().astype(np.int32)
                done_img_idx_list = np.where(not_done_np == 0)[0].tolist()
                delete_all = False
                if done_img_idx_list:
                    for skip_index in done_img_idx_list:  # 两次循环，第一次循环先汇报出去，第二次循环删除
                        batch_idx = img_idx_to_batch_idx[skip_index]
                        pos = selected[batch_idx].item()
                        # 先汇报被删减的值self.query_all
                        for key in ['query', 'correct', 'not_done',
                                    'success', 'success_query', 'not_done_loss', 'not_done_prob']:
                            value_all = getattr(self, key + "_all")
                            value = eval(key)[skip_index].item()
                            value_all[pos] = value

                    images, adv_images, prior, query, true_labels, target_labels, correct, not_done =\
                        self.delete_tensor_by_index_list(done_img_idx_list, images, adv_images, prior, query, true_labels, target_labels, correct, not_done)
                    img_idx_to_batch_idx.del_by_index_list(done_img_idx_list)
                    delete_all = images is None

                if delete_all:
                    break

            # report to all stats the rest unsuccess
            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', 'not_done_loss', 'not_done_prob']:
                for img_idx, batch_idx in img_idx_to_batch_idx.proj_dict.items():
                    pos = selected[batch_idx].item()
                    value_all = getattr(self, key + "_all")
                    value = eval(key)[img_idx].item()
                    value_all[pos] = value # 由于value_all是全部图片都放在一个数组里，当前batch选择出来
            img_idx_to_batch_idx.proj_dict.clear()

            tmp_info_dict = {"batch_idx": data_idx + 1, "batch_size":args.batch_size}
            for key in ['query_all', 'correct_all', 'not_done_all',
                        'success_all', 'success_query_all']:
                value_all = getattr(self, key).detach().cpu().numpy().tolist()
                tmp_info_dict[key] = value_all
            with open(tmp_dump_path, "w") as result_file_obj:
                json.dump(tmp_info_dict, result_file_obj, sort_keys=True)


        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.bool()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.bool()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.bool()].median().item(),
                          "max_query": self.success_query_all[self.success_all.bool()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_loss": self.not_done_loss_all[self.not_done_all.bool()].mean().item(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.bool()].mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))
        self.query_all.fill_(0)
        self.correct_all.fill_(0)
        self.not_done_all.fill_(0)
        self.success_all.fill_(0)
        self.success_query_all.fill_(0)
        self.not_done_loss_all.fill_(0)
        self.not_done_prob_all.fill_(0)
        model.cpu()

def get_exp_dir_name(dataset, loss, norm, targeted, target_type, distillation_loss, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.ablation_study:
        dirname = 'AblationStudy_{}@{}-cw_loss-{}-{}-mse'.format(args.study_subject, dataset,norm, target_str)
        return dirname
    if args.simulator == "vanilla_ensemble":
        if args.attack_defense:
            dirname = 'vanilla_simulator_attack_on_defensive_model-{}-{}_loss-{}-{}-{}'.format(dataset,
                                                                                                     loss, norm,
                                                                                                     target_str,
                                                                                                     distillation_loss)
        else:
            dirname = 'vanilla_simulator_attack-{}-{}_loss-{}-{}-{}'.format(dataset, loss, norm, target_str,
                                                                                  distillation_loss)
    else:
        if args.attack_defense:
            dirname = 'simulator_attack_on_defensive_model-{}-{}_loss-{}-{}-{}'.format(dataset,
                                                                                                  loss, norm, target_str, distillation_loss)
        else:
            dirname = 'simulator_attack-{}-{}_loss-{}-{}-{}'.format(dataset, loss, norm, target_str, distillation_loss)
    return dirname



def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    import subprocess
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=str, required=True)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--online-lr', type=float, help='Learning rate for the prior')
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--exploration', type=float,
                        help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile-size', type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--json-config', type=str, default='./configures/meta_simulator_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument("--notdone_threshold", type=float,default=None)
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, help='batch size for bandits attack.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random', 'least_likely',"increment"])
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument("--meta_train_type", type=str, default="2q_distillation",
                        choices=["logits_distillation", "2q_distillation"])
    parser.add_argument("--data_loss", type=str, required=True, choices=["xent", "cw"])
    parser.add_argument("--distillation_loss", type=str, required=True, choices=["mse", "pair_mse"])
    parser.add_argument("--finetune_times", type=int, default=10)
    parser.add_argument('--seed', default=1398, type=int, help='random seed')
    parser.add_argument("--meta_predict_steps", type=int, default=5)
    parser.add_argument("--warm_up_steps", type=int, default=None)
    parser.add_argument("--meta_seq_len", type=int, default=10)
    parser.add_argument("--meta_arch",type=str, required=True)
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--simulator', type=str, default="meta_simulator", choices=["meta_simulator", "vanilla_ensemble"])
    parser.add_argument('--ablation_study',action='store_true')
    parser.add_argument('--study_subject', type=str)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    gpu_num = len(args.gpu.split(","))
    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
        args_dict = defaults
    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 50000

    # bandit_expr_dir_path = osp.join(args.exp_dir, get_bandits_exp_dir_name(args.dataset, args.data_loss, args.norm, args.targeted,
    #                                                                args.target_type))

    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.data_loss,
                                                           args.norm, args.targeted, args.target_type,
                                                           args.distillation_loss, args))


    os.makedirs(args.exp_dir, exist_ok=True)

    log.info("using GPU {}".format(args.gpu))
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    if args.ablation_study:
        if args.study_subject == 'warm_up':
            key = args.warm_up_steps
        elif args.study_subject == 'meta_seq_len':
            key = args.meta_seq_len
        log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(key))

    elif args.test_archs:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}_meta_interval_{}.log'.format(args.arch, args.meta_predict_steps))
    set_log_file(log_file_path)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    meta_finetuner = MemoryEfficientMetaModelFinetune(args.dataset, args.batch_size, args.meta_arch,
                                                      args.meta_train_type,
                                                      args.distillation_loss,
                                                      args.data_loss, args.norm, args.targeted, args.simulator,
                                                      args.data_loss == "xent", without_resnet=args.attack_defense)
    attacker = MetaSimulatorBanditsAttack(args, meta_finetuner)
    for arch in archs:
        if args.ablation_study:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, key)
            tmp_result_path = args.exp_dir + "/tmp_{}_{}_result.json".format(arch, key)
        elif args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_meta_interval_{}_result.json".format(arch, args.defense_model, args.meta_predict_steps)
            tmp_result_path = args.exp_dir + "/tmp_{}_{}_meta_interval_{}_result.json".format(arch, args.defense_model, args.meta_predict_steps)
        else:
            save_result_path = args.exp_dir + "/{}_meta_interval_{}_result.json".format(arch, args.meta_predict_steps)
            tmp_result_path = args.exp_dir + "/tmp_{}_meta_interval_{}_result.json".format(arch, args.meta_predict_steps)
        # else:
        #     if args.attack_defense:
        #         save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        #         tmp_result_path = args.exp_dir + "/tmp_{}_{}_result.json".format(arch, args.defense_model)
        #     else:
        #         save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        #         tmp_result_path = args.exp_dir + "/tmp_{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        attacker.attack_all_images(args, arch,tmp_result_path, save_result_path)
        os.remove(tmp_result_path)
