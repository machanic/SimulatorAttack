import argparse
import random
import sys
import os
sys.path.append(os.getcwd())
import json
import os
from types import SimpleNamespace
import os.path as osp
import numpy as np
from torch.nn.modules import Upsample
from cifar_models_myself import *
from config import IN_CHANNELS, IMAGE_SIZE, CLASS_NUM, PY_ROOT, MODELS_TRAIN_STANDARD
import glog as log
from dataset.standard_model import StandardModel
from collections import  deque
from dataset.dataset_loader_maker import DataLoaderMaker

class BanditAttack(object):
    @staticmethod
    def norm(t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    ###
    # Below is different optimization steps
    # All take the form of func(x, g, lr)
    # eg: exponentiated gradients
    # l2/linf: projected gradient descent
    @staticmethod
    def eg_step(x, g, lr):
        real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
        pos = real_x * torch.exp(lr * g)
        neg = (1 - real_x) * torch.exp(-lr * g)
        new_x = pos / (pos + neg)
        return new_x * 2 - 1

    @staticmethod
    def linf_step(x, g, lr):
        return x + lr * torch.sign(g)

    @staticmethod
    def l2_prior_step(x, g, lr):
        new_x = x + lr * g / BanditAttack.norm(g)
        norm_new_x = BanditAttack.norm(new_x)
        norm_mask = (norm_new_x < 1.0).float()
        return new_x * norm_mask + (1 - norm_mask) * new_x / norm_new_x

    @staticmethod
    def gd_prior_step(x, g, lr):
        return x + lr * g

    @staticmethod
    def l2_image_step(x, g, lr):
        return x + lr * g / BanditAttack.norm(g)

    ##
    # Projection steps for l2 and linf constraints:
    # All take the form of func(new_x, old_x, epsilon)
    ##
    @staticmethod
    def l2_proj(image, eps):
        orig = image.clone()
        def proj(new_x):
            delta = new_x - orig
            out_of_bounds_mask = (BanditAttack.norm(delta) > eps).float()
            x = (orig + eps * delta / BanditAttack.norm(delta)) * out_of_bounds_mask
            x += new_x * (1 - out_of_bounds_mask)
            return x
        return proj

    @staticmethod
    def linf_proj(image, eps):
        orig = image.clone()
        def proj(new_x):
            return orig + torch.clamp(new_x - orig, -eps, eps)
        return proj

    @staticmethod
    def cw_loss(logit, label, target=None):
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

    @staticmethod
    def xent_loss(logit, label, target=None):
        if target is not None:
            return -F.cross_entropy(logit, target, reduction='none')
        else:
            return F.cross_entropy(logit, label, reduction='none')

    @classmethod
    def make_adversarial_examples(cls, image, true_label, target_label, args, attack_norm, model_to_fool):
        '''
        The attack process for generating adversarial examples with priors.
        '''
        # Initial setup
        orig_images = image.clone()
        prior_size = IMAGE_SIZE[args.dataset][0] if not args.tiling else args.tile_size
        assert args.tiling == (args.dataset == "ImageNet")
        if args.tiling:
            upsampler = Upsample(size=(IMAGE_SIZE[args.dataset][0], IMAGE_SIZE[args.dataset][1]))
        else:
            upsampler = lambda x: x
        total_queries = torch.zeros(args.batch_size).cuda()
        prior = torch.zeros(args.batch_size, IN_CHANNELS[args.dataset], prior_size, prior_size).cuda()
        dim = prior.nelement() / args.batch_size  # nelement() --> total number of elements
        prior_step = BanditAttack.gd_prior_step if attack_norm == 'l2' else BanditAttack.eg_step
        image_step = BanditAttack.l2_image_step if attack_norm == 'l2' else BanditAttack.linf_step
        proj_maker = BanditAttack.l2_proj if attack_norm == 'l2' else BanditAttack.linf_proj  # 调用proj_maker返回的是一个函数
        proj_step = proj_maker(orig_images, args.epsilon)
        # Loss function
        criterion = BanditAttack.cw_loss if args.loss == "cw" else BanditAttack.xent_loss
        # Original classifications
        orig_classes = model_to_fool(image).argmax(1).cuda()
        correct_classified_mask = (orig_classes == true_label).float()
        not_dones_mask = correct_classified_mask.clone()  # 分类分对的mask
        log.info("correct ratio : {:.3f}".format(correct_classified_mask.mean()))
        normalized_q1 = deque(maxlen=100)
        normalized_q2 = deque(maxlen=100)
        images = deque(maxlen=100)
        logits_q1_list = deque(maxlen=100)
        logits_q2_list = deque(maxlen=100)

        # 有选择的选择一个段落，比如说从中间开始截取一个段落
        assert args.max_queries//2 >= 100
        slice_iteration_end = random.randint(100, args.max_queries//2)
        for i in range(slice_iteration_end):
            if not args.nes:
                ## Updating the prior:
                # Create noise for exporation, estimate the gradient, and take a PGD step
                exp_noise = args.exploration * torch.randn_like(prior) / (dim ** 0.5)  # parameterizes the exploration to be done around the prior
                exp_noise = exp_noise.cuda()
                # Query deltas for finite difference estimator
                q1 = upsampler(prior + exp_noise)  # 这就是Finite Difference算法， prior相当于论文里的v，这个prior也会更新，把梯度累积上去
                q2 = upsampler(prior - exp_noise)   # prior 相当于累积的更新量，用这个更新量，再去修改image，就会变得非常准
                # Loss points for finite difference estimator
                logits_q1 = model_to_fool(image + args.fd_eta * q1 / BanditAttack.norm(q1))
                logits_q2 = model_to_fool(image + args.fd_eta * q2 / BanditAttack.norm(q2))
                l1 = criterion(logits_q1, true_label, target_label)
                l2 = criterion(logits_q2, true_label, target_label)
                if i >= slice_iteration_end - 100:
                    images.append(image.detach().cpu().numpy())
                    normalized_q1.append((args.fd_eta * q1 / BanditAttack.norm(q1)).detach().cpu().numpy())
                    normalized_q2.append((args.fd_eta * q2 / BanditAttack.norm(q2)).detach().cpu().numpy())
                    logits_q1_list.append(logits_q1.detach().cpu().numpy())
                    logits_q2_list.append(logits_q2.detach().cpu().numpy())

                # Finite differences estimate of directional derivative
                est_deriv = (l1 - l2) / (args.fd_eta * args.exploration)  # 方向导数 , l1和l2是loss
                # 2-query gradient estimate
                est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise  # B, C, H, W,
                # Update the prior with the estimated gradient
                prior = prior_step(prior, est_grad, args.online_lr)  # 注意，修正的是prior,这就是bandit算法的精髓
            else:  # NES方法
                prior = torch.zeros_like(image).cuda()
                for grad_iter_t in range(args.gradient_iters):
                    exp_noise = torch.randn_like(image) / (dim ** 0.5)
                    logits_q1 = model_to_fool(image + args.fd_eta * exp_noise)
                    logits_q2 = model_to_fool(image - args.fd_eta * exp_noise)
                    l1 = criterion(logits_q1, true_label, target_label)
                    l2 = criterion(logits_q2, true_label, target_label)
                    est_deriv = (l1-l2) / args.fd_eta
                    prior += est_deriv.view(-1, 1, 1, 1) * exp_noise
                    if i* args.gradient_iters + grad_iter_t >= slice_iteration_end - 100:
                        images.append(image.detach().cpu().numpy())
                        normalized_q1.append((args.fd_eta * exp_noise).detach().cpu().numpy())
                        normalized_q2.append((-args.fd_eta * exp_noise).detach().cpu().numpy())
                        logits_q1_list.append(logits_q1.detach().cpu().numpy())
                        logits_q2_list.append(logits_q2.detach().cpu().numpy())
                # Preserve images that are already done,
                # Unless we are specifically measuring gradient estimation
                prior = prior * not_dones_mask.view(-1, 1, 1, 1).cuda()

            ## Update the image:
            # take a pgd step using the prior
            new_im = image_step(image, upsampler(prior * correct_classified_mask.view(-1, 1, 1, 1)), args.image_lr)  # prior放大后相当于累积的更新量，可以用来更新
            image = proj_step(new_im)
            image = torch.clamp(image, 0, 1)

            ## Continue query count
            total_queries += 2 * args.gradient_iters * not_dones_mask  # gradient_iters是一个int值
            with torch.no_grad():
                adv_pred = model_to_fool(image).argmax(1)
            if args.targeted:
                not_dones_mask = not_dones_mask * (1 - adv_pred.eq(target_label).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_dones_mask = not_dones_mask * adv_pred.eq(true_label).float()  # 只要是跟原始label相等的，就还需要query，还没有成功

            ## Logging stuff
            success_mask = (1 - not_dones_mask) * correct_classified_mask
            num_success = success_mask.sum()
            current_success_rate = (num_success.detach().cpu() / correct_classified_mask.detach().cpu().sum()).cpu().item()
            if num_success == 0:
                success_queries = 0
            else:
                success_queries = ((success_mask * total_queries).sum() / num_success).cpu().item()
            max_curr_queries = total_queries.max().cpu().item()
            # log.info("%d-th: Queries: %d | Success rate: %f | Average queries: %f" % (i, max_curr_queries, current_success_rate, success_queries))
            # if current_success_rate == 1.0:
            #     break

        normalized_q1 = np.ascontiguousarray(np.transpose(np.stack(list(normalized_q1)), axes=(1,0,2,3,4)))
        normalized_q2 = np.ascontiguousarray(np.transpose(np.stack(list(normalized_q2)), axes=(1,0,2,3,4)))
        images = np.ascontiguousarray(np.transpose(np.stack(list(images)), axes=(1,0,2,3,4)))
        logits_q1_list = np.ascontiguousarray(np.transpose(np.stack(list(logits_q1_list)),axes=(1,0,2)))  # B,T,#class
        logits_q2_list = np.ascontiguousarray(np.transpose(np.stack(list(logits_q2_list)),axes=(1,0,2)))  # B,T,#class

        return {
            'average_queries': success_queries,
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
            'success_rate': current_success_rate,
            'images_orig': orig_images.cpu().numpy(),
            'images_adv': image.cpu().numpy(),
            'all_queries': total_queries.cpu().numpy(),
            'correctly_classified': correct_classified_mask.cpu().numpy(),
            'success': success_mask.cpu().numpy(),
            "q1":normalized_q1,
            "q2":normalized_q2,
            "images": images,
            "logits_q1": logits_q1_list,
            "logits_q2": logits_q2_list
        }
    @staticmethod
    def chunks(l, n, archs):
        n = max(1, n)
        batch_idx_arch_dict = dict()
        the_list = [l[i:i + n] for i in range(0, len(l), n)]
        for arch_idx, arch in enumerate(archs):
            for batch_idx in the_list[arch_idx%len(the_list)]:
                batch_idx_arch_dict[batch_idx] = arch
        return batch_idx_arch_dict

    @classmethod
    def attack(cls, args, dataset_loader, model_info_list, attack_norm, save_dir):
        batch_size = args.batch_size
        num_batches = int(np.ceil(min(len(dataset_loader.dataset), args.total_images) / batch_size))
        n = int(np.ceil(num_batches / len(model_info_list)))
        assign_arch_list = BanditAttack.chunks(np.arange(num_batches).tolist(), n, model_info_list)
        last_arch = assign_arch_list[0]["arch_name"]
        normalized_q1_list = []
        normalized_q2_list = []
        images_list = []
        logits_q1_list = []
        logits_q2_list = []
        all_gt_labels = []
        all_targets = []
        targeted_str = "untargeted" if not args.targeted else "targeted_{}".format(args.target_type)
        total_correct, total_adv, total_queries = 0, 0, 0
        for batch_idx, (images, labels) in enumerate(dataset_loader):
            if batch_idx * batch_size > args.total_images:
                break
            attacked_network_info =  assign_arch_list.get(batch_idx, random.choice(list(assign_arch_list.values())))
            arch = attacked_network_info["arch_name"]
            if os.path.exists("{}/dataset_{}@attack_{}@arch_{}@loss_{}@{}@images.npy".format(save_dir, args.dataset,
                                                                                       attack_norm,
                                                                                       arch, args.loss,
                                                                                       targeted_str)):
                log.info("skip {}".format(arch))
                continue
            if last_arch != arch and normalized_q1_list:
                save_path_prefix = "{}/dataset_{}@attack_{}@arch_{}@loss_{}@{}".format(save_dir, args.dataset,
                                                                                       attack_norm,
                                                                                       last_arch, args.loss,
                                                                                       targeted_str)
                normalized_q1_list = np.concatenate(normalized_q1_list, 0)  # N,T,C,H,W
                normalized_q2_list = np.concatenate(normalized_q2_list, 0)  # N,T,C,H,W
                images_list = np.concatenate(images_list, 0)  # N,T,C,H,W
                logits_q1_list = np.concatenate(logits_q1_list, 0)  # N,T,class
                logits_q2_list = np.concatenate(logits_q2_list, 0)  # N,T,class
                store_shape = str(images_list.shape)
                gt_labels = np.array(all_gt_labels).astype(np.int32)
                if args.targeted:
                    all_targets = np.array(all_targets).astype(np.int32)
                os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
                q1_path = "{}@q1.npy".format(save_path_prefix)
                q2_path = "{}@q2.npy".format(save_path_prefix)
                img_path = "{}@images.npy".format(save_path_prefix)
                logits_q1_path = "{}@logits_q1.npy".format(save_path_prefix)
                logits_q2_path = "{}@logits_q2.npy".format(save_path_prefix)
                gt_labels_path = "{}@gt_labels.npy".format(save_path_prefix)
                targets_path = "{}@targets.npy".format(save_path_prefix)
                count_path = "{}@shape.txt".format(save_path_prefix)
                fp = np.memmap(q1_path, dtype='float32', mode='w+', shape=normalized_q1_list.shape)
                fp[:, :, :, :, :] = normalized_q1_list[:, :, :, :, :]
                del fp
                del normalized_q1_list
                fp = np.memmap(q2_path, dtype='float32', mode='w+', shape=normalized_q2_list.shape)
                fp[:, :, :, :, :] = normalized_q2_list[:, :, :, :, :]
                del fp
                del normalized_q2_list
                fp = np.memmap(img_path, dtype='float32', mode='w+', shape=images_list.shape)
                fp[:, :, :, :, :] = images_list[:, :, :, :, :]
                del fp
                del images_list
                fp = np.memmap(logits_q1_path, dtype='float32', mode='w+', shape=logits_q1_list.shape)
                fp[:, :, :] = logits_q1_list[:, :, :]
                del fp
                del logits_q1_list
                fp = np.memmap(logits_q2_path, dtype='float32', mode='w+', shape=logits_q2_list.shape)
                fp[:, :, :] = logits_q2_list[:, :, :]
                del fp
                del logits_q2_list

                np.save(gt_labels_path, gt_labels)
                if args.targeted:
                    np.save(targets_path, all_targets)
                with open(count_path, "w") as file_count:
                    file_count.write(store_shape)
                    file_count.flush()

                log.info("write {} done".format(save_path_prefix))
                normalized_q1_list = []
                normalized_q2_list = []
                images_list = []
                logits_q1_list = []
                logits_q2_list = []
                all_gt_labels = []
                all_targets = []
                model_to_fool.cpu()
                log.info("-" * 80)
                if total_adv > 0 and total_correct > 0:
                    log.info("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
                        aq=total_queries / total_adv,
                        succ=total_adv / total_correct))
                else:
                    log.info("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
                        aq=0,
                        succ=0))
                log.info("-" * 80)
                total_correct, total_adv, total_queries = 0, 0, 0

            images, labels = images.cuda(), labels.long().cuda()
            model_to_fool = attacked_network_info["model"].cuda().eval()
            last_arch = arch
            if args.targeted:
                if args.target_type == 'random':
                    target = torch.randint(low=0, high=CLASS_NUM[args.dataset], size=labels.size()).long().cuda()
                elif args.target_type == 'least_likely':
                    with torch.no_grad():
                        logits = model_to_fool(images)
                    target = logits.argmin(dim=1)
                # make sure target is not equal to label for any example
                invalid_target_index = target.eq(labels).long()
                while invalid_target_index.sum().item() > 0:
                    target[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                                 size=target[invalid_target_index].shape).long().cuda()
                    invalid_target_index = target.eq(labels)
                target = target.cuda()
            else:
                target = None
            res = BanditAttack.make_adversarial_examples(images, labels, target, args, attack_norm, model_to_fool)
            normalized_q1 = res["q1"]
            normalized_q2 = res["q2"]
            images = res["images"]
            logits_q1 = res["logits_q1"]
            logits_q2 = res["logits_q2"]
            normalized_q1_list.append(normalized_q1)
            normalized_q2_list.append(normalized_q2)
            images_list.append(images)
            logits_q1_list.append(logits_q1)
            logits_q2_list.append(logits_q2)
            all_gt_labels.extend(labels.detach().cpu().numpy().tolist())
            if args.targeted:
                all_targets.extend(target.detach().cpu().numpy().tolist())
            ncc = res['num_correctly_classified']  # Number of correctly classified images (originally)
            num_adv = ncc * res['success_rate']  # Success rate was calculated as (# adv)/(# correct classified)
            queries = num_adv * res[
                'average_queries']  # Average queries was calculated as (total queries for advs)/(# advs)
            total_correct += ncc
            total_adv += num_adv
            total_queries += queries

        if normalized_q1_list:
            save_path_prefix = "{}/dataset_{}@attack_{}@arch_{}@loss_{}@{}".format(save_dir, args.dataset,
                                                                                   attack_norm,
                                                                                   last_arch, args.loss,
                                                                                   targeted_str)
            normalized_q1_list = np.concatenate(normalized_q1_list, 0)  # N,T,C,H,W
            normalized_q2_list = np.concatenate(normalized_q2_list, 0)  # N,T,C,H,W
            images_list = np.concatenate(images_list, 0)  # N,T,C,H,W
            logits_q1_list = np.concatenate(logits_q1_list, 0)  # N,T,class
            logits_q2_list = np.concatenate(logits_q2_list, 0)  # N,T,class
            store_shape = str(images_list.shape)
            gt_labels = np.array(all_gt_labels).astype(np.int32)
            if args.targeted:
                all_targets = np.array(all_targets).astype(np.int32)
            os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
            q1_path = "{}@q1.npy".format(save_path_prefix)
            q2_path = "{}@q2.npy".format(save_path_prefix)
            img_path = "{}@images.npy".format(save_path_prefix)
            logits_q1_path = "{}@logits_q1.npy".format(save_path_prefix)
            logits_q2_path = "{}@logits_q2.npy".format(save_path_prefix)
            gt_labels_path = "{}@gt_labels.npy".format(save_path_prefix)
            targets_path = "{}@targets.npy".format(save_path_prefix)
            count_path = "{}@shape.txt".format(save_path_prefix)
            fp = np.memmap(q1_path, dtype='float32', mode='w+', shape=normalized_q1_list.shape)
            fp[:, :, :, :, :] = normalized_q1_list[:, :, :, :, :]
            del fp
            del normalized_q1_list
            fp = np.memmap(q2_path, dtype='float32', mode='w+', shape=normalized_q2_list.shape)
            fp[:, :, :, :, :] = normalized_q2_list[:, :, :, :, :]
            del fp
            del normalized_q2_list
            fp = np.memmap(img_path, dtype='float32', mode='w+', shape=images_list.shape)
            fp[:, :, :, :, :] = images_list[:, :, :, :, :]
            del fp
            del images_list
            fp = np.memmap(logits_q1_path, dtype='float32', mode='w+', shape=logits_q1_list.shape)
            fp[:, :, :] = logits_q1_list[:, :, :]
            del fp
            del logits_q1_list
            fp = np.memmap(logits_q2_path, dtype='float32', mode='w+', shape=logits_q2_list.shape)
            fp[:, :, :] = logits_q2_list[:, :, :]
            del fp
            del logits_q2_list

            np.save(gt_labels_path, gt_labels)
            if args.targeted:
                np.save(targets_path, all_targets)
            with open(count_path, "w") as file_count:
                file_count.write(store_shape)
                file_count.flush()

            log.info("write {} done".format(save_path_prefix))
            log.info("-" * 80)
            if total_adv > 0 and total_correct > 0:
                log.info("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
                    aq=total_queries / total_adv,
                    succ=total_adv / total_correct))
            else:
                log.info("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
                    aq=0,
                    succ=0))
            log.info("-" * 80)

def get_log_path(dataset, loss, norm, targeted, target_type):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    dirname = 'generate_data-{}-{}loss-{}-{}.log'.format(dataset, loss, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    # set log file
    import subprocess
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--json-config-file', default="./configures/bandits_attack_conf.json",
                        type=str, help='a configures file to be passed in instead of arguments')
    parser.add_argument("--dataset", type=str, choices=["CIFAR-10","CIFAR-100","MNIST","FashionMNIST","TinyImageNet","ImageNet"])
    parser.add_argument("--batch-size", type=int,default=100)
    parser.add_argument("--total-images",type=int,default=100000)
    parser.add_argument('--targeted', action="store_true", help="the targeted attack data")
    parser.add_argument("--target_type",type=str, default="random", choices=["least_likely","random"])
    parser.add_argument("--loss",type=str, default="cw", choices=["xent", "cw"])
    parser.add_argument("--max-queries", type=int,default=1000)
    parser.add_argument("--norm",type=str, choices=['linf','l2',"all"], required=True)
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()
    if args.dataset == "ImageNet":
        args.tiling = True
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    save_dir_path = "./data_bandit_attack/{}/{}".format(args.dataset,
                                                         "targeted_attack" if args.targeted else "untargeted_attack")
    os.makedirs(save_dir_path, exist_ok=True)
    log_path = osp.join(save_dir_path, get_log_path(args.dataset, args.loss, args.norm, args.targeted, args.target_type))  # 随机产生一个目录用于实验
    set_log_file(log_path)
    log.info("Log file is located in {}".format(log_path))
    log.info("All the data will be saved into {}".format(save_dir_path))
    log.info("Using GPU {}".format(args.gpu))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))

    # If a json file is given, use the JSON file as the base, and then update it with args
    with open(args.json_config_file, "r") as file_obj:
        attack_json = json.load(file_obj)
    attack_type_params = []

    for attack_norm, attack_conf in attack_json[args.dataset].items():
        if args.norm != "all" and args.norm != attack_norm:
            continue
        attack_conf.update(vars(args))
        params = SimpleNamespace(**attack_conf)
        attack_type_params.append((attack_norm, params))
    trn_data_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, is_train=True)  # 生成的是训练集而非测试集

    models = []
    for arch in MODELS_TRAIN_STANDARD[args.dataset]:
        if StandardModel.check_arch(arch, args.dataset):
            model = StandardModel(args.dataset, arch, True)
            models.append({"arch_name":arch, "model":model})
    with torch.no_grad():
        for attack_norm, args_item in attack_type_params:
            log.info('Called with args:')
            print_args(args_item)
            BanditAttack.attack(args_item, trn_data_loader, models, attack_norm, save_dir_path)
