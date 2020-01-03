#!/usr/bin/env python3
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import os
import os.path as osp
import glog as log
import argparse
import json
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
from target_models.standard_model import StandardModel
from dataset.data_loader_maker import make_loader


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp-dir', default='output/debug', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--dataset', default='CIFAR-10', type=str, choices=['cifar10', 'imagenet'],
                        help='which dataset to use')
    parser.add_argument('--save-every-step', action='store_true',
                        help='save meta-information every PGD step to disk')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--phase', default='validation', type=str, choices=['validation', 'test'],
                        help='train_simulate_grad_mode, val, test')
    parser.add_argument('--num-image', default=1000, type=int,
                        help='how many images to compute')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--delta-size', default=0, type=int,
                        help='size (width/height) of delta. if not equal to image shape, we resize delta to the image'
                             'shape and then add the resized delta to image. set this to 0 to disable resizing')
    parser.add_argument('--arch', default='wrn-28-10-drop', type=str,
                        help='network architecture')
    parser.add_argument('--ref-arch', nargs='*', default=[],
                        help='reference architectures for gradient subspace computation')
    parser.add_argument('--ref-arch-train-data', default='full', type=str,
                        choices=['full', 'cifar10.1', 'imagenetv2-val'],
                        help='ref target_models are trained on which training set')
    parser.add_argument('--ref-arch-epoch', default='final', type=str,
                        help='use ref target_models at which epoch, could be final, best, or a epoch number')
    parser.add_argument('--ref-arch-init-drop', default=0.0, type=float,
                        help='for dropout probability for ref model')
    parser.add_argument('--ref-arch-max-drop', default=0.5, type=float,
                        help='maximum allowed dropout probability')
    parser.add_argument('--ref-arch-drop-grow-iter', default=10, type=int,
                        help='increase dropout probability at this iteration')
    parser.add_argument('--ref-arch-drop-gamma', default=0.01, type=float,
                        help='control dropout probability increasing speed')
    parser.add_argument('--fix-grad', action='store_true',
                        help='fix gradient from dL/dv to dl/dv')
    parser.add_argument('--loss', default='cw', type=str, choices=['xent', 'cw'],
                        help='loss function, could be cw or xent')
    parser.add_argument('--exploration', default=0.1, type=float,
                        help='exploration for finite difference prior')
    parser.add_argument('--fd-eta', default=0.1, type=float,
                        help='finite difference eta')
    parser.add_argument('--image-lr', default=1./255, type=float,
                        help='learning rate for image')
    parser.add_argument('--prior-lr', default=1.0, type=float,
                        help='learning rate for prior')
    parser.add_argument('--prior-update', default='momentum', choices=['eg', 'gd', 'momentum'], type=str,
                        help='update method of prior')
    parser.add_argument('--eg-clip', default=3.0, type=float,
                        help='clip for eg update')
    parser.add_argument('--num-fix-direction', default=0, type=int,
                        help='fix normal direction for illustration experiments')   # fix normal direction
    parser.add_argument('--fix-direction-seed', default=1234, type=int,
                        help='seed for fix direction generation')
    parser.add_argument('--norm-type', default='linf', type=str, choices=['l2', 'linf'],
                        help='l_p norm type, could be l2 or linf')
    parser.add_argument('--epsilon', default=8./255, type=float,
                        help='allowed l_p perturbation size')
    parser.add_argument('--max-query', default=10000, type=int,
                        help='maximum allowed queries for each image')
    parser.add_argument('--attack-type', default='untargeted', choices=['untargeted', 'targeted'],
                        help='type of attack, could be targeted or untargeted')
    parser.add_argument('--target-type', default='random', type=str, choices=['random', 'least_likely'],
                        help='how to choose target class for targeted attack, could be random or least_likely')
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("using GPU :{}".format(args.gpu))
    return args




def norm(t, p=2):
    assert len(t.shape) == 4
    if p == 2:
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
    elif p == 1:
        norm_vec = t.abs().sum(dim=[1, 2, 3]).view(-1, 1, 1, 1)
    else:
        raise NotImplementedError('Unknown norm p={}'.format(p))
    norm_vec += (norm_vec == 0).float() * 1e-8
    return norm_vec


def eg_prior_step(x, g, lr):
    real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
    lrg = torch.clamp(lr * g, -args.eg_clip, args.eg_clip)
    pos = real_x * torch.exp(lrg)
    neg = (1 - real_x) * torch.exp(-lrg)
    new_x = pos / (pos + neg)
    return new_x * 2 - 1


def gd_prior_step(x, g, lr):
    return x + lr * g


def momentum_prior_step(x, g, lr):
    # adapted from Boosting Adversarial Attacks with Momentum, CVPR 2018
    return x + lr * g / norm(g, p=1)


def linf_image_step(x, g, lr):
    return x + lr * torch.sign(g)


def l2_image_step(x, g, lr):
    return x + lr * g / norm(g)


def l2_proj_step(image, epsilon, adv_image):
    delta = adv_image - image
    out_of_bounds_mask = (norm(delta) > epsilon).float()
    return out_of_bounds_mask * (image + epsilon * delta / norm(delta)) + (1 - out_of_bounds_mask) * adv_image


def linf_proj_step(image, epsilon, adv_image):
    return image + torch.clamp(adv_image - image, -epsilon, epsilon)


def cw_loss(logit, label, target=None):
    if target is not None:
        # targeted cw loss: logit_t - max_{i\neq t}logit_i
        _, argsort = logit.sort(dim=1, descending=True)
        target_is_max = argsort[:, 0].eq(target)
        second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
        target_logit = logit[torch.arange(logit.shape[0]), target]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return target_logit - second_max_logit
    else:
        # untargeted cw loss: max_{i\neq y}logit_i - logit_y
        _, argsort = logit.sort(dim=1, descending=True)
        gt_is_max = argsort[:, 0].eq(label)
        second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
        gt_logit = logit[torch.arange(logit.shape[0]), label]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return second_max_logit - gt_logit


def xent_loss(logit, label, target=None):
    if target is not None:
        return -F.cross_entropy(logit, target, reduction='none')
    else:
        return F.cross_entropy(logit, label, reduction='none')


def main():
    # make model(s)
    log.info('Initializing target model {} on {}'.format(args.arch, args.dataset))
    model = StandardModel(args.dataset, args.arch, no_grad=True, train_data='full', epoch='final').eval()

    ref_models = OrderedDict()
    for i, ref_arch in enumerate(args.ref_arch):
        params = dict()
        params['train_data'] = args.ref_arch_train_data
        params['epoch'] = args.ref_arch_epoch
        log.info('Initializing ref model {} on {} ({} of {}), params: {}'.format(
            ref_arch, args.dataset, i + 1, len(args.ref_arch), params))
        ref_models[ref_arch] = StandardModel(args.dataset, ref_arch, no_grad=False, **params).eval()
    log.info('All target_models have been initialized, including 1 target model and {} ref target_models'.format(len(args.ref_arch)))

    # make loader
    kwargs = dict()
    if args.dataset == 'imagenet':
        kwargs['size'] = model.input_size[-1]
    loader = make_loader(args.dataset, args.phase, args.batch_size, args.seed, **kwargs)

    # make operators

    prior_step = eval('{}_prior_step'.format(args.prior_update))
    image_step = eval('{}_image_step'.format(args.norm_type))
    proj_step = eval('{}_proj_step'.format(args.norm_type))

    if args.delta_size > 0:
        # resize
        upsampler = lambda x: F.interpolate(x, size=model.input_size[-1], mode='bilinear', align_corners=True)  # 这就是低维度的sub space放大回来
        downsampler = lambda x: F.interpolate(x, size=args.delta_size, mode='bilinear', align_corners=True)
    else:
        # no resize, upsampler = downsampler = identity
        upsampler = downsampler = lambda x: x # CIFAR-10不用缩尺寸, ImageNet需要缩尺寸

    # make loss function
    loss_func = eval('{}_loss'.format(args.loss))

    # init arrays for saving results
    query_all = torch.zeros(args.num_image)
    correct_all = torch.zeros_like(query_all)
    not_done_all = torch.zeros_like(query_all)  # always set to 0 if the original image is misclassified
    success_all = torch.zeros_like(query_all)
    success_query_all = torch.zeros_like(query_all)
    not_done_loss_all = torch.zeros_like(query_all)
    not_done_prob_all = torch.zeros_like(query_all)

    # make directory for saving results
    result_dirname = osp.join(args.exp_dir, 'results')
    os.makedirs(result_dirname, exist_ok=True)

    # fixed direction for illustration experiments
    if args.num_fix_direction > 0:
        if len(args.ref_arch) == 0:
            # fixed random direction
            assert args.dataset == 'CIFAR-10'
            state = np.random.get_state()
            np.random.seed(args.fix_direction_seed)
            fix_direction = np.random.randn(3072, *model.input_size)[:args.num_fix_direction]
            np.random.set_state(state)
            fix_direction = np.ascontiguousarray(fix_direction)
            fix_direction = torch.FloatTensor(fix_direction).to(device)
        else:
            # fixed gradient direction (calculated at clean inputs)
            assert args.num_fix_direction == len(args.ref_arch)

    for batch_index, (image_id, image, label) in enumerate(loader):
        assert image.max().item() <= 1
        assert image.min().item() >= 0

        # move image and label to device
        image_id = image_id.to(device)
        image = image.to(device)
        label = label.to(device)
        adv_image = image.clone()

        # get logit and prob
        logit = model(image)
        adv_logit = logit.clone()
        pred = logit.argmax(dim=1)

        # choose target classes for targeted attack
        if args.attack_type == 'targeted':
            if args.target_type == 'random':
                target = torch.randint(low=0, high=logit.shape[1], size=label.shape).long().to(device)
            elif args.target_type == 'least_likely':
                target = logit.argmin(dim=1)
            else:
                raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            # make sure target is not equal to label for any example
            invalid_target_index = target.eq(label)
            while invalid_target_index.sum().item() > 0:
                target[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                             size=target[invalid_target_index].shape).long().to(device)
                invalid_target_index = target.eq(label)
        else:
            target = None

        # init masks and selectors
        correct = pred.eq(label).float()
        query = torch.zeros(args.batch_size).to(device)
        not_done = correct.clone()
        selected = torch.arange(batch_index * args.batch_size, (batch_index + 1) * args.batch_size)

        # init prior
        if args.delta_size > 0:
            prior = torch.zeros(args.batch_size, model.input_size[0], args.delta_size, args.delta_size).to(device)
        else:
            prior = torch.zeros(args.batch_size, *model.input_size).to(device)

        # perform attack
        for step_index in range(args.max_query // 2):
            # increase query counts
            query = query + 2 * not_done  # not_done = 预测与gt label相等的图片个数

            # calculate drop probability
            if step_index < args.ref_arch_drop_grow_iter:
                drop = args.ref_arch_init_drop
            else:
                # args.ref_arch_max_drop 默认等于0.5
                drop = args.ref_arch_max_drop - \
                    (args.ref_arch_max_drop - args.ref_arch_init_drop) * \
                    np.exp(-(step_index - args.ref_arch_drop_grow_iter) * args.ref_arch_drop_gamma)

            # finite difference for gradient estimation
            if len(ref_models) > 0:
                # select ref model to calculate gradient
                #
                selected_ref_arch_index = torch.randint(low=0, high=len(ref_models), size=(1,)).long().item()

                # get original model logit's grad
                adv_logit = adv_logit.detach()
                adv_logit.requires_grad = True
                loss = loss_func(adv_logit, label, target).mean()
                logit_grad = torch.autograd.grad(loss, [adv_logit])[0]

                # calculate gradient for all ref target_models
                def calc_ref_grad(adv_image_, ref_model_, drop_=0):
                    adv_image_ = adv_image_.detach()
                    adv_image_.requires_grad = True
                    if adv_image_.grad:
                        adv_image_.grad[:] = 0.
                    ref_model_.zero_grad()

                    # assign dropout probability
                    ref_model_.drop = drop_  # 这个可以进模型代码看看，drop怎么做的

                    # forward ref model
                    if ref_model_.input_size != model.input_size:
                        ref_logit_ = ref_model_(F.interpolate(adv_image_, size=ref_model_.input_size[-1],
                                                              mode='bilinear', align_corners=True))
                    else:
                        ref_logit_ = ref_model_(adv_image_)

                    # backward ref model using logit_grad from the victim model
                    ref_grad_ = torch.autograd.grad(ref_logit_, [adv_image_], grad_outputs=[logit_grad])[0]
                    ref_grad_ = downsampler(ref_grad_)  # 高维度缩小， subspace的精髓

                    # compute dl/dv
                    if args.fix_grad:
                        if prior.view(prior.shape[0], -1).norm(dim=1).min().item() > 0:
                            # -1 / ||v|| ** 3 (||v|| ** 2 dL/dv - v(v^T dL/dv))
                            g1 = norm(prior) ** 2 * ref_grad_
                            g2 = prior * (prior * ref_grad_).sum(dim=(1, 2, 3)).view(-1, 1, 1, 1)
                            ref_grad_ = g1 - g2
                    return ref_grad_ / norm(ref_grad_)  # 拿到direction

                # calculate selected ref model's gradient
                if args.num_fix_direction == 0:
                    # 随机选择一个模型，输入adv_image,得到梯度.这个梯度是否准确不知道，因为是随机选择的模型，不如用网络生成
                    direction = calc_ref_grad(adv_image, list(ref_models.values())[selected_ref_arch_index], drop_=drop)
                else:
                    # for illustrate experiment in rebuttal
                    assert args.loss == 'cw'
                    assert drop == 0
                    direction = calc_ref_grad(image, list(ref_models.values())[selected_ref_arch_index], drop_=drop)

            else:
                # use random search direction solely
                if args.num_fix_direction > 0:
                    # use fixed direction (for illustration experiments)
                    if len(args.ref_arch) == 0:
                        # fixed random direction
                        # fix_direction.shape: [num_fix_direction, C, H, W]
                        # coeff.shape: [num_Fix_direction, N]
                        coeff = torch.randn(args.num_fix_direction, prior.shape[0]).to(device)
                        direction = (fix_direction.view(fix_direction.shape[0], 1, *fix_direction.shape[1:]) *
                                     coeff.view(coeff.shape[0], coeff.shape[1], 1, 1, 1)).sum(dim=0)
                    else:
                        # fixed gradient direction (calculated at clean inputs) for rebuttal
                        # direction has already been calculated
                        assert direction.shape[0] == image.shape[0]
                else:
                    direction = torch.randn_like(prior)

            # normalize search direction
            direction = direction / norm(direction)  #这个方向是用随机选择一个model，估计出来的梯度，可以换成用meta预测方向。ground truth用prior来给最终loss，由于meta训练已知模型的话梯度非常好给，所以可以用梯度累积后的prior做ground truth

            # 第一个初步的想法，用MAML，输入图像，预测这个direction，然后继续向后算出l1和l2，最后算出grad，监督的ground truth用真正的对图像求导的梯度来监督. 这就把原本的随机exp_noise变成有预测的了,反向传播，算出这个direction是否合理，其中后一个model的参数固定
            # 更好地想法，以往Meta Learning没有考虑时序关系，但是这个prior很明显是有时序关系的，因此可以用ConvLSTM
            # 采用基于ConvLSTM的MAML模型，ConvLSTM的输入和输出都是T,N,C,H,W，然后每个时间步可以用梯度放入监督信号，为了简便起见，训练的时候输入的x用PGD更新过的N个步骤x'作为输入。
            # 然后分成K个task，最后的query set用干净的未经修改的图输入，而输出是T个时间步的扰动的累积，或者压根不要query set了
            q1 = upsampler(prior + args.exploration * direction)  # 这两个upsampler和downsampler，使用在ImageNet大图起作用
            q2 = upsampler(prior - args.exploration * direction)  # 方向来自于随机选取的model，可以改为meta learning设计一下
            l1 = loss_func(model(adv_image + args.fd_eta * q1 / norm(q1)), label, target)  # 需要查询
            l2 = loss_func(model(adv_image + args.fd_eta * q2 / norm(q2)), label, target)
            grad = (l1 - l2) / (args.fd_eta * args.exploration)  # 需要2次查询，grad是论文Alg1第11行左边那个Delta_t，用于更新梯度的一个量
            # 这段抄袭的bandit attack，但是把原来的exp_noise换成了direction
            grad = grad.view(-1, 1, 1, 1) * direction     # grad.shape == direction.shape == prior.shape ?= image.shape 这就是精髓所在

            # update prior，其实prior就是梯度，因为后一个prior和前一个有联系，类似贝叶斯的prior
            prior = prior_step(prior, grad, args.prior_lr)  # 用grad更新prior，piror最后更新到图像上。prior就是图像梯度

            # extract grad from prior
            grad = upsampler(prior)  # prior相当于梯度

            # update adv_image (correctly classified images only)
            adv_image = image_step(adv_image, grad * correct.view(-1, 1, 1, 1), args.image_lr)
            adv_image = proj_step(image, args.epsilon, adv_image)
            adv_image = torch.clamp(adv_image, 0, 1)

            # update statistics
            adv_logit = model(adv_image)
            adv_pred = adv_logit.argmax(dim=1)
            adv_prob = F.softmax(adv_logit, dim=1)
            adv_loss = loss_func(adv_logit, label, target)
            if args.attack_type == 'targeted':
                not_done = not_done * (1 - adv_pred.eq(target)).float()
            else:
                not_done = not_done * adv_pred.eq(label).float()
            success = (1 - not_done) * correct  # currently done, originally correct --> success
            success_query = success * query
            not_done_loss = adv_loss * not_done
            not_done_prob = adv_prob[torch.arange(args.batch_size), label] * not_done

            # log
            log.info('Attacking image {} - {} / {}, step {}, max query {}'.format(
                batch_index * args.batch_size, (batch_index + 1) * args.batch_size,
                args.num_image, step_index + 1, int(query.max().item())
            ))
            log.info('        correct: {:.4f}'.format(correct.mean().item()))
            log.info('       not_done: {:.4f}'.format(not_done.mean().item()))
            log.info('      fd_scalar: {:.4f}'.format((l1 - l2).mean().item()))
            log.info('           drop: {:.4f}'.format(drop))
            if success.sum().item() > 0:
                log.info('     mean_query: {:.4f}'.format(success_query[success.byte()].mean().item()))
                log.info('   median_query: {:.4f}'.format(success_query[success.byte()].median().item()))
            if not_done.sum().item() > 0:
                log.info('  not_done_loss: {:.4f}'.format(not_done_loss[not_done.byte()].mean().item()))
                log.info('  not_done_prob: {:.4f}'.format(not_done_prob[not_done.byte()].mean().item()))

            if args.save_every_step and batch_index == 0 and step_index <= 500:
                # save meta-info in each step to disk for further analysis
                for single_image_index, single_image_id in enumerate(image_id.cpu().numpy().astype(np.int)[:30]):
                    result_fname = osp.join(result_dirname, 'step-results', str(single_image_id),
                                            'step-{}.pth'.format(step_index))
                    os.makedirs(osp.dirname(result_fname), exist_ok=True)
                    torch.save({'adv_loss': adv_loss[single_image_index].item(),
                                'not_done': not_done[single_image_index].item(),
                                'adv_logit': adv_logit[single_image_index].cpu()}, result_fname)

            # early break if all succeed
            if not not_done.byte().any():
                log.info('  image {} - {} all succeed in step {}, break'.format(
                    batch_index * args.batch_size, (batch_index + 1) * args.batch_size, step_index
                ))
                break

        # save results to arrays
        # 下面这段代码统计最终的统计量，比如success_query_all
        for key in ['query', 'image_id', 'label', 'logit', 'correct', 'adv_logit', 'adv_loss', 'not_done',
                    'success', 'success_query', 'not_done_loss', 'not_done_prob']:
            value_all = eval('{}_all'.format(key))
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()

        # save image and adv_image to disk
        for single_image_index, single_image_id in enumerate(image_id.cpu().numpy().astype(np.int)):
            # index is the position in current batch
            # id (defined in data_loader_maker.py) is the identifier in the whole dataset
            # save image
            result_fname = osp.join(result_dirname, 'images', str(single_image_id), 'image.pth')
            os.makedirs(osp.dirname(result_fname), exist_ok=True)
            torch.save(image[single_image_index].cpu(), result_fname)

            # save adv_image
            result_fname = osp.join(result_dirname, 'images', str(single_image_id), 'adv_image.pth')
            os.makedirs(osp.dirname(result_fname), exist_ok=True)
            torch.save(adv_image[single_image_index].cpu(), result_fname)

        # break if we've already attacked args.num_image images
        if (batch_index + 1) * args.batch_size >= args.num_image:
            break

    # log statistics for args.num_image images
    log.info('Attack finished ({} images)'.format(args.num_image))
    log.info('        avg correct: {:.4f}'.format(correct_all.mean().item()))
    log.info('       avg not_done: {:.4f}'.format(not_done_all.mean().item()))
    if success_all.sum().item() > 0:
        log.info('     avg mean_query: {:.4f}'.format(success_query_all[success_all.byte()].mean().item()))
        log.info('   avg median_query: {:.4f}'.format(success_query_all[success_all.byte()].median().item()))
    if not_done_all.sum().item() > 0:
        log.info('  avg not_done_loss: {:.4f}'.format(not_done_loss_all[not_done_all.byte()].mean().item()))
        log.info('  avg not_done_prob: {:.4f}'.format(not_done_prob_all[not_done_all.byte()].mean().item()))

    # save results to disk
    log.info('Saving results to {}'.format(result_dirname))
    for key in ['query', 'image_id', 'label', 'logit', 'correct', 'adv_logit', 'adv_loss', 'not_done',
                'success', 'success_query', 'not_done_loss', 'not_done_prob']:
        value_all = eval('{}_all'.format(key))
        result_fname = osp.join(result_dirname, '{}_all.pth'.format(key))
        torch.save(value_all.cpu(), result_fname)
        log.info('{}_all saved to {}'.format(key, result_fname))

    log.info('Done')


def set_log_file(fname):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    import subprocess

    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def print_args():
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname


if __name__ == '__main__':
    # before going to the attack function, we do following things:
    # 1. setup output directory
    # 2. make global variables: args, model (on cpu), loaders and device

    # 1. setup output directory
    args = parse_args()

    args.exp_dir = osp.join(args.exp_dir, get_random_dir_name())  # 随机产生一个目录用于实验
    if not osp.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # set log file
    set_log_file(osp.join(args.exp_dir, 'run.log'))

    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args()

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    fname = __file__
    if fname.endswith('pyc'):
        fname = fname[:-1]
    os.system('cp {} {}'.format(fname, args.exp_dir))
    os.system('cp -r *.py target_models {}'.format(args.exp_dir))

    # 2. make global variables

    # check device
    device = torch.device('cuda')

    # set random seed before init model
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)

    # do the business
    main()