import sys
import os

sys.path.append(os.getcwd())
import argparse
import glob
import json
from types import SimpleNamespace
from dataset.defensive_model import DefensiveModel

import glog as log
import numpy as np
import torch
from torch.nn import functional as F

from config import CLASS_NUM, PY_ROOT, MODELS_TEST_STANDARD, IMAGE_DATA_ROOT
from dataset.standard_model import StandardModel
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.target_class_dataset import ImageNetDataset,CIFAR10Dataset,CIFAR100Dataset

class NES(object):
    def __init__(self, dataset_name, targeted):
        self.dataset_name = dataset_name
        self.num_classes = CLASS_NUM[self.dataset_name]
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset_name, 1)
        self.total_images = len(self.dataset_loader.dataset)
        self.targeted = targeted
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)

    def get_image_of_target_class(self,dataset_name, target_labels, target_model):

        images = []
        for label in target_labels:  # length of target_labels is 1
            if dataset_name == "ImageNet":
                dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name],label.item(), "validation")
            elif dataset_name == "CIFAR-10":
                dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name=="CIFAR-100":
                dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")

            index = np.random.randint(0, len(dataset))
            image, true_label = dataset[index]
            image = image.unsqueeze(0)
            if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                image = F.interpolate(image,
                                       size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logits = target_model(image.cuda())
            while logits.max(1)[1].item() != label.item():
                index = np.random.randint(0, len(dataset))
                image, true_label = dataset[index]
                image = image.unsqueeze(0)
                if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                    image = F.interpolate(image,
                                       size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
                with torch.no_grad():
                    logits = target_model(image.cuda())
            assert true_label == label.item()
            images.append(torch.squeeze(image))
        return torch.stack(images) # B,C,H,W

    def xent_loss(self, logits, noise, true_labels, target_labels, top_k):
        if self.targeted:
            return F.cross_entropy(logits, target_labels, reduction='none'), noise  # FIXME 修改测试
        else:
            assert target_labels is None, "target label must set to None in untargeted attack"
            return F.cross_entropy(logits, true_labels, reduction='none'), noise

    def partial_info_loss(self, logits, noise, true_labels, target_labels, top_k):
        # logit 是融合了batch_size of noise 的, shape = (batch_size, num_classes)
        losses, noise = self.xent_loss(logits=logits,noise=noise, true_labels=true_labels, target_labels=target_labels, top_k=top_k)
        vals, inds = torch.topk(logits, dim=1, k=top_k, largest=True, sorted=True) # inds shape = (B, top_k)
        # inds is batch_size x k
        target_class = target_labels[0].item()  # 一个batch的target都是一样的
        good_image_inds = torch.sum(inds == target_class, dim=1).byte()    # shape = (batch_size,)
        losses = losses[good_image_inds]
        noise = noise[good_image_inds]
        return losses, noise

    #  STEP CONDITION (important for partial-info attacks)
    def robust_in_top_k(self, target_model, adv_images, target_labels, top_k):
        # 我自己增加的代码
        if self.targeted:  # FIXME 作者默认targeted模式top_k < num_classes
            eval_logits = target_model(adv_images)
            t = target_labels[0].item()
            pred = eval_logits.max(1)[1][0].item()
            return pred == t
        if top_k == self.num_classes:   #
            return True
        eval_logits = target_model(adv_images)
        t = target_labels[0].item()
        _, top_pred_indices = torch.topk(eval_logits, k=top_k, largest=True,
                                               sorted=True)  # top_pred_indices shape = (1, top_k)
        top_pred_indices = top_pred_indices.view(-1).detach().cpu().numpy().tolist()
        if t not in top_pred_indices:
            return False
        return True

    def get_grad(self, x, sigma, samples_per_draw, batch_size, true_labels, target_labels, target_model, loss_fn, top_k):
        num_batches = samples_per_draw // batch_size  # 一共产生多少个samples噪音点，每个batch
        losses = []
        grads = []

        for _ in range(num_batches):
            assert x.size(0) == 1
            noise_pos = torch.randn((batch_size//2,) + (x.size(1), x.size(2), x.size(3)))  # B//2, C, H, W
            noise = torch.cat([-noise_pos, noise_pos], dim=0).cuda()  # B, C, H, W
            eval_points = x + sigma * noise  # 1,C,H,W + B, C, H, W = B,C,H,W
            logits = target_model(eval_points)  # B, num_classes
            # losses shape = (batch_size,)
            if target_labels is not None:
                target_labels = target_labels.repeat(batch_size)
            loss, noise = loss_fn(logits, noise, true_labels, target_labels, top_k)  # true_labels and target_labels have already repeated for batch_size
            loss = loss.view(-1,1,1,1) # shape = (B,1,1,1)
            grad = torch.mean(loss * noise, dim=0, keepdim=True)/sigma # loss shape = (B,1,1,1) * (B,C,H,W), then mean axis= 0 ==> (1,C,H,W)
            losses.append(loss.mean())
            grads.append(grad)
        losses = torch.stack(losses).mean()  # (1,)
        grads = torch.mean(torch.stack(grads), dim=0)  # (1,C,H,W)
        return losses.item(), grads

    def attack_all_images(self, args, arch_name, target_model, result_dump_path):

        for batch_idx, data_tuple in enumerate(self.dataset_loader):
            # if batch_idx >= self.total_images:
            #     break
            if self.dataset_name == "ImageNet":
                if target_model.input_size[-1]>=299:
                    images, true_labels = data_tuple[1],data_tuple[2]
                else:
                    images, true_labels = data_tuple[0],data_tuple[2]
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
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                            size=target_labels[
                                                                                invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    with torch.no_grad():
                        logit = target_model(images)
                    target_labels = logit.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset]).cuda()
            else:
                target_labels = None
            with torch.no_grad():
                self.make_adversarial_examples(batch_idx, images.cuda(), true_labels.cuda(), target_labels, args, target_model)

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.info('     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            log.info('   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
            log.info('     max query: {}'.format(self.success_query_all[self.success_all.byte()].max().item()))
        log.info('Saving results to {}'.format(result_dump_path))

        query_all_np = self.query_all.detach().cpu().numpy().astype(np.int32)
        not_done_all_np = self.not_done_all.detach().cpu().numpy().astype(np.int32)
        correct_all_np = self.correct_all.detach().cpu().numpy().astype(np.int32)
        out_of_bound_indexes = np.where(query_all_np > args.max_queries)[0]
        if len(out_of_bound_indexes) > 0:
            not_done_all_np[out_of_bound_indexes] = 1
        success_all_np = (1 - not_done_all_np) * correct_all_np
        success_query_all_np = success_all_np * query_all_np
        success_indexes = np.nonzero(success_all_np)[0]

        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": np.mean(not_done_all_np[np.nonzero(correct_all_np)[0]]).item(),
                          "mean_query": np.mean(success_query_all_np[success_indexes]).item(),
                          "median_query": np.median(success_query_all_np[success_indexes]).item(),
                          "max_query": np.max(success_query_all_np[success_indexes]).item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": not_done_all_np.tolist(),
                          "query_all": query_all_np.tolist(),
                          'args': vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj,  sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


    def norm(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def l2_image_step(self, x, g, lr):
        if self.targeted:
            return x - lr * g / self.norm(g)
        return x + lr * g / self.norm(g)

    def linf_image_step(self, x, g, lr):
        if self.targeted:
            return x - lr * torch.sign(g)
        return x + lr * torch.sign(g)

    def l2_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            delta = new_x - orig
            out_of_bounds_mask = (self.norm(delta) > eps).float()
            x = (orig + eps * delta / self.norm(delta)) * out_of_bounds_mask
            x += new_x * (1 - out_of_bounds_mask)
            return x
        return proj

    def linf_proj(self, image, eps):
        orig = image.clone()
        def proj(new_x):
            return orig + torch.clamp(new_x - orig, -eps, eps)
        return proj

    def make_adversarial_examples(self, batch_index, images, true_labels, target_labels, args, target_model):
        batch_size = args.batch_size  # Actually, the batch size of images is 1, the goal of args.batch_size is to sample noises
        # some statistics variables
        assert images.size(0) == 1
        with torch.no_grad():
            logit = target_model(images)
        pred = logit.argmax(dim=1)
        query = torch.zeros(1).cuda()
        correct = pred.eq(true_labels).float()  # shape = (1,)
        not_done = correct.clone()  # shape = (1,)
        success = (1 - not_done) * correct  # correct = 0 and 1-not_done = 1 --> success = 0
        success_query = success * query

        selected = torch.arange(batch_index, batch_index + 1)  # 选择这个batch的所有图片的index
        adv_images = images.clone()

        samples_per_draw = args.samples_per_draw  # samples per draw
        epsilon = args.epsilon   # 最终目标的epsilon
        goal_epsilon = epsilon
        max_lr = args.max_lr
        # ----- partial info params -----
        k = args.top_k
        adv_thresh = args.adv_thresh
        if k > 0 or self.targeted:
            assert self.targeted, "Partial-information attack is a targeted attack."
            adv_images = self.get_image_of_target_class(self.dataset_name, target_labels, target_model)
            epsilon = args.starting_eps
        else:   # if we don't want to top-k paritial attack set k = -1 as the default setting
            k = self.num_classes
        delta_epsilon = args.starting_delta_eps
        g = torch.zeros_like(adv_images).cuda()
        last_ls = []
        true_labels = true_labels.repeat(batch_size)  # for noise sampling points
        # max_iters = int(np.ceil(args.max_queries / args.samples_per_draw)) if k == self.num_classes else int(np.ceil(args.max_queries / (args.samples_per_draw + 1)))
        # if self.targeted:
        #     max_iters = int(np.ceil(args.max_queries / (args.samples_per_draw + 1)))
        loss_fn = self.partial_info_loss if k < self.num_classes else self.xent_loss  # 若非paritial_information模式，k = num_classes
        image_step = self.l2_image_step if args.norm == 'l2' else self.linf_image_step
        proj_maker = self.l2_proj if args.norm == 'l2' else self.linf_proj  # 调用proj_maker返回的是一个函数
        proj_step = proj_maker(images, args.epsilon)
        while query[0].item() < args.max_queries:
            # CHECK IF WE SHOULD STOP
            if not not_done.byte().any() and epsilon <= goal_epsilon:  # all success
                success_indicator_str = "success" if query[0].item() > 0 else "on a incorrectly classified image"
                log.info("Attack {} on {}-th image by using {} queries".format(success_indicator_str,
                                                                               batch_index, query[0].item()))
                break

            prev_g = g.clone()
            l, g = self.get_grad(adv_images, args.sigma, samples_per_draw, batch_size, true_labels, target_labels,
                                 target_model, loss_fn, k)
            query += samples_per_draw
            # log.info("Query :{}".format(query[0].item()))
            # SIMPLE MOMENTUM
            g = args.momentum * prev_g + (1.0 - args.momentum) * g
            # PLATEAU LR ANNEALING
            last_ls.append(l)
            last_ls = last_ls[-args.plateau_length:]  # FIXME 为何targeted的梯度会不断变大？
            condition = last_ls[-1] > last_ls[0] # if self.targeted else last_ls[-1] > last_ls[0]
            if condition and len(last_ls) == args.plateau_length:  # > 改成 < 号了调试，FIXME bug，原本的tf的版本里面loss不带正负号，如果loss变大，就降低lr
                if max_lr > args.min_lr:
                    max_lr = max(max_lr / args.plateau_drop, args.min_lr)
                    log.info("[log] Annealing max_lr : {:.5f}".format(max_lr))
                last_ls = []
            # SEARCH FOR LR AND EPSILON DECAY
            current_lr = max_lr
            prop_de = 0.0
            # if l < adv_thresh and epsilon > goal_epsilon:
            if epsilon > goal_epsilon:
                prop_de = delta_epsilon

            while current_lr >= args.min_lr:
                # PARTIAL INFORMATION ONLY
                # if k < self.num_classes: #FIXME 我认为原作者写错了，这个地方改成targeted
                if self.targeted:
                    proposed_epsilon = max(epsilon - prop_de, goal_epsilon)
                    proj_step = proj_maker(images, proposed_epsilon)
                # GENERAL LINE SEARCH
                proposed_adv = image_step(adv_images, g, current_lr)
                proposed_adv = proj_step(proposed_adv)
                proposed_adv = torch.clamp(proposed_adv, 0, 1)
                if self.targeted or k != self.num_classes:
                    query += 1 # we must query for check robust_in_top_k
                if self.robust_in_top_k(target_model, proposed_adv, target_labels, k):
                    if prop_de > 0:
                        delta_epsilon = max(prop_de, args.min_delta_eps)
                        # delta_epsilon = prop_de
                    adv_images = proposed_adv
                    if self.targeted:
                        epsilon = proposed_epsilon   # FIXME 我自己增加的代码
                    else:
                        epsilon = max(epsilon - prop_de / args.conservative, goal_epsilon)
                    break
                elif current_lr >= args.min_lr * 2:
                    current_lr = current_lr / 2
                else:
                    prop_de = prop_de / 2
                    if prop_de == 0:
                        break
                    if prop_de < 2e-3:
                        prop_de = 0
                    current_lr = max_lr
                    log.info("[log] backtracking eps to {:.3f}".format(epsilon - prop_de,))


            with torch.no_grad():
                adv_logit = target_model(adv_images)
            adv_pred = adv_logit.argmax(dim=1)  # shape = (1, )
            # adv_prob = F.softmax(adv_logit, dim=1)
            # adv_loss, _ = loss_fn(adv_logit, None, true_labels[0].unsqueeze(0), target_labels, top_k=k)
            if self.targeted:
                not_done = (1 - adv_pred.eq(target_labels).float()).float()
            else:
                not_done =  adv_pred.eq(true_labels[0].unsqueeze(0)).float() # 只要是跟原始label相等的，就还需要query，还没有成功
            success = (1 - not_done) * correct * float(epsilon <= goal_epsilon)
            success_query = success * query
        else:
            log.info("Attack failed on {}-th image".format(batch_index))

        if epsilon > goal_epsilon:
            not_done.fill_(1.0)
            success.fill_(0.0)
            success_query = success * query

        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来

def get_exp_dir_name(dataset, norm, targeted, target_type, args):

    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'NES-attack_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        dirname = 'NES-attack-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--samples-per-draw', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=1e-3, help="Sampling variance.")
    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--log-iters', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--save-iters', type=int, default=50)
    parser.add_argument('--plateau-drop', type=float, default=2.0)
    parser.add_argument('--min-lr-ratio', type=int, default=200)
    parser.add_argument('--plateau-length', type=int, default=5)
    parser.add_argument('--imagenet-path', type=str)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--max-lr', type=float, default=None)
    parser.add_argument('--min-lr', type=float, default=5e-5)
    # PARTIAL INFORMATION ARGUMENTS
    parser.add_argument('--top-k', type=int, default=-1, help="if you don't want to use the partial information mode, "
                                                              "just leave this argument to -1 as the default setting."
                                                              "Note that top-k must be set to true class number in the targeted attack.")
    parser.add_argument('--adv-thresh', type=float, default=-1.0)
    # LABEL ONLY ARGUMENTS
    parser.add_argument('--label-only', action='store_true', help="still on developing in progress")
    parser.add_argument('--zero-iters', type=int, default=100, help="how many points to use for the proxy score, which is still on developing")
    parser.add_argument('--label-only-sigma', type=float, default=1e-3, help="distribution width for proxy score, which is still on developing")

    parser.add_argument('--starting-eps', type=float, default=None)
    parser.add_argument('--starting-delta-eps', type=float, default=None)
    parser.add_argument('--min-delta-eps', type=float, default=None)
    parser.add_argument('--conservative', type=int, default=2,
                        help="How conservative we should be in epsilon decay; increase if no convergence")
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10','CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],help='which dataset to use')
    parser.add_argument('--norm', type=str, required=True, choices=["linf","l2"], help='Which lp constraint to update image [linf|l2]')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=["random", "least_likely","increment"])
    parser.add_argument('--json_config', type=str,
                        default='./configures/NES_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    # parser.add_argument("--total-images", type=int, default=1000)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    target_str = "targeted" if args.targeted else "untargeted"
    json_conf = json.load(open(args.json_config))[args.dataset][target_str][args.norm]
    args = vars(args)
    # json_conf = {k: v for k, v in json_conf.items()}
    args.update(json_conf)
    args = SimpleNamespace(**args)
    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 50000
    args.exp_dir = os.path.join(args.exp_dir, get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.test_archs:
        if args.attack_defense:
            log_file_path = os.path.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = os.path.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            log_file_path = os.path.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = os.path.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    DataLoaderMaker.setup_seed(args.seed)
    archs = []
    dataset = args.dataset
    if args.test_archs:
        if dataset == "CIFAR-10" or dataset == "CIFAR-100":
            for arch in MODELS_TEST_STANDARD[dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT,
                    dataset, arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
                else:
                    log.info(test_model_path + " does not exists!")
        elif args.dataset == "TinyImageNet":
            for arch in MODELS_TEST_STANDARD[dataset]:
                test_model_list_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(
                    root=PY_ROOT, dataset=args.dataset, arch=arch)
                test_model_path = list(glob.glob(test_model_list_path))
                if test_model_path and os.path.exists(test_model_path[0]):
                    archs.append(arch)
        else:
            for arch in MODELS_TEST_STANDARD[dataset]:
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
                    PY_ROOT,
                    dataset, arch)
                test_model_list_path = list(glob.glob(test_model_list_path))
                if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
                    continue
                archs.append(arch)
        args.arch = ",".join(archs)
    else:
        archs.append(args.arch)
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("using GPU {}".format(args.gpu))
    log.info("Log file is written in {}".format(os.path.join(args.exp_dir, 'run.log')))
    log.info('Called with args:')
    print_args(args)
    attacker = NES(args.dataset, args.targeted)
    for arch in archs:
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        attacker.attack_all_images(args, arch, model, save_result_path)
        model.cpu()
    log.info("All done!")