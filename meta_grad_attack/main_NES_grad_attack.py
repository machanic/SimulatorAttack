import glob
import sys

from utils.statistics_toolkit import success_rate_and_query_coorelation, success_rate_avg_query

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
import json
import os
import os.path as osp
import random
from types import SimpleNamespace
from dataset.model_constructor import StandardModel
import glog as log
import numpy as np
import torch
from torch.nn import functional as F
from config import CLASS_NUM, PY_ROOT, IN_CHANNELS, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from torch import nn
from meta_grad_regression_auto_encoder.network.autoencoder import AutoEncoder
from optimizer.radam import RAdam


class MetaNESGradAttack(object):
    def __init__(self, auto_encoder, args):
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size)
        self.total_images = args.total_images
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)
        self.inner_lr = args.inner_lr
        self.mse_loss = nn.MSELoss(reduction="mean").cuda()
        self.auto_encoder = auto_encoder.cuda()
        self.auto_encoder.eval()
        self.optimizer = RAdam(self.auto_encoder.parameters(), lr=self.inner_lr)
        self.CONFIDENCE = args.confidence

    def norm(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    ##
    # Projection steps for l2 and linf constraints:
    # All take the form of func(new_x, old_x, epsilon)
    ##
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

    def negative_cw_loss(self, logit, label, target):
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), target]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return torch.clamp(second_max_logit - target_logit + self.CONFIDENCE, min=0.0)
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return torch.clamp(gt_logit - second_max_logit + self.CONFIDENCE, min=0.0)

    def repeat_labels(self, labels, noise_batch_size):# shape = (N,)
        labels = labels.view(-1, 1)  # (N,1)
        labels = labels.repeat(1, noise_batch_size) # (N,B)
        return labels.view(-1)

    def get_grad_by_NES(self, x, sigma, samples_per_draw, noise_batch_size, true_labels, target_labels, target_model):
        num_noise_batches = samples_per_draw // noise_batch_size  # 一共产生多少个samples噪音点，每个batch
        grads = []
        for _ in range(num_noise_batches):
            noise_pos = torch.randn((noise_batch_size // 2,) + (x.size(1), x.size(2), x.size(3)))  # B//2, C, H, W
            noise = torch.cat([-noise_pos, noise_pos], dim=0).cuda()  # B, C, H, W for each image
            # N,1,C,H,W + 1,B,C,H,W = N,B,C,H,W
            eval_points = x.view(x.size(0), 1, noise.size(1), noise.size(2), noise.size(3)) + sigma * noise.unsqueeze(0)
            eval_points = eval_points.view(-1, x.size(1), x.size(2), x.size(3))  # N*B, C,H,W
            logits = target_model(eval_points)  # N*B, num_classes
            # loss shape = (N*B,)  # true_labels shape = (N,) -->
            loss = self.negative_cw_loss(logits, self.repeat_labels(true_labels, noise_batch_size),
                                         self.repeat_labels(target_labels, noise_batch_size))
            loss = loss.view(x.size(0), noise.size(0), 1,1,1) # shape = (N,B,1,1,1)
            grad = torch.mean(loss * noise.unsqueeze(0), dim=1)/sigma # loss shape = (N,B,1,1,1) * (1,B,C,H,W), then mean axis= 1 ==> (N,C,H,W)
            grads.append(grad)
        grads = torch.mean(torch.stack(grads), dim=0)  # (N,C,H,W)
        return grads

    def get_grad_by_backward(self, x, true_labels, target_labels, target_model):
        logits = target_model(x)
        loss_cw_val = self.negative_cw_loss(logits, true_labels, target_labels)
        target_model.zero_grad()
        loss_cw_val.backward()
        gradient_map = x.grad.clone()
        return gradient_map

    def make_adversarial_examples(self, batch_index, images, true_labels, args, target_model):
        '''
        The attack process for generating adversarial examples with priors.
        '''
        with torch.no_grad():
            logit = target_model(images)
        pred = logit.argmax(dim=1)
        query = torch.zeros(args.batch_size).cuda()
        correct = pred.eq(true_labels).float()  # shape = (batch_size,)
        not_done = correct.clone()  # shape = (batch_size,)
        selected = torch.arange(batch_index * args.batch_size,
                                (batch_index + 1) * args.batch_size)  # 选择这个batch的所有图片的index
        if args.targeted:
            if args.target_type == 'random':
                target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset], size=true_labels.size()).long().cuda()
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
        proj_maker = self.l2_proj if args.norm == 'l2' else self.linf_proj  # 调用proj_maker返回的是一个函数
        proj_step = proj_maker(images, args.epsilon)
        # Loss function
        adv_images = images.clone()
        query_count = 0
        t = 0
        while query_count <= args.max_queries:
            if (t + 1) % args.meta_interval == 0:
                if args.method == "meta_attack":
                    gradient = self.get_grad_by_NES(adv_images, args.sigma, args.samples_per_draw, args.noise_batch_size, true_labels,
                                                target_labels, target_model)  # N,C,H,W
                elif args.method == "meta_guided":
                    gradient = self.get_grad_by_backward(adv_images, true_labels, target_labels, target_model)
                self.auto_encoder.train()
                for _ in range(args.finetune_times):
                    predict_gradient = self.auto_encoder(adv_images)
                    self.optimizer.zero_grad()
                    loss = self.mse_loss(predict_gradient, gradient)
                    loss.backward()
                    self.optimizer.step()
                self.auto_encoder.eval()
                query_count += args.samples_per_draw
            else:
                gradient = self.auto_encoder(adv_images)

            adv_images = adv_images - args.image_lr * gradient  # gradient shape = (N,C,H,W)
            adv_images = proj_step(adv_images)
            adv_images = torch.clamp(adv_images, 0, 1)
            with torch.no_grad():
                adv_logit = target_model(adv_images)
            adv_pred = adv_logit.argmax(dim=1)
            adv_prob = F.softmax(adv_logit, dim=1)
            adv_loss = self.negative_cw_loss(adv_logit, true_labels, target_labels)
            if (t + 1) % args.meta_interval == 0:
                query = query + args.samples_per_draw * not_done
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels)).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels).float() # 只要是跟原始label相等的，就还需要query，还没有成功
            t += 1
            success = (1 - not_done) * correct
            success_query = success * query
            not_done_loss = adv_loss * not_done
            not_done_prob = adv_prob[torch.arange(args.batch_size), true_labels] * not_done
            log.info('Attacking image {} - {} / {}, step {}, max query {}'.format(
                batch_index * args.batch_size, (batch_index + 1) * args.batch_size,
                self.total_images, t + 1, int(query.max().item())))
            log.info('        correct: {:.4f}'.format(correct.mean().item()))
            log.info('       not_done: {:.4f}'.format(not_done.mean().item()))
            if success.sum().item() > 0:
                log.info('     mean_query: {:.4f}'.format(success_query[success.byte()].mean().item()))
                log.info('   median_query: {:.4f}'.format(success_query[success.byte()].median().item()))
            if not_done.sum().item() > 0:
                log.info('  not_done_loss: {:.4f}'.format(not_done_loss[not_done.byte()].mean().item()))
                log.info('  not_done_prob: {:.4f}'.format(not_done_prob[not_done.byte()].mean().item()))

            if not not_done.byte().any(): # all success
                break

        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_loss', 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来


    def attack_all_images(self, args, arch_name, target_model, result_dump_path):
        batch_size = args.batch_size
        for batch_idx, (images, true_labels) in enumerate(self.dataset_loader):
            if batch_idx * batch_size >= self.total_images:
                break
            batch_size = images.size(0)
            self.make_adversarial_examples(batch_idx, images.cuda(), true_labels.cuda(), args, target_model)

        query_all_ = self.query_all.detach().cpu().numpy().astype(np.int32)
        not_done_all_ = self.not_done_all.detach().cpu().numpy().astype(np.int32)
        query_threshold_success_rate, query_success_rate = success_rate_and_query_coorelation(query_all_, not_done_all_)
        success_rate_to_avg_query = success_rate_avg_query(query_all_, not_done_all_)
        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.info('     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            log.info('   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
            log.info('     max query: {}'.format(self.success_query_all[self.success_all.byte()].max().item()))
        if self.not_done_all.sum().item() > 0:
            log.info('  avg not_done_loss: {:.4f}'.format(self.not_done_loss_all[self.not_done_all.byte()].mean().item()))
            log.info('  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all.mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().tolist(),
                          "query_threshold_success_rate_dict": query_threshold_success_rate,
                          "query_success_rate_dict": query_success_rate,
                          "success_rate_to_avg_query": success_rate_to_avg_query,
                          "not_done_loss": self.not_done_loss_all[self.not_done_all.byte()].mean().item(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          "args":vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

def get_exp_dir_name(dataset, method, norm, targeted, target_type):
    from datetime import datetime
    dirname = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    dirname = 'meta_NES_grad_of_{}-{}-{}-{}-'.format(method, dataset, norm, target_str) + dirname
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
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--norm', type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument("--method", type=str, default="meta_attack", choices=["meta_guided", "meta_attack"],
                        help="the meta_guided method uses backward to calculate real gradient instead of estimation")
    parser.add_argument('--sigma', type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="cw", choices=["xent", "cw"])
    parser.add_argument('--exploration', type=float,
                        help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument("--meta-interval", type=int, required=True)
    parser.add_argument("--samples-per-draw", type=int, default=50)
    parser.add_argument('--json-configures', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/meta_gradient_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, help='batch size for bandits')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--arch', default="", type=str, help='network architecture')
    parser.add_argument('--test-archs', action="store_true")
    parser.add_argument("--total-images",type=int)
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=["random", "least_likely","increment"])
    parser.add_argument("--confidence", default=0, type=float, help="the attack confidence")
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"
    print("using GPU {}".format(args.gpu))
    if args.json_config:
        assert os.path.exists(args.json_config)
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]  # 不同norm的探索应该有所区别
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)

    if args.targeted:
        args.max_queries = 20000
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.method, args.norm, args.targeted, args.target_type))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)
    set_log_file(osp.join(args.exp_dir, 'run.log'))

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    archs = [args.arch]
    if args.test_archs:
        archs = []
        for arch in MODELS_TEST_STANDARD[args.dataset]:
            if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT,
                    args.dataset, arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
            elif args.dataset == "ImageNet":
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}-*.pth.tar".format(
                    PY_ROOT,
                    args.dataset, arch)
                test_model_path = list(glob.glob(test_model_list_path))
                if test_model_path and os.path.exists(test_model_path[0]):
                    archs.append(arch)
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(osp.join(args.exp_dir, 'run.log')))
    log.info('Called with args:')
    print_args(args)

    auto_encoder_model_path = "{}/train_pytorch_model/{}/{}@{}@TRAIN_I_TEST_II@model_AE@*.tar".format(
        PY_ROOT, args.study_subject, "MetaGradRegression", args.dataset)
    model_path_list = list(glob.glob(auto_encoder_model_path))
    assert model_path_list, "{} does not exists!".format(auto_encoder_model_path)
    auto_encoder_model_path = model_path_list[0]
    auto_encoder = AutoEncoder(IN_CHANNELS[args.dataset])
    auto_encoder.load_state_dict(torch.load(auto_encoder_model_path, map_location=lambda storage, location: storage)["state_dict"])
    log.info("loading auto encoder from {}".format(auto_encoder_model_path))
    attacker = MetaNESGradAttack(auto_encoder, args)
    for arch in archs:
        model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        attacker.attack_all_images(args, arch, model, save_result_path)
        model.cpu()
