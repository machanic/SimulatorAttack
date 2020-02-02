import glob
import sys
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
from config import CLASS_NUM, PY_ROOT, CIFAR_ALL_MODELS, IMAGENET_ALL_MODELS, IN_CHANNELS, IMAGE_SIZE, \
    MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from torch import nn
from meta_grad_regression_auto_encoder.network.autoencoder import AutoEncoder
from optimizer.radam import RAdam

class MetaZooGradAttack(object):
    def __init__(self, auto_encoder, args):
        assert args.batch_size == 1
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)
        self.finetune_lr = args.finetune_lr
        self.mse_loss = nn.MSELoss(reduction="mean").cuda()
        self.auto_encoder = auto_encoder.cuda()
        self.auto_encoder.eval()
        self.optimizer = RAdam(self.auto_encoder.parameters(), lr=self.finetune_lr)
        self.num_channels = IN_CHANNELS[args.dataset]
        self.img_size = IMAGE_SIZE[args.dataset]
        self.targeted = args.targeted

        # the following variables are used for estimating gradient by ZOO
        self.real_modifier = np.zeros((1, self.num_channels, self.img_size[0], self.img_size[1]), dtype=np.float32)
        self.var_size = self.num_channels * self.img_size[0] * self.img_size[1]
        self.use_var_len = self.var_size
        self.var_list  = np.array(range(0, self.use_var_len),dtype=np.int32)
        self.num_top_q = args.top_q
        # self.INIT_CONST = args.init_const
        self.CONFIDENCE = args.confidence
        self.l2dist_loss = lambda newimg, timg: torch.sum((newimg - timg).pow(2), (1,2,3))

    def norm(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

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

    def get_newimg(self, timg, img_modifier):
        # img_modifier shape = (2B+1, C, H, W), timg shape = (1,C,H,W)
        self.modifier_up = 1.0 - timg
        self.modifier_down = 0.0 - timg
        cond1 = torch.gt(img_modifier, self.modifier_up).float()
        cond2 = torch.le(img_modifier, self.modifier_up).float()
        cond3 = torch.gt(img_modifier, self.modifier_down).float()
        cond4 = torch.le(img_modifier, self.modifier_down).float()
        img_modifier = torch.mul(cond1, self.modifier_up) + torch.mul(torch.mul(cond2, cond3),
                             img_modifier) + torch.mul(cond4, self.modifier_down)
        newimg = img_modifier + timg
        return newimg

    def get_grad_by_ZOO(self, timg, true_label, target_label, last_iter_grad_map, target_model):
        assert last_iter_grad_map.dim() == 4
        # shape of last_iter_grad must be (1,C,H,W) not (1,top_q)!
        var = np.repeat(self.real_modifier, self.num_top_q * 2 + 1, axis=0)  # estimate top q gradients on each image, 2B,C,H,W
        batch_size = timg.shape[0]
        assert batch_size == 1
        _, top_q_indexes = torch.topk(last_iter_grad_map.view(last_iter_grad_map.size(0), -1), args.top_q, dim=1)   # shape = N, top_q
        last_iter_grad_top_indexes = top_q_indexes.view(-1).detach().cpu().numpy() # 上一轮的梯度选择最大的q个坐标位置
        for i in range(self.num_top_q):
            var[i * 2 + 1].reshape(-1)[last_iter_grad_top_indexes[i]] += 0.0001
            var[i * 2 + 2].reshape(-1)[last_iter_grad_top_indexes[i]] -= 0.0001
        modifier = torch.from_numpy(var).cuda()
        newimg = self.get_newimg(timg, modifier)  # (2B+1, C, H, W)
        with torch.no_grad():
            logits = target_model(newimg)
        losses = self.negative_cw_loss(logits, true_label, target_label)  # shape = (batch_size,)
        grad = torch.zeros(self.num_top_q).cuda().float()
        for i in range(self.num_top_q):
            grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
        grad = grad.view(batch_size, -1)
        return grad, top_q_indexes

    def get_grad_by_backward(self, adv_images, true_label, target_label, last_iter_grad_map, target_model):
        batch_size = adv_images.size(0)
        assert batch_size == last_iter_grad_map.size(0)
        logits = target_model(adv_images)
        loss_cw_val = self.negative_cw_loss(logits, true_label, target_label)
        target_model.zero_grad()
        loss_cw_val.backward()
        gradient_map = adv_images.grad.clone()
        _, top_q_indexes = torch.topk(last_iter_grad_map.view(last_iter_grad_map.size(0), -1), args.top_q, dim=1)
        gradient = gradient_map.view(batch_size, -1)
        gradient = torch.gather(gradient,1,top_q_indexes)
        return gradient, top_q_indexes


    def make_adversarial_examples(self, batch_index, images, true_labels, args, target_model):
        '''
        The attack process for generating adversarial examples with priors.
        '''
        self.real_modifier.fill(0)
        with torch.no_grad():
            logit = target_model(images)
        pred = logit.argmax(dim=1)
        query = torch.zeros(images.size(0)).cuda()
        correct = pred.eq(true_labels).float()  # shape = (batch_size,)
        not_done = correct.clone()  # shape = (batch_size,)
        selected = torch.arange(batch_index * args.batch_size,
                                (batch_index + 1) * args.batch_size)  # 选择这个batch的所有图片的index
        if args.targeted:
            if args.target_type == 'random':
                target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset], size=true_labels.size()).long().cuda()
            elif args.target_type == 'least_likely':
                target_labels = logit.argmin(dim=1)
            invalid_target_index = target_labels.eq(true_labels)
            while invalid_target_index.sum().item() > 0:
                target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                    size=target_labels[
                                                                        invalid_target_index].shape).long().cuda()
                invalid_target_index = target_labels.eq(true_labels)
        else:
            target_labels = None
        proj_maker = self.l2_proj if args.norm == 'l2' else self.linf_proj  # 调用proj_maker返回的是一个函数
        proj_step = proj_maker(images, args.epsilon)
        not_success_images = None
        last_gradient_map = None
        adv_images = images.clone()
        # upper_bound = 1e10
        # o_bestl2 = 1e10
        # bestl2 = 1e10
        query_count = 0
        t = 0
        while query_count <= args.max_queries:
            if (t + 1) % args.meta_interval == 0:
                assert last_gradient_map is not None
                # gradient shape = (N, top_q), I_t = (N,top_q)
                if args.method == "meta_attack":
                    gradient, I_t = self.get_grad_by_ZOO(images, true_labels, target_labels, last_gradient_map, target_model)
                elif args.method == "meta_guided":
                    gradient, I_t = self.get_grad_by_backward(adv_images, true_labels, target_labels, last_gradient_map,target_model)
                self.auto_encoder.train()
                for _ in range(args.finetune_times):
                    predict_gradient = self.auto_encoder(adv_images)
                    self.optimizer.zero_grad()
                    predict_gradient_top = torch.gather(predict_gradient.view(predict_gradient.size(0),-1), 1, index=I_t)  # (N, top_q)
                    loss = self.mse_loss(predict_gradient_top, gradient)
                    loss.backward()
                    self.optimizer.step()
                self.auto_encoder.eval()
                query_count += self.num_top_q * 2
            else:
                gradient_map = self.auto_encoder(adv_images)
                gradient, I_t = torch.topk(gradient_map.view(gradient_map.size(0), -1), self.num_top_q, dim=1)  # shape = N, top_q
                last_gradient_map = gradient_map.clone().detach()

            # 2-dimensional update modifier
            I_t_np = I_t.detach().cpu().numpy()  # shape = (N, top_q)
            old_val = np.take_along_axis(self.real_modifier, I_t_np, axis=1)  # (N,top_q)
            old_val -= args.image_lr * gradient.detach().cpu().numpy()   # (N,top_q)
            np.put_along_axis(self.real_modifier, I_t_np, old_val, axis=1)

            # m = self.real_modifier.reshape(-1)
            # I_t_flatten = I_t.detach().cpu().numpy().reshape(-1)
            # old_val = m[I_t_flatten]
            # old_val -= args.image_lr * gradient.detach().cpu().numpy().reshape(-1)  # 注意是积累式的
            # m[I_t_flatten] = old_val

            adv_images = self.get_newimg(images, torch.from_numpy(self.real_modifier).cuda())


            # update_value = torch.gather(adv_images.view(adv_images.size(0), -1), 1, index=I_t) + args.image_lr * gradient
            # adv_images.view(adv_images.size(0),-1).scatter_(1, I_t, update_value)
            adv_images = proj_step(adv_images)
            adv_images = torch.clamp(adv_images, 0, 1)

            with torch.no_grad():
                adv_logit = target_model(adv_images)
            adv_pred = adv_logit.argmax(dim=1)
            adv_prob = F.softmax(adv_logit, dim=1)
            adv_loss = self.negative_cw_loss(adv_logit, true_labels, target_labels)
            if (t + 1) % args.meta_interval == 0:
                query = query + self.num_top_q * 2 * not_done
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels)).float()
            else:
                not_done = not_done * adv_pred.eq(true_labels).float()
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
        else:
            not_success_images = images[not_done.byte()].detach().cpu()

        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_loss', 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来

        return not_success_images


    def attack_all_images(self, args, arch_name, target_model, result_dump_path):

        not_success_images_list = []
        batch_size = args.batch_size
        for batch_idx, (images, true_labels) in enumerate(self.dataset_loader):
            if batch_idx * batch_size >= self.total_images:
                break
            batch_size = images.size(0)
            not_success_images = self.make_adversarial_examples(batch_idx, images.cuda(), true_labels.cuda(), args, target_model)
            if not_success_images is not None:
                not_success_images_list.append(not_success_images)
        if not_success_images_list:
            all_not_success_images = torch.cat(not_success_images_list, 0).detach().cpu().numpy()

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
                          "not_done_loss": self.not_done_loss_all[self.not_done_all.byte()].mean().item(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item()}
        meta_info_dict['args'] = vars(args)
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, indent=4, sort_keys=True)
        save_npy_path = os.path.dirname(result_dump_path) + "/{}_attack_not_success_images.npy".format(arch)
        if not_success_images_list:
            np.save(save_npy_path, all_not_success_images)
        log.info("done, write stats info to {}".format(result_dump_path))

def get_exp_dir_name(dataset, method, norm, targeted, target_type):
    from datetime import datetime
    dirname = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    dirname = 'meta_ZOO_grad_of_{}-{}-{}-{}-'.format(method, dataset, norm, target_str) + dirname
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
    parser.add_argument('--meta-interval',type=int,default=None)
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, default=4e-3, help='Learning rate for the image (iterative attack)')
    parser.add_argument("--method",type=str, default="meta_attack", choices=["meta_guided","meta_attack"],help="the meta_guided method uses ")
    parser.add_argument('--finetune-lr',type=float, default=1e-2)
    parser.add_argument('--norm', type=str, required=True, choices=["l2","linf"], help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument("--meta-interval", type=int, required=True)
    parser.add_argument("-c", "--init_const", type=float, default=1, help="the initial setting of the constant lambda")
    parser.add_argument("--confidence", default=0, type=float, help="the attack confidence")
    parser.add_argument('--json-config', type=str, default='/home1/machen/meta_perturbations_black_box_attack/meta_gradient_attack_conf.json',
                        help='a config file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, help='batch size for bandits')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to be attacked')
    parser.add_argument('--arch', default="", type=str, help='network architecture')
    parser.add_argument('--top-q',type=int, default=None, help='the top q coordinates to estimate gradients')
    parser.add_argument('--test-archs', action="store_true")
    parser.add_argument("--total-images",type=int)
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='random', choices=["random", "least_likely"])
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument("--confidence", type=int,default=0)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument("--study_subject", type=str, default="meta_grad_regression")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"
    print("using GPU {}".format(args.gpu))
    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]  # 不同norm的探索应该有所区别
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
    if args.targeted:
        args.max_queries = 50000
    if args.method == "meta_attack":
        assert args.batch_size == 1
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
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(PY_ROOT,
                                                                                        args.dataset,  arch)
            elif args.dataset == "ImageNet":
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}-*.pth.tar".format(
                    PY_ROOT,
                    args.dataset, arch)
                test_model_path = list(glob.glob(test_model_list_path))
            if test_model_path and os.path.exists(test_model_path):
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
    attacker = MetaZooGradAttack(auto_encoder, args)
    for arch in archs:
        model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        attacker.attack_all_images(args, arch, model, save_result_path)
        model.cpu()
