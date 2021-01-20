import argparse
import glob
import os
import random
import os.path as osp
import json
import numpy as np
import glog as log
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
sys.path.append("/home/machen/meta_perturbations_black_box_attack")
import torch
from types import SimpleNamespace

from ZOO_attack.zoo_attack import ZOOAttack
from config import MODELS_TEST_STANDARD, PY_ROOT, CLASS_NUM, IN_CHANNELS, IMAGE_SIZE
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel
from utils.statistics_toolkit import success_rate_and_query_coorelation, success_rate_avg_query


class ZooAttackFramework(object):

    def __init__(self, args):
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, 1)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)


    def make_adversarial_examples(self, batch_index, images, true_labels, args,  attacker):
        target_model = attacker.target_model
        batch_size = images.size(0)
        selected = torch.arange(batch_index * batch_size, min((batch_index + 1) * batch_size, self.total_images))  # 选择这个batch的所有图片的index
        if args.targeted:
            if args.target_type == "random":
                with torch.no_grad():
                    logit = target_model(images)
                target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset], size=true_labels.size()).long().cuda()
                invalid_target_index = target_labels.eq(true_labels)
                while invalid_target_index.sum().item() > 0:
                    target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                 size=target_labels[invalid_target_index].shape).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
            elif args.target_type == 'least_likely':
                with torch.no_grad():
                    logit = target_model(images)
                target_labels = logit.argmin(dim=1)
            elif args.target_type == "increment":
                target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
        else:
            target_labels = None
        log.info("Begin attack batch {}!".format(batch_index))
        with torch.no_grad():
            adv_images, stats_info = attacker.attack(images, true_labels, target_labels)
        query = stats_info["query"]
        correct = stats_info["correct"]
        not_done = stats_info["not_done"]
        success = stats_info["success"]
        success_query = stats_info["success_query"]
        not_done_prob = stats_info["not_done_prob"]
        not_done_loss = stats_info["not_done_loss"]
        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_loss', 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来

    def attack_dataset_images(self, args, arch_name, target_model, result_dump_path):
        attacker = ZOOAttack(target_model, IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset][0], args)
        for batch_idx, data_tuple in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet":
                if target_model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            self.make_adversarial_examples(batch_idx, images.cuda(), true_labels.cuda(),
                                                                 args,  attacker)
        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.info(
                '     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            log.info(
                '   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
            log.info('     max query: {}'.format(self.success_query_all[self.success_all.byte()].max().item()))
        if self.not_done_all.sum().item() > 0:
            log.info(
                '  avg not_done_loss: {:.4f}'.format(self.not_done_loss_all[self.not_done_all.byte()].mean().item()))
            log.info(
                '  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_loss": self.not_done_loss_all[self.not_done_all.byte()].mean().item(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item(),
                          'args': vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))




def get_exp_dir_name(dataset, use_tanh, use_log, use_uniform_pick, targeted, target_type):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    use_tanh_str = "tanh" if use_tanh else "no_tanh"
    use_log_str = "log_softmax" if use_log else "no_log"
    randomly_pick_coordinate = "randomly_sample" if use_uniform_pick else "importance_sample"
    dirname = 'ZOO-{}-{}-{}-{}-{}'.format(dataset, use_tanh_str, use_log_str, target_str, randomly_pick_coordinate)
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



def main(args, arch):
    model = StandardModel(args.dataset, arch, args.solver != "fake_zero")
    model.cuda()
    model.eval()
    if args.init_size is None:
        args.init_size = model.input_size[-1]
        log.info("Argument init_size is not set and not using autoencoder, set to image original size:{}".format(
            args.init_size))

    target_str = "untargeted" if not args.targeted else "targeted_{}".format(args.target_type)
    save_result_path = args.exp_dir + "/data_{}@arch_{}@solver_{}@{}_result.json".format(args.dataset,
                                                                                            arch, args.solver,
                                                                                            target_str)
    if os.path.exists(save_result_path):
        model.cpu()
        return
    attack_framework = ZooAttackFramework(args)
    attack_framework.attack_dataset_images(args, arch, model, save_result_path)
    model.cpu()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="the batch size for zoo, zoo_ae attack")
    parser.add_argument("-c", "--init_const", type=float, default=0.0, help="the initial setting of the constant lambda")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["CIFAR-10", "CIFAR-100", "TinyImageNet", "ImageNet", "MNIST", "FashionMNIST"])
    parser.add_argument("-m", "--max_iterations", type=int, default=10000, help="set 0 to use the default value")
    parser.add_argument("-p", "--print_every", type=int, default=100,
                        help="print information every PRINT_EVERY iterations")
    parser.add_argument("--binary_steps", type=int, default=1)
    parser.add_argument("--targeted", action="store_true",
                        help="the type of attack")
    parser.add_argument("--target_type", type=str, default="increment", choices=['random', 'least_likely',"increment"],
                        help="if set, choose random target, otherwise attack every possible target class, only works when using targeted")
    parser.add_argument("-o", "--early_stop_iters", type=int, default=100,
                        help="print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--confidence", default=0, type=float, help="the attack confidence")

    parser.add_argument("--lr", default=None,type=float)
    parser.add_argument("--abort_early", action="store_true")
    parser.add_argument("--init_size", default=None, type=int, help = "starting with this size when --use_resize")
    parser.add_argument("--resize", action="store_true",
                        help="this option only works for the preprocess resize of images")

    parser.add_argument("-r", "--reset_adam", action='store_true', help="reset adam after an initial solution is found")
    parser.add_argument("--solver", choices=["adam", "newton", "adam_newton", "fake_zero"], default="adam")
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--arch', type=str, help='network architecture')
    parser.add_argument('--exp_dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument("--start_iter", default=0, type=int,
                        help="iteration number for start, useful when loading a checkpoint")
    parser.add_argument("--use_tanh", action="store_true")
    parser.add_argument("--use_log", action='store_true')
    parser.add_argument("--epsilone", type=float)
    parser.add_argument('--seed', default=1216, type=int, help='random seed')
    parser.add_argument('--json_config', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/configures/zoo_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument("--uniform", action='store_true', help="disable importance sampling")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    os.environ["TORCH_HOME"] = "/home1/machen/.cache/torch/pretrainedmodels"
    json_conf = json.load(open(args.json_config))[args.dataset]
    args = vars(args)
    json_conf = {k: v for k, v in json_conf.items() if k not in args or args[k] is None}
    args.update(json_conf)
    args = SimpleNamespace(**args)
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.use_tanh, args.use_log, args.uniform,
                                                                 args.targeted, args.target_type))
    os.makedirs(args.exp_dir, exist_ok=True)
    if args.test_archs:
        set_log_file(os.path.join(args.exp_dir, 'run.log'))
    else:
        set_log_file(os.path.join(args.exp_dir, 'run_{}.log'.format(args.arch)))

    args.abort_early = True
    if args.targeted:
        args.max_iterations = 20000
    if args.init_const == 0.0:
        if args.binary_steps != 0:
            args.init_const = 0.01
        else:
            args.init_const = 0.5

    random.seed(args.seed)
    np.random.seed(args.seed)
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
            for arch in MODELS_TEST_STANDARD[args.dataset]:
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
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(osp.join(args.exp_dir, 'run.log')))
    log.info('Called with args:')
    print_args(args)
    for arch in archs:
        main(args, arch)



