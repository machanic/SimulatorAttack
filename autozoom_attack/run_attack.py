import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
import glob
import json
import os
import os.path as osp
import random
import glog as log
import numpy as np
import torch
from torch.nn import functional as F
from autozoom_attack.autozoom_attack import ZOO, ZOO_AE, AutoZOOM_BiLIN, AutoZOOM_AE
from autozoom_attack.codec import Codec
from config import IN_CHANNELS, CLASS_NUM, PY_ROOT, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel


class AutoZoomAttackFramework(object):

    def __init__(self, args):
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args["dataset"], 1)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)

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

    def make_adversarial_examples(self, batch_index, images, true_labels, args, attacker, target_model, codec):
        if args["attack_method"] == "zoo_ae" or args["attack_method"] == "autozoom_ae":
            # log ae info
            decode_img = codec(images)
            diff_img = (decode_img - images)
            diff_mse = torch.mean(diff_img.view(-1).pow(2)).item()
            log.info("[AE] MSE:{:.4f}".format(diff_mse))

        batch_size = 1
        selected = torch.arange(batch_index * batch_size,
                                (batch_index + 1) * batch_size)  # 选择这个batch的所有图片的index
        if args["attack_type"] == "targeted":

            if args["target_type"] == "random":
                with torch.no_grad():
                    logit = target_model(images)
                target_labels = torch.randint(low=0, high=CLASS_NUM[args["dataset"]], size=true_labels.size()).long().cuda()
                invalid_target_index = target_labels.eq(true_labels)
                while invalid_target_index.sum().item() > 0:
                    target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                        size=target_labels[
                                                                            invalid_target_index].shape).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
            elif args["target_type"] == 'least_likely':
                with torch.no_grad():
                    logit = target_model(images)
                target_labels = logit.argmin(dim=1)
            else:
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
        adv_logit = stats_info["adv_logit"]
        adv_loss = self.cw_loss(adv_logit, true_labels, target_labels)
        not_done_loss = adv_loss * not_done
        for key in ['query', 'correct',  'not_done',
                    'success', 'success_query', 'not_done_loss', 'not_done_prob']:
            value_all = getattr(self, key+"_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来


    def attack_dataset_images(self, args, attacker, arch_name, target_model, codec, result_dump_path):
        for batch_idx, data_tuple in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet":
                if target_model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]

            if images.size(-1) != target_model.input_size[-1]:
                images = F.interpolate(images, size=target_model.input_size[-1], mode='bilinear',align_corners=True)
            self.make_adversarial_examples(batch_idx, images.cuda(), true_labels.cuda(),
                                                                 args, attacker, target_model, codec)

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
                          "avg_not_done": self.not_done_all.mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "not_done_loss": self.not_done_loss_all[self.not_done_all.byte()].mean().item(),
                          "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item()}
        meta_info_dict['args'] = vars(args)
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, indent=4, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def main(args, arch):
    model = StandardModel(args["dataset"], arch, False)
    model.cuda()
    model.eval()
    # attack related settings
    if args["attack_method"] == "zoo" or args["attack_method"] == "autozoom_bilin":
        if args["img_resize"] is None:
            args["img_resize"] = model.input_size[-1]
            log.info("Argument img_resize is not set and not using autoencoder, set to image original size:{}".format(
                args["img_resize"]))

    codec = None
    if args["attack_method"] == "zoo_ae" or args["attack_method"] == "autozoom_ae":
        codec = Codec(model.input_size[-1], IN_CHANNELS[args["dataset"]], args["compress_mode"], args["resize"], use_tanh=args["use_tanh"])
        codec.load_codec(args["codec_path"])
        codec.cuda()
        decoder = codec.decoder
        args["img_resize"] = decoder.input_shape[1]
        log.info("Loading autoencoder: {}, set the attack image size to:{}".format(args["codec_path"], args["img_resize"]))

    # setup attack
    if args["attack_method"] == "zoo":
        blackbox_attack = ZOO(model, args["dataset"], args)
    elif args["attack_method"] == "zoo_ae":
        blackbox_attack = ZOO_AE(model, args["dataset"], args, decoder)
    elif args["attack_method"] == "autozoom_bilin":
        blackbox_attack = AutoZOOM_BiLIN(model, args["dataset"], args)
    elif args["attack_method"] == "autozoom_ae":
        blackbox_attack = AutoZOOM_AE(model, args["dataset"], args, decoder)
    target_str = "untargeted" if  args["attack_type"]!="targeted" else "targeted_{}".format(args["target_type"])
    save_result_path = args["exp_dir"] + "/data_{}@arch_{}@attack_{}@{}_result.json".format(args["dataset"],
                                        arch, args["attack_method"], target_str)
    attack_framework = AutoZoomAttackFramework(args)
    attack_framework.attack_dataset_images(args, blackbox_attack, arch, model, codec, save_result_path)
    model.cpu()


def get_exp_dir_name(dataset, method, targeted, target_type):
    from datetime import datetime
    dirname = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    dirname = 'AutoZOOM-{}-{}-{}-'.format(dataset, method, target_str) + dirname
    return dirname

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def print_args(args):
    keys = sorted(args.keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args[key]))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("-a", "--attack_method", type=str, required=True,
                        choices=["zoo", "zoo_ae", "autozoom_bilin", "autozoom_ae"], help="the attack method")
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="the batch size for zoo, zoo_ae attack")
    parser.add_argument("-c", "--init_const", type=float, default=1, help="the initial setting of the constant lambda")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["CIFAR-10", "CIFAR-100", "ImageNet", "MNIST", "FashionMNIST"])
    parser.add_argument("-m", "--max_iterations", type=int, default=None, help="set 0 to use the default value")
    parser.add_argument("-p", "--print_every", type=int, default=100,
                        help="print information every PRINT_EVERY iterations")
    parser.add_argument("--attack_type", default="untargeted", choices=["targeted", "untargeted"],
                        help="the type of attack")
    parser.add_argument("--early_stop_iters", type=int, default=100,
                        help="print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    parser.add_argument("--confidence", default=0, type=float, help="the attack confidence")
    parser.add_argument("--codec_path", default=None, type=str, help="the coedec path, load the default codec is not set")
    parser.add_argument("--target_type", type=str, default="increment",  choices=['random', 'least_likely',"increment"],
                        help="if set, choose random target, otherwise attack every possible target class, only works when ATTACK_TYPE=targeted")
    parser.add_argument("--num_rand_vec", type=int, default=1,
                        help="the number of random vector for post success iteration")
    parser.add_argument("--img_offset", type=int, default=0,
                        help="the offset of the image index when getting attack data")
    parser.add_argument("--img_resize", default=None, type=int,
                        help="this option only works for ATTACK METHOD zoo and autozoom_bilin")
    parser.add_argument("--epsilone", type=float, help="the maximum threshold of L2 constraint")
    parser.add_argument("--resize", default=None,type=int, help="this option only works for the preprocess resize of images")
    parser.add_argument("--switch_iterations", type=int, default=None,
                        help="the iteration number for dynamic switching")
    parser.add_argument("--compress_mode", type=int, default=None,
                        help="specify the compress mode if autoencoder is used")
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--arch',  type=str, help='network architecture')
    parser.add_argument('--exp_dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--json-config', type=str,
                        default='./configures/AutoZOOM.json',
                        help='a configures file to be passed in instead of arguments')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    args = vars(args)
    if args["json_config"]:
        json_data = json.load(open(args["json_config"]))
        json_conf = json_data[args["dataset"]]
        json_conf = {k: v for k, v in json_conf.items() if k not in args or args[k] is None}
        args.update(json_conf)
        method_conf = json_data[args["attack_method"]]
        method_conf = {k:v for k,v in method_conf.items() if k not in args or args[k] is None}
        args.update(method_conf)
    args["codec_path"] = list(glob.glob(args["codec_path"].format(PY_ROOT)))[0]

    if args["img_resize"] is not None:
        if args["attack_method"] == "zoo_ae" or args["attack_method"] == "autozoom_ae":
            log.info("Attack method {} cannot use option img_resize, arugment ignored".format(args["attack_method"]))

    if args["attack_type"] == "targeted" and args["max_iterations"] < 20000:
            args["max_iterations"] = 5 * args["max_iterations"]

    args["exp_dir"] = osp.join(args["exp_dir"], get_exp_dir_name(args['dataset'],  args['attack_method'],
                                                                 args["attack_type"] == "targeted", "random"))
    os.makedirs(args["exp_dir"], exist_ok=True)
    set_log_file(osp.join(args["exp_dir"], 'run.log'))
    log.info("using GPU :{}".format(args["gpu"]))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(osp.join(args["exp_dir"], 'run.log')))
    log.info('Called with args:')

    # setup random seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args["seed"])


    archs = [args["arch"]]
    dataset = args["dataset"]
    if args["test_archs"]:
        archs.clear()
        if dataset == "CIFAR-10" or dataset == "CIFAR-100":
            for arch in MODELS_TEST_STANDARD[dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT,
                    dataset, arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
                else:
                    log.info(test_model_path + " does not exists!")
        else:
            for arch in MODELS_TEST_STANDARD[dataset]:
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth.tar".format(
                    PY_ROOT,
                    dataset, arch)
                test_model_list_path = list(glob.glob(test_model_list_path))
                if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
                    continue
                archs.append(arch)
        args["arch"] = ",".join(archs)
    print_args(args)
    for arch in archs:
        main(args, arch)


