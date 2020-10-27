import json
import glob
import numpy as np
import os
import torch
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from torch import nn
from MGA_attack.mi_fgsm import MI_FGSM_ENS
import argparse
from config import CLASS_NUM, PY_ROOT, MODELS_TEST_STANDARD, MODELS_TRAIN_STANDARD, \
    MODELS_TRAIN_WITHOUT_RESNET
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel
from dataset.defensive_model import DefensiveModel
from types import SimpleNamespace
import glog as log
from torch.nn import functional as F


class MGAAttack(object):
    def __init__(self, pop_size=5, generations=1000, cross_rate=0.7,
                 mutation_rate=0.001, max_queries=2000,
                 epsilon=8. / 255, iters=10, ensemble_models=None, targeted=False):
        self.loss_fn = nn.CrossEntropyLoss()
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, 1)
        self.total_images = len(self.dataset_loader.dataset)
        # parameters about evolution algorithm
        self.pop_size = pop_size
        self.generations = generations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        # parameters about attack
        self.epsilon = epsilon
        self.clip_min = 0
        self.clip_max = 1
        # ensemble MI-FGSM parameters, use ensemble MI-FGSM attack generate adv as initial population
        self.ensemble_models = ensemble_models
        self.iters = iters
        self.targeted = targeted
        self.max_queries = max_queries
        self.idx = np.random.choice(np.arange(self.pop_size), size=2, replace=False)
        self.is_change = np.zeros(self.pop_size)
        self.pop_fitness = np.zeros(self.pop_size)

        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)

    def is_success(self, logits, y):
        label = logits.argmax(dim=1).item()
        if self.targeted and label == y[0].item():
            return True
        elif not self.targeted and label != y[0].item():
            return True
        return False

    def fitness_helper(self, model, individual, x, y):

        # resize to image size
        individual = individual.copy()
        zeros = (individual == 0)
        individual[zeros] = -1

        delta = individual * self.epsilon
        delta = torch.from_numpy(delta).to(dtype=x.dtype, device=x.device)

        adv = x + delta.unsqueeze(0)
        adv = torch.clamp(adv, self.clip_min, self.clip_max)
        # only imagenet dataset needs preprocess
        logits = model(adv)
        loss = self.loss_fn(logits, y)
        if self.is_success(logits, y):
            self.adv = adv.detach().cpu()
        return loss.item()

    def get_fitness(self, model, lw, x, y, query):  # lw的shape = (2,C,H,W)
        first, second = self.idx[0], self.idx[1]
        if self.is_change[first] == 1:  # losser changed, so fitness also change
            f1 = self.fitness_helper(model, lw[0], x, y)
            query += 1
            self.pop_fitness[first] = f1
            self.is_change[first] = 0
        else:
            f1 = self.pop_fitness[first]

        if self.is_change[second] == 1:
            f2 = self.fitness_helper(model, lw[1], x, y)
            query += 1
            self.pop_fitness[second] = f2
            self.is_change[second] = 0
        else:
            f2 = self.pop_fitness[second]

        return np.array([f1, f2])

    def cross_over(self, lw):
        cross_point = np.random.rand(lw.shape[-3], lw.shape[-2], lw.shape[-1]) < self.cross_rate
        lw[0, cross_point] = lw[1, cross_point]  # 输家70%的概率从赢家拿perturbation给输家
        return lw

    def mutate(self, lw):

        # generate mutation point
        mutation_point = np.random.rand(lw.shape[-3], lw.shape[-2], lw.shape[-1]) < self.mutation_rate
        # reverse the value at mutation point 1->0, 0->1
        zeros = (lw[0] == 0)
        ones = (lw[0] == 1)
        lw[0, mutation_point & zeros] = 1  # 被修改的是输家
        lw[0, mutation_point & ones] = 0
        return lw

    def init_pop(self, x, y):

        adversary = MI_FGSM_ENS(self.ensemble_models, epsilon=self.epsilon, iters=self.iters, targeted=self.targeted)
        datas, labels = x.repeat((self.pop_size, 1, 1, 1)), y.repeat(self.pop_size)  # pop size = 5
        adv = adversary.perturb(datas, labels)
        delta = adv - x
        negative = delta <= 0
        positive = delta > 0
        delta[negative] = 0
        delta[positive] = 1
        return delta.detach().cpu().numpy()

    def make_adversarial_examples(self, model, images, labels):
        """
        :param images: the original images
        :param labels: the original ground truth label in untargeted attack and the target class in targeted attack
        :return: adversarial_images, query_number
        """
        images, labels = images.cuda(), labels.cuda()
        # input train_data parameter
        batch_size, channels, height, width = images.size()
        adv = images.clone()
        self.adv = None
        query = torch.zeros(images.size(0))
        self.pop_fitness = np.zeros(self.pop_size)
        self.is_change = np.zeros(self.pop_size)
        if not self.ensemble_models:
            # initial population
            pop = np.random.randint(0, 2, (self.pop_size, channels, height, width))
        else:
            pop = self.init_pop(images, labels)
        # this expense 5 queries, this thy the median always 5
        # init pop fitness, this can reduce query, cause in mga, not all individual changes in a generation
        for n in range(self.pop_size):
            self.pop_fitness[n] = self.fitness_helper(model, pop[n], images, labels)  # 根据每个初始化的pop得到loss
            query+=1
        for i in range(self.generations):  # 1000次循环
            self.idx = np.random.choice(np.arange(self.pop_size), size=2, replace=False)  # 从[0-4]随机生成2个数字
            lw = pop[self.idx].copy()  # short for losser winner
            fitness = self.get_fitness(model, lw, images, labels, query)
            # if success, abort early
            if self.adv is not None:
                return self.adv, query
            # in target situation, the smaller fitness is, the better
            if self.targeted:
                fidx = np.argsort(-fitness)  # loss从大到小，由于targeted attack的loss越小越好，因此第一个位置是输家
            else:
                fidx = np.argsort(fitness)  # loss从小到大，由于untargeted attack的loss越大越好，因此第一个位置是输家
            lw = lw[fidx]
            lw = self.cross_over(lw)  # 只修改那个输家，也就是位置=0的
            lw = self.mutate(lw)  # 只修改那个输家，也就是位置=0的

            lw = lw[fidx]
            # update population
            pop[self.idx] = lw.copy()

            # losser changed, so fitness should also change
            self.is_change[self.idx[fidx[0]]] = 1
            if query[0].item() >= self.max_queries:  # 失败了
                delta = lw[1] * self.epsilon
                delta = torch.from_numpy(delta).to(dtype=images.dtype, device=images.device)
                adv = images + delta.unsqueeze(0)
                adv = torch.clamp(adv, self.clip_min, self.clip_max).detach().cpu()
                break
        return adv, query

    def attack_all_images(self, args, arch_name, target_model, result_dump_path):

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
            with torch.no_grad():
                logit = target_model(images)
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels).float()
            not_done = correct.clone()
            if self.targeted:
                adv_images, query = self.make_adversarial_examples(target_model, images, target_labels)
            else:
                adv_images, query = self.make_adversarial_examples(target_model, images, true_labels)
            adv_images = adv_images.cuda()
            with torch.no_grad():
                adv_logit = target_model(adv_images)
            adv_pred = adv_logit.argmax(dim=1)
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(
                    target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels).float()  # 只要是跟原始label相等的，就还需要query，还没有成功
            success = (1 - not_done) * correct
            success = success.detach().cpu()
            success_query = success * query


            log.info('Attack {}-th image over, query:{}, succes: {}'.format(
                batch_idx, int(query[0].item()), bool(success[0].item())
            ))
            log.info('        correct: {:.4f}'.format(correct.mean().item()))
            log.info('       not_done: {:.4f}'.format(not_done[correct.byte()].mean().item()))
            if success.sum().item() > 0:
                log.info('     mean_query: {:.4f}'.format(success_query[success.byte()].mean().item()))
                log.info('   median_query: {:.4f}'.format(success_query[success.byte()].median().item()))
            selected = torch.arange(batch_idx * 1,
                                    min((batch_idx + 1) * 1, self.total_images))
            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query']:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()  #
        self.not_done_all[(self.query_all > args.max_queries).byte()] = 1
        self.success_all[(self.query_all > args.max_queries).byte()] = 0
        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.info(
                '     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            log.info(
                '   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
            log.info('     max query: {}'.format(self.success_query_all[self.success_all.byte()].max().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.byte()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def get_exp_dir_name(dataset, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.ensemble_models:
        if args.attack_defense:
            dirname = 'MGA_attack_ensemble_models_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
        else:
            dirname = 'MGA_attack_ensemble_models-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        if args.attack_defense:
            dirname = 'MGA_attack_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
        else:
            dirname = 'MGA_attack-{}-{}-{}'.format(dataset, norm, target_str)
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

    parser.add_argument('--ensemble_models', action='store_true')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epsilon', type=float, default=0.03137)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--pop_size', type=int, default=5)
    parser.add_argument('--mr', type=float, default=0.001)
    parser.add_argument('--cr', type=float, default=0.7)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--targeted', default=False, action='store_true')
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--json-config', type=str,
                        # default='/home1/machen/meta_perturbations_black_box_attack/configures/MGA_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    args = parser.parse_args()
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

    # defense method
    # if args.defense_method:
    #     defense_method = args.defense_method+"()"
    #     input_trans = eval(defense_method)
    # else:
    #     input_trans = None
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
    models = []
    if args.ensemble_models:
        train_model_names = {"CIFAR-10": ["densenet-bc-100-12", "vgg19_bn", "resnet-110"],
                        "CIFAR-100": ["densenet-bc-100-12", "vgg19_bn", "resnet-110"],
                        "TinyImageNet": ["vgg19_bn","resnet101","resnet152"], "ImageNet": ["densenet121","dpn68","resnext101_32x4d"]}


        for surr_arch in train_model_names[args.dataset]:
            if surr_arch in archs:
                continue
            surrogate_model = StandardModel(args.dataset, surr_arch, False)
            surrogate_model.eval()
            models.append(surrogate_model)

    args.exp_dir = os.path.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, "linf", args.targeted, args.target_type,
                                             args))  # 随机产生一个目录用于实验
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
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    attacker = MGAAttack(pop_size=args.pop_size, generations=50000, cross_rate=args.cr, targeted=args.targeted,
                        mutation_rate=args.mr, max_queries=args.max_queries,
                        epsilon=args.epsilon, iters=args.iters, ensemble_models=models)
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
        attacker.attack_all_images(args, arch, model, save_result_path)
        model.cpu()

