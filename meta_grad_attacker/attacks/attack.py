import copy
import glob
import os
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import time
from torch.nn import functional as F
import glog as log
import json
import numpy as np
import torch
from utils.statistics_toolkit import success_rate_and_query_coorelation, success_rate_avg_query
from config import MODELS_TEST_STANDARD, PY_ROOT, CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.model_constructor import StandardModel
from meta_grad_attacker.attacks.black_box_attack import MetaGradL2Attack
from meta_grad_attacker.attacks.options import get_parse_args
from meta_grad_attacker.meta_training.load_attacked_and_meta_model import load_meta_model

def get_expr_dir_name(dataset, targeted, target_type, use_tanh):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    use_tanh_str = "tanh" if use_tanh else "no_tanh"
    dirname = 'MetaGradAttack_{}_{}_{}'.format(dataset,target_str, use_tanh_str)
    return dirname
def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def main(args, result_dir_path):
    log.info('Loading %s model and test data' % args.dataset)
    dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, 1)
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
    meta_model_path =  '{}/train_pytorch_model/meta_grad_regression/{}.pth.tar'.format(PY_ROOT, args.dataset)
    assert os.path.exists(meta_model_path), "{} does not exist!".format(meta_model_path)
    meta_model = load_meta_model(meta_model_path)
    log.info("Load meta model from {}".format(meta_model_path))
    attack = MetaGradL2Attack(args,
                              targeted=args.targeted, search_steps=args.binary_steps,
                              max_steps=args.maxiter, use_log=not args.use_zvalue, cuda=not args.no_cuda)

    for arch in archs:
        model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda().eval()
        query_all = []
        not_done_all = []
        correct_all = []
        img_no = 0
        total_success = 0
        l2_total = 0.0
        avg_step = 0
        avg_time = 0
        avg_qry = 0
        result_dump_path = result_dir_path + "/{}_result.json".format(arch)
        # if os.path.exists(result_dump_path):
        #     continue
        log.info("Begin attack {} on {}".format(arch, args.dataset))
        for i, data_tuple in enumerate(dataset_loader):
            if args.dataset == "ImageNet":
                if model.input_size[-1] >= 299:
                    img, true_labels = data_tuple[1], data_tuple[2]
                else:
                    img, true_labels = data_tuple[0], data_tuple[2]
            else:
                img, true_labels = data_tuple[0], data_tuple[1]
            args.init_size = model.input_size[-1]
            if img.size(-1) != model.input_size[-1]:
                img = F.interpolate(img, size=model.input_size[-1], mode='bilinear',align_corners=True)

            img, true_labels = img.to(0), true_labels.to(0)
            with torch.no_grad():
                pred_logit = model(img)
            pred_label = pred_logit.argmax(dim=1)
            correct = pred_label.eq(true_labels).detach().cpu().numpy().astype(np.int32)
            correct_all.append(correct)
            if pred_label[0].item() != true_labels[0].item():
                log.info("Skip wrongly classified image no. %d, original class %d, classified as %d" % (
                i, pred_label.item(), true_labels.item()))
                query_all.append(0)
                not_done_all.append(0)
                continue
            img_no += 1
            timestart = time.time()
            meta_model_copy = copy.deepcopy(meta_model)
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=pred_logit.shape[1],
                                                                            size=target_labels[
                                                                                invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    target_labels = pred_logit.argmin(dim=1)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None
            target = true_labels if not args.targeted else target_labels
            adv, const, first_step = attack.run(model, meta_model_copy, img, target)
            timeend = time.time()
            if len(adv.shape) == 3:
                adv = adv.reshape((1,) + adv.shape)
            adv = torch.from_numpy(adv).permute(0, 3, 1, 2).cuda()  # BHWC -> BCHW
            diff = (adv - img).detach().cpu().numpy()
            l2_distortion = np.sqrt(np.sum(np.square(diff))).item()
            with torch.no_grad():
                adv_pred_logit = model(adv)
                adv_pred_label = adv_pred_logit.argmax(dim=1)
            success = False
            if not args.targeted:  # target is true label
                if adv_pred_label[0].item() != target[0].item():
                    success = True
            else:
                if adv_pred_label[0].item() == target[0].item():
                    success = True
            if l2_distortion > args.epsilone:
                success = False
            if success:
                # (first_step-1)//args.finetune_intervalargs.update_pixels2+first_step
                # The first step is the iterations used that find the adversarial examples;
                # args.finetune_interval is the finetuning per iterations;
                # args.update_pixels is the queried pixels each iteration.
                # Currently, we find only f(x+h)-f(x) could estimate the gradient well, so args.update_pixels*1 in my updated codes.
                not_done_all.append(0)
                # only 1 query for i pixle, because the estimated function is f(x+h)-f(x)/h
                query_all.append((first_step-1)//args.finetune_interval*args.update_pixels*1+first_step)
                total_success += 1
                l2_total += l2_distortion
                avg_step += first_step
                avg_time += timeend - timestart
                avg_qry += (first_step-1)//args.finetune_interval*args.update_pixels*1+first_step
                log.info("Attach {}-th image: {}".format(i, "success"))
            else:
                not_done_all.append(1)
                query_all.append(args.max_queries)
                log.info("Attach {}-th image: {}".format(i, "fail"))
        model.cpu()
        if total_success != 0:
            log.info(
                "[STATS][L1] total = {}, time = {:.3f}, distortion = {:.5f}, avg_step = {:.5f},avg_query = {:.5f}, success_rate = {:.3f}".format(
                    img_no, avg_time / total_success, l2_total / total_success,
                    avg_step / total_success, avg_qry / total_success, total_success / float(img_no)))
        correct_all = np.concatenate(correct_all, axis=0).astype(np.int32)
        query_all = np.array(query_all).astype(np.int32)
        not_done_all = np.array(not_done_all).astype(np.int32)
        success = (1 - not_done_all) * correct_all
        success_query = success * query_all
        query_threshold_success_rate, query_success_rate = success_rate_and_query_coorelation(query_all,
                                                                                              not_done_all)
        success_rate_to_avg_query = success_rate_avg_query(query_all, not_done_all)

        query_all_bounded = query_all.copy()
        not_done_all_bounded = not_done_all.copy()
        out_of_bound_indexes = np.where(query_all_bounded > args.max_queries)[0]
        if len(out_of_bound_indexes) > 0:
            not_done_all_bounded[out_of_bound_indexes] = 1
        success_bounded = (1-not_done_all_bounded) * correct_all
        success_query_bounded = success_bounded * query_all_bounded

        query_threshold_success_rate_bounded, query_success_rate_bounded = success_rate_and_query_coorelation(query_all_bounded, not_done_all_bounded)
        success_rate_to_avg_query_bounded = success_rate_avg_query(query_all_bounded, not_done_all_bounded)

        meta_info_dict = {"query_all": query_all.tolist(), "not_done_all": not_done_all.tolist(),
                          "correct_all": correct_all.tolist(),
                          "mean_query_unbounded_max_queries": np.mean(success_query[np.nonzero(success)[0]]).item(),
                          "max_query_unbounded_max_queries": np.max(success_query[np.nonzero(success)[0]]).item(),
                          "median_query_unbounded_max_queries": np.median(success_query[np.nonzero(success)[0]]).item(),
                          "avg_not_done_unbounded_max_queries": np.mean(not_done_all.astype(np.float32)).item(),

                          "mean_query_bounded_max_queries": np.mean(success_query_bounded[np.nonzero(success_bounded)[0]]).item(),
                          "max_query_bounded_max_queries": np.max(success_query_bounded[np.nonzero(success_bounded)[0]]).item(),
                          "median_query_bounded_max_queries": np.median(success_query_bounded[np.nonzero(success_bounded)[0]]).item(),
                          "avg_not_done_bounded_max_queries": np.mean(not_done_all_bounded.astype(np.float32)).item(),

                          "query_threshold_success_rate_dict_unbounded": query_threshold_success_rate,
                          "query_success_rate_dict_unbounded": query_success_rate,
                          "success_rate_to_avg_query_unbounded": success_rate_to_avg_query,

                          "query_threshold_success_rate_dict_bounded": query_threshold_success_rate_bounded,
                          "query_success_rate_dict_bounded": query_success_rate_bounded,
                          "success_rate_to_avg_query_bounded": success_rate_to_avg_query_bounded,

                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

def print_args(args):
    keys = sorted(args.keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args[key]))

if __name__ == "__main__":
    args = get_parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    result_dir_path = os.path.join("logs",get_expr_dir_name(args.dataset, args.targeted, args.target_type, args.use_tanh))
    os.makedirs(result_dir_path,exist_ok=True)
    if args.test_archs:
        set_log_file(result_dir_path + "/run.log")
    elif args.arch is not None:
        set_log_file(result_dir_path + "/run_{}.log".format(args.arch))
    log.info("using GPU :{}".format(args.gpu))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(result_dir_path))
    log.info('Called with args:')
    print_args(vars(args))
    main(args, result_dir_path)