import sys



sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from meta_model.meta_distillation_learner import MetaDistillationLearner
import argparse
from config import PY_ROOT
import random
import numpy as np
from meta_model.meta_grad_regression_learner import MetaGradRegressionLearner
import torch
from constant_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import os
from meta_two_queries_distillation_learner import MetaTwoQueriesLearner
from meta_prior_regression_learner import MetaPriorRegressionLearner

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta Model Training')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to train")
    parser.add_argument("--epoch", type=int, default=4, help="number of epochs.")
    parser.add_argument('--meta_batch_size', type=int, default=30, help='number of tasks sampled per meta-update')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='the base learning rate')
    parser.add_argument('--inner_lr', type=float, default=1e-2, help="lr for inner update")
    parser.add_argument('--lr_decay_itr',type=int, default=7000, help="* 1/10. the number of iteration that the meta lr should decay")
    parser.add_argument('--num_updates', type=int, default=12,
                        help='number of inner gradient updates(on support set) during training.')
    parser.add_argument('--tot_num_tasks', type=int, default=20000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument('--arch', type=str, default='AE',help='network name')  #10 层
    parser.add_argument('--meta_learner', type=str, default="2q_distillation", choices=["2q_distillation", "grad_regression",
                                                                                        "logits_distillation", "prior_regression"])
    parser.add_argument("--num_support", type=int, default=20)
    parser.add_argument('--test_num_updates', type=int, default=20, help='number of inner gradient updates during testing')
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="the dataset to train")
    parser.add_argument("--split_protocol", required=True, type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL),
                        help="split protocol of data")
    parser.add_argument("--load_task_mode", default=LOAD_TASK_MODE.LOAD, type=LOAD_TASK_MODE, choices=list(LOAD_TASK_MODE),
                        help="load task mode")
    parser.add_argument("--study_subject", type=str, default="cross_arch_attack", required=True)
    parser.add_argument("--data_loss", type=str, default="xent", choices=["logits_loss", "xent"])
    parser.add_argument("--distill_loss", type=str, default="MSE", choices=["MSE","CSE"])
    parser.add_argument("--data_attack_type", type=str)
    parser.add_argument("--evaluate", action="store_true")

    ## Logging, saving, and testing options
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("using GPU :{}".format(args.gpu))
    return args

def main():
    args = parse_args()
    random.seed(1337)
    np.random.seed(1337)
    print('Setting GPU to', str(args.gpu))

    if not args.evaluate:
        data_type_str = args.data_attack_type if args.meta_learner in ["2q_distillation","prior_regression"] else args.data_loss
        param_prefix = "{}@{}@model_{}@data_{}@distill_loss_{}@epoch_{}@meta_batch_size_{}@num_support_{}@num_updates_{}@lr_{}@inner_lr_{}".format(
            args.dataset,
            args.split_protocol, args.arch, data_type_str, args.distill_loss, args.epoch, args.meta_batch_size,
            args.num_support, args.num_updates, args.meta_lr, args.inner_lr)
        model_path = '{}/train_pytorch_model/{}/{}@{}.pth.tar'.format(
            PY_ROOT, args.study_subject, args.meta_learner.upper(), param_prefix)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if args.meta_learner == "grad_regression":
            learner = MetaGradRegressionLearner(args.dataset, args.arch, args.meta_batch_size, args.meta_lr, args.inner_lr,
                                                args.lr_decay_itr, args.epoch, args.num_updates, args.load_task_mode,
                                                args.split_protocol, args.tot_num_tasks, args.num_support, args.data_loss,
                                                param_prefix)
        elif args.meta_learner == "prior_regression":
            learner = MetaPriorRegressionLearner(args.dataset, args.arch, args.meta_batch_size, args.meta_lr, args.inner_lr,
                                                 args.lr_decay_itr, args.epoch, args.num_updates, args.load_task_mode,
                                                 args.split_protocol, args.tot_num_tasks, args.num_support, args.data_attack_type,
                                                 param_prefix)
        elif args.meta_learner == "logits_distillation":
            learner = MetaDistillationLearner(args.dataset, args.arch, args.meta_batch_size, args.meta_lr, args.inner_lr,
                                              args.lr_decay_itr, args.epoch, args.num_updates, args.load_task_mode,
                                              args.split_protocol, args.tot_num_tasks, args.num_support, args.distill_loss,
                                              args.data_loss, param_prefix)
        elif args.meta_learner == "2q_distillation":
            learner = MetaTwoQueriesLearner(args.dataset, args.arch, args.meta_batch_size, args.meta_lr, args.inner_lr,
                                              args.lr_decay_itr, args.epoch, args.num_updates, args.load_task_mode,
                                              args.split_protocol, args.tot_num_tasks, args.num_support, args.data_attack_type,
                                              param_prefix)

        resume_epoch = 0
        if os.path.exists(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            resume_epoch = checkpoint['epoch']
            learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
            learner.opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
        print("The model will be stored in {}".format(model_path))
        learner.train(model_path, resume_epoch)

if __name__ == '__main__':
    main()
