import sys


sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
from config import PY_ROOT
import random
import numpy as np
import torch
from constant_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import os
from meta_simulator_benign_images.meta_training.meta_learner import MetaLearner
from meta_simulator_benign_images.meta_training.reptile_meta_learner import ReptileMetaLearner

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta Model Training')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to train")
    parser.add_argument("--epoch", type=int, default=3, help="number of epochs.")
    parser.add_argument('--meta_batch_size', type=int, default=30, help='number of tasks sampled per meta-update')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='the base learning rate')
    parser.add_argument('--inner_lr', type=float, default=1e-2, help="lr for inner update")
    parser.add_argument('--lr_decay_itr',type=int, default=10000, help="* 1/10. the number of iteration that the meta lr should decay")
    parser.add_argument('--num_updates', type=int, default=12,
                        help='number of inner gradient updates(on support set) during training.')
    parser.add_argument('--tot_num_tasks', type=int, default=30000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument('--arch', type=str, required=True,help='network name')  #10 层
    parser.add_argument("--num_support", type=int, default=10)
    parser.add_argument("--num_query", type=int, default=10)
    parser.add_argument('--test_num_updates', type=int, default=20, help='number of inner gradient updates during testing')
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to train")
    parser.add_argument("--split_protocol", default=SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II, type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL),
                        help="split protocol of data")
    parser.add_argument("--load_task_mode", default=LOAD_TASK_MODE.NO_LOAD, type=LOAD_TASK_MODE, choices=list(LOAD_TASK_MODE),
                        help="load task mode")
    parser.add_argument("--study_subject", type=str, default="meta_simulator_on_benign_images")
    # the following args are set for choosing which npy data
    parser.add_argument("--without_resnet",action="store_true")
    parser.add_argument("--reptile", action="store_true")
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
    if args.reptile:
        args.study_subject = "meta_simulator_reptile_on_benign_images"
    param_prefix = "{}@{}@model_{}@epoch_{}@meta_batch_size_{}@num_support_{}@num_query_{}@num_updates_{}@lr_{}@inner_lr_{}".format(
        args.dataset,
        args.split_protocol, args.arch, args.epoch, args.meta_batch_size,
        args.num_support, args.num_query, args.num_updates, args.meta_lr, args.inner_lr)
    model_path = '{}/train_pytorch_model/{}/{}.pth.tar'.format(
        PY_ROOT, args.study_subject,  param_prefix)
    if args.without_resnet:
        model_path = '{}/train_pytorch_model/{}/{}@without_resnet.pth.tar'.format(
            PY_ROOT, args.study_subject,  param_prefix)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if args.reptile:
        learner = ReptileMetaLearner(args.dataset, args.arch, args.meta_batch_size, args.meta_lr, args.inner_lr,
                                     args.lr_decay_itr, args.epoch, args.num_updates, args.load_task_mode,
                                     args.split_protocol, args.tot_num_tasks, args.num_support, args.without_resnet)
    else:
        learner = MetaLearner(args.dataset, args.arch, args.meta_batch_size, args.meta_lr, args.inner_lr,
                                        args.lr_decay_itr, args.epoch, args.num_updates, args.load_task_mode,
                                    args.split_protocol, args.tot_num_tasks, args.num_support, args.num_query, args.without_resnet)

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
