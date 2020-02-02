import sys

import torch

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
import os
from constant_enum import SPLIT_DATA_PROTOCOL
from meta_grad_regression_auto_encoder.meta_learner import MetaLearner
from config import PY_ROOT

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta Model Training')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to train")
    parser.add_argument("--epoch", type=int, default=10, help="number of epochs.")
    parser.add_argument('--meta_batch_size', type=int, default=30, help='number of tasks sampled per meta-update')
    parser.add_argument('--inner_batch_size',type=int, default=10)
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='the base learning rate')
    parser.add_argument("--num_inner_updates", type=int, default=1)  # Meta Attack论文里内部更新只有1次或多次
    parser.add_argument('--lr_decay_itr',type=int, default=7000, help="* 1/10. the number of iteration that the meta lr should decay")
    parser.add_argument("--inner_lr", type=float, default=1e-2, help="the inner step size")
    parser.add_argument('--tot_num_tasks', type=int, default=50000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument("--split_protocol", required=True, type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL),
                        help="split protocol of data")
    parser.add_argument("--dataset", type=str, default="CIFAR-10", help="the dataset to train")
    parser.add_argument("--study_subject", type=str, default="meta_grad_regression")
    # the following args are set for choosing which npy data
    parser.add_argument("--evaluate", action="store_true")
    ## Logging, saving, and testing options
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("using GPU :{}".format(args.gpu))
    return args

def main():
    args = parse_args()
    param_prefix = "{}@{}@model_AE@epoch_{}@tot_tasks_{}@meta_batch_size_{}@num_updates_{}@lr_{}@inner_lr_{}".format(
        args.dataset, args.split_protocol, args.epoch, args.tot_num_tasks, args.meta_batch_size, args.num_inner_updates,
        args.meta_lr, args.inner_lr)
    model_path = '{}/train_pytorch_model/{}/{}@{}.pth.tar'.format(PY_ROOT, args.study_subject, "MetaGradRegression", param_prefix)
    os.makedirs(os.path.dirname(model_path),exist_ok=True)

    meta_learner = MetaLearner(args.dataset, args.meta_batch_size, args.inner_batch_size, args.inner_lr, args.epoch,
                               args.num_inner_updates, args.split_protocol, args.tot_num_tasks)
    resume_epoch = 0
    if os.path.exists(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        resume_epoch = checkpoint['epoch']
        meta_learner.network.load_state_dict(checkpoint['state_dict'], strict=True)
        meta_learner.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))
    meta_learner.train(args.meta_lr, resume_epoch, model_path)

if __name__ == "__main__":
    main()