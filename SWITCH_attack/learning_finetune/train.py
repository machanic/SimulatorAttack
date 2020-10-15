import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
from config import PY_ROOT
import random
import numpy as np
import torch
from constant_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE
import os
from SWITCH_attack.learning_finetune.meta_learner import MetaLearner

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta Model Training')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to train")
    parser.add_argument("--epoch", type=int, default=3, help="number of epochs.")
    parser.add_argument('--meta_batch_size', type=int, default=30, help='number of tasks sampled per meta-update')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='the base learning_finetune rate')
    parser.add_argument('--inner_lr', type=float, default=1e-2, help="lr for inner update")
    parser.add_argument('--lr_decay_itr',type=int, default=10000, help="* 1/10. the number of iteration that the meta lr should decay")
    parser.add_argument('--num_updates', type=int, default=12,
                        help='number of inner gradient updates(on support set) during training.')
    parser.add_argument('--tot_num_tasks', type=int, default=30000, help='the maximum number of tasks in total, which is repeatly processed in training.')
    parser.add_argument('--arch', type=str, default='resnet34',help='network name')
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to train")
    parser.add_argument("--split_protocol", required=True, type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL),
                        help="split protocol of data")
    parser.add_argument("--load_task_mode", default=LOAD_TASK_MODE.NO_LOAD, type=LOAD_TASK_MODE, choices=list(LOAD_TASK_MODE),
                        help="load task mode")
    parser.add_argument("--study_subject", type=str, default="meta_simulator_surrogate_gradient")
    parser.add_argument("--data_loss_type", type=str, default='cw', choices=["xent", "cw"])
    parser.add_argument("--adv_norm", type=str, choices=["l2", "linf"], required=True)
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    parser.add_argument("--target_type", type=str, default="random", choices=["random", "least_likely"])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--without_resnet",action="store_true")
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
        data_str = "{loss}_{norm}_{target}".format(loss=args.data_loss_type, norm=args.adv_norm, target="untargeted" if not args.targeted else "targeted")
        param_prefix = "{}@{}@model_{}@loss_{}@epoch_{}@meta_batch_size_{}@num_updates_{}@lr_{}@inner_lr_{}".format(
            args.dataset, args.split_protocol, args.arch,  data_str, args.epoch, args.meta_batch_size,
            args.num_updates, args.meta_lr, args.inner_lr)
        model_path = '{}/train_pytorch_model/{}/{}.pth.tar'.format(
            PY_ROOT, args.study_subject, param_prefix)
        if args.without_resnet:
            model_path = '{}/train_pytorch_model/{}/@{}@without_resnet.pth.tar'.format(
                PY_ROOT, args.study_subject,  param_prefix)
        print("The trained model will saved to {}".format(model_path))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        learner = MetaLearner(args.dataset, args.arch, args.meta_batch_size, args.meta_lr, args.inner_lr,
                              args.lr_decay_itr, args.epoch, args.num_updates, args.load_task_mode,
                              args.split_protocol, args.tot_num_tasks, args.data_loss_type,
                              args.adv_norm, args.targeted, args.target_type, args.without_resnet)
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
