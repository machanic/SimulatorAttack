import sys

from torch.optim import Adam

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from advertorch.attacks import LinfPGDAttack

from adversarial_defense.com_defend.compression_network import ComDefend

from config import PY_ROOT
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from config import IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel
import numpy as np
import glog as log

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


def test_attack(threshold, arch, dataset, test_loader):
    target_model = StandardModel(dataset, arch, no_grad=False)
    if torch.cuda.is_available():
        target_model = target_model.cuda()
    target_model.eval()
    attack = LinfPGDAttack(target_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=threshold, nb_iter=30,
                           eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    all_count = 0
    success_count = 0
    all_adv_images = []
    all_true_labels = []
    for idx, (img, true_label) in enumerate(test_loader):
        img = img.cuda()
        true_label = true_label.cuda().long()

        adv_image = attack.perturb(img, true_label) # (3, 224, 224), float
        if adv_image is None:
            continue
        adv_label = target_model.forward(adv_image).max(1)[1].detach().cpu().numpy().astype(np.int32)
        # adv_image = np.transpose(adv_image, (0, 2, 3, 1)) # N,C,H,W -> (N, H, W, 3), float
        all_count += len(img)
        true_label_np = true_label.detach().cpu().numpy().astype(np.int32)
        success_count+= len(np.where(true_label_np != adv_label)[0])
        all_adv_images.append(adv_image.cpu().detach().numpy())
        all_true_labels.append(true_label_np)
    attack_success_rate = success_count / float(all_count)
    log.info("Before train. Attack success rate is {:.3f}".format(attack_success_rate))
    return target_model, np.concatenate(all_adv_images,0), np.concatenate(all_true_labels, 0)  # N,224,224,3

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
def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', type=str, required=True, help="The arch used to generate adversarial images for testing")
    parser.add_argument("--gpu",type=str,required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--test_mode', default=0, type=int, choices=list(range(10)))
    # parser.add_argument('--model', default='res', type=str)
    parser.add_argument('--n_epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=10, type=int)
    parser.add_argument('--lambd', default=0.0001, type=float)
    parser.add_argument('--noise_dev', default=20.0, type=float)
    parser.add_argument('--Linfinity', default=8/255, type=float)
    parser.add_argument('--binary_threshold', default=0.5, type=float)
    parser.add_argument('--lr_mode', default=0, type=int)
    parser.add_argument('--test_interval', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument("--use_res_net",action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cudnn.deterministic = True
    model_path = '{}/train_pytorch_model/adversarial_train/com_defend/{}@{}@epoch_{}@batch_{}.pth.tar'.format(
        PY_ROOT, args.dataset, args.arch, args.n_epoch, args.batch_size)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    set_log_file(os.path.dirname(model_path) + "/train_{}.log".format(args.dataset))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)

    in_channels = IN_CHANNELS[args.dataset]
    # if args.use_res_net:
    #     if args.test_mode == 0:
    #         com_defender = ModelRes(in_channels=in_channels, com_disable=True,rec_disable=True)
    #         args.save_model = 'normal'
    #     elif args.test_mode == 1:
    #         com_defender = ModelRes(in_channels=in_channels, n_com=1,n_rec=3,com_disable=False,rec_disable=True)
    #         args.save_model = '1_on_off'
    #     elif args.test_mode == 2:
    #         com_defender = ModelRes(in_channels=in_channels, n_com=2,n_rec=3,com_disable=False,rec_disable=True)
    #         args.save_model = '2_on_off'
    #     elif args.test_mode == 3:
    #         com_defender = ModelRes(in_channels=in_channels, n_com=3,n_rec=3,com_disable=False,rec_disable=True)
    #         args.save_model = '3_on_off'
    #     elif args.test_mode == 4:
    #         com_defender = ModelRes(in_channels=in_channels, n_com=3,n_rec=1,com_disable=True,rec_disable=False)
    #         args.save_model = 'off_on_1'
    #     elif args.test_mode == 5:
    #         com_defender = ModelRes(in_channels=in_channels, n_com=3,n_rec=2,com_disable=True,rec_disable=False)
    #         args.save_model = 'off_on_2'
    #     elif args.test_mode == 6:
    #         com_defender = ModelRes(in_channels=in_channels, n_com=3,n_rec=3,com_disable=True,rec_disable=False)
    #         args.save_model = 'off_on_3'
    #     elif args.test_mode == 7:
    #         com_defender = ModelRes(in_channels=in_channels, n_com=1,n_rec=1,com_disable=False,rec_disable=False)
    #         args.save_model = '1_1'
    #     elif args.test_mode == 8:
    #         com_defender = ModelRes(in_channels=in_channels, n_com=2,n_rec=2,com_disable=False,rec_disable=False)
    #         args.save_model = '2_2'
    #     elif args.test_mode == 9:
    #         com_defender = ModelRes(in_channels=in_channels, n_com=3,n_rec=3,com_disable=False,rec_disable=False)
    #         args.save_model = '3_3'
    # else:
    com_defender = ComDefend(in_channels, args.noise_dev)
    args.save_model = "normal_network"
    log.info('test mode: {}, model name: {}'.format(args.test_mode, args.save_model))

    if args.gpu is not None:
        log.info("Use GPU: {} for training".format(args.gpu))
    log.info("=> creating model '{}'".format(args.arch))

    log.info("after train, model will be saved to {}".format(model_path))
    com_defender.cuda()
    cudnn.benchmark = True
    train_loader = DataLoaderMaker.get_imgid_img_label_data_loader(args.dataset, args.batch_size, True, seed=1234)
    test_attack_dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size)
    log.info("Begin generate the adversarial examples.")
    target_model, adv_images, adv_true_labels = test_attack(args.Linfinity, args.arch, args.dataset,
                                                      test_attack_dataset_loader)  # 这些图片被用来验证

    log.info("Generate adversarial examples done!")
    best_acc = torch.zeros(1)
    for epoch in range(0, args.n_epoch):
        train(args, train_loader, com_defender, epoch, target_model, adv_images, adv_true_labels,best_acc, model_path)




def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, train_loader, com_defender, epoch, target_model, adv_images, adv_true_labels, best_acc, model_path):

    # switch to train_simulate_grad_mode mode
    com_defender.train()
    sigmoid_func = nn.Sigmoid().cuda()
    end = time.time()
    optimizer= Adam(com_defender.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for i, (_, input,  _) in enumerate(train_loader):
        # measure data loading time
        global_steps = len(train_loader) * epoch + i
        input = input.cuda()
        noisy_x = input
        linear_code = com_defender.forward_com(noisy_x)
        # add gaussian before sigmoid to encourage binary code
        noisy_code = linear_code -torch.randn_like(linear_code).cuda() * args.noise_dev
        # noisy_code = linear_code - torch.normal(mean=torch.zeros_like(linear_code), std=torch.ones_like(linear_code) * args.noise_dev).cuda()
        binary_code =sigmoid_func(noisy_code)
        y =com_defender.forward_rec(binary_code)
        # binary_code_test = (binary_code > args.binary_threshold).float()
        # y_test =com_defender.forward_rec(binary_code_test)
        loss = torch.mean((y - noisy_x).pow(2)) + torch.mean(binary_code.pow(2)) * args.lambd
        # if args.lr_mode == 0:
        #     # constant
        #     lr = 0.001
        #     adjust_learning_rate(optimizer, lr)
        # elif args.lr_mode == 1:
        #     # constant decay
        #     iter_total = len(train_loader.dataset) // args.batch_size * args.n_epoch
        #     boundaries = [int(iter_total * 0.25), int(iter_total * 0.75), int(iter_total * 0.9)]
        #     values = [0.01, 0.001, 0.0005, 0.0001]
        #     if global_steps < boundaries[0]:
        #         lr = values[0]
        #     elif global_steps>=boundaries[0] and global_steps < boundaries[1]:
        #         lr = values[1]
        #     elif global_steps >= boundaries[1] and global_steps < boundaries[2]:
        #         lr = values[2]
        #     else:
        #         lr = values[3]
        #     adjust_learning_rate(optimizer, lr)
        # elif args.lr_mode == 2:
        #     iter_total = len(train_loader.dataset) // args.batch_size * args.n_epoch
        #     lr_start = 0.01
        #     decay_steps = iter_total // 100
        #     decay_rate = 0.96
        #     # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        #     lr = lr_start  * decay_rate  ** (global_steps / (iter_total // 100))
        #     adjust_learning_rate(optimizer, lr)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        if global_steps%args.test_interval == 0: # and global_steps > 0:
            with torch.no_grad():
                com_defender.eval()
                target_model.eval()
                all_adv_images_length = len(adv_images)
                correct_count= 0
                all_count = 0
                for i in range(0, all_adv_images_length, args.batch_size):
                    adv_image =adv_images[i:i+args.batch_size]
                    adv_true_label = adv_true_labels[i:i+args.batch_size]
                    linear_code = com_defender.forward_com(torch.from_numpy(adv_image).cuda())  # 一次从进1000张图
                    noisy_code = linear_code - torch.randn_like(linear_code).cuda() * args.noise_dev
                    # noisy_code = linear_code - torch.normal(mean=torch.zeros_like(linear_code), std=torch.ones_like(linear_code) * args.noise_dev).cuda()
                    binary_code = sigmoid_func(noisy_code)
                    y = com_defender.forward_rec(binary_code)
                    binary_code_test = (binary_code >args.binary_threshold).float()
                    y_test = com_defender.forward_rec(binary_code_test)
                    label_clean_pred= target_model.forward(y_test).max(1)[1].detach().cpu().numpy()
                    correct_count += np.sum((adv_true_label == label_clean_pred).astype(np.float32)).item()
                    all_count += len(label_clean_pred)
                accuracy = float(correct_count) / all_count
                if accuracy > best_acc[0].item():
                    best_acc.fill_(accuracy)
                    torch.save({"epoch": epoch + 1,
                                "accuracy": accuracy,
                                "accuracy_arch": args.arch,
                                "state_dict": com_defender.state_dict()}, f=model_path)
                log.info("steps:{}. Accracy after defense is {:.3f}".format(global_steps, accuracy))
                com_defender.train()

if __name__ == '__main__':
    main()
