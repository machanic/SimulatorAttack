import argparse
import glob
import os
import sys
import time
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from dataset.standard_model import MetaLearnerModelBuilder
import glog as log
import numpy as np
import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from adversarial_defense.model.tinyimagenet_resnet_return_feature import resnet101, resnet152, resnet34, resnet50
from torch import nn
from adversarial_defense.high_level_guided_denoiser.dataset.adv_images_dataset import AdvImagesDataset
from adversarial_defense.model.guided_denoiser_network import Net
from adversarial_defense.model.resnet_return_feature import ResNet
from adversarial_defense.model.wrn_return_feature import WideResNet
from config import IMAGE_SIZE, PY_ROOT, CLASS_NUM, IN_CHANNELS


def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def train(epoch, net, data_loader, optimizer, get_lr, loss_idcs, requires_control=True):
    start_time = time.time()
    net.eval()
    net.denoise.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    orig_acc = []
    acc = []
    loss = []
    if requires_control:
        control_acc = []
        control_loss = []
    for i, (orig, adv, label) in enumerate(data_loader):
        orig = orig.cuda()
        adv = adv.cuda()
        label = label.long()
        if not requires_control:
            orig_pred, adv_pred, l = net(orig, adv, requires_control=False, train=True)
        else:
            orig_pred, adv_pred, l, control_pred, cl = net(orig, adv, requires_control=True, train=True)

        _, idcs = orig_pred.data.cpu().max(1)
        orig_acc.append(float(torch.sum((idcs == label).float())) / len(label))
        _, idcs = adv_pred.data.cpu().max(1)
        acc.append(torch.sum(idcs == label).float().item() / len(label))
        total_loss = 0
        for idx in loss_idcs:
            total_loss += l[idx].mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_values = []
        for ll in l:
            loss_values.append(ll.mean().item())
        loss.append(loss_values)

        if requires_control:
            _, idcs = control_pred.data.cpu().max(1)
            control_acc.append(float(torch.sum(idcs == label)) / len(label))
            loss_values = []
            for ll in cl:
                loss_values.append(ll.mean().item())
            control_loss.append(loss_values)

    orig_acc = np.mean(orig_acc)
    acc = np.mean(acc)
    loss = np.mean(loss, 0)
    if requires_control:
        control_acc = np.mean(control_acc)
        control_loss = np.mean(control_loss, 0)
    end_time = time.time()
    dt = end_time - start_time

    if requires_control:
        log.info('Epoch %3d (lr %.5f): orig_acc %.3f, acc %.3f, control_acc %.3f, time %3.1f' % (
            epoch, lr, orig_acc, acc, control_acc, dt))
    else:
        log.info('Epoch %3d (lr %.5f): orig_acc %.3f, acc %.3f, time %3.1f' % (
            epoch, lr, orig_acc, acc, dt))

    log.info('loss: %.5f, %.5f, %.5f' % (loss[0], loss[1], loss[2]))

    if requires_control:
        log.info('\tloss: %.5f, %.5f, %.5f' % (control_loss[0], control_loss[1], control_loss[2]))


def validate(net, data_loader, requires_control=True):
    start_time = time.time()
    net.eval()

    orig_acc = []
    acc = []
    loss = []
    if requires_control:
        control_acc = []
        control_loss = []
    for i, (orig, adv, label) in enumerate(data_loader):
        orig = orig.cuda()
        adv = adv.cuda()
        label = label.long()
        if not requires_control:
            orig_pred, adv_pred, l = net(orig, adv, requires_control=False, train=True)
        else:
            orig_pred, adv_pred, l, control_pred, cl = net(orig, adv, requires_control=True, train=True)

        _, idcs = orig_pred.data.cpu().max(1)
        orig_acc.append(float(torch.sum((idcs == label).float())) / len(label))
        _, idcs = adv_pred.data.cpu().max(1)
        acc.append(float(torch.sum((idcs == label).float())) / len(label))
        loss_values = []
        for ll in l:
            loss_values.append(ll.mean().item())
        loss.append(loss_values)

        if requires_control:
            _, idcs = control_pred.data.cpu().max(1)
            control_acc.append(float(torch.sum((idcs == label).float())) / len(label))
            loss_values = []
            for ll in cl:
                loss_values.append(ll.mean().item())
            control_loss.append(loss_values)

    orig_acc = np.mean(orig_acc)
    acc = np.mean(acc)
    loss = np.mean(loss, 0)
    if requires_control:
        control_acc = np.mean(control_acc)
        control_loss = np.mean(control_loss, 0)
    end_time = time.time()
    dt = end_time - start_time

    if requires_control:
        log.info('Validation: orig_acc %.3f, acc %.3f, control_acc %.3f, time %3.1f' % (
            orig_acc, acc, control_acc, dt))
    else:
        log.info('Validation: orig_acc %.3f, acc %.3f, time %3.1f' % (
            orig_acc, acc, dt))

    log.info('\tloss: %.5f, %.5f, %.5f' % (
        loss[0], loss[1], loss[2]))

    if requires_control:
        log.info('\tloss: %.5f, %.5f, %.5f' % (
            control_loss[0], control_loss[1], control_loss[2]))

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch defense model')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                        help='save frequency')
    parser.add_argument('--print-iter', default=0, type=int, metavar='I',
                        help='print per iter')
    parser.add_argument('--optimizer', default='Adam', type=str, choices=["Adam","SGD"],
                        help='optimizer')
    parser.add_argument('--dataset',type=str, required=True)
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--GD",type=str, choices=["FGD","LGD"], required=True)
    args = parser.parse_args()
    return args

def load_weight_from_pth_checkpoint(model, fname):
    raw_state_dict = torch.load(fname, map_location='cpu')['state_dict']
    state_dict = dict()
    for key, val in raw_state_dict.items():
        new_key = key.replace('module.', '')
        state_dict[new_key] = val
    model.load_state_dict(state_dict)


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    train_dataset = AdvImagesDataset(args.dataset) # 这个dataset没有经mean和std normalize过的
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0)
    test_dataset = AdvImagesDataset(args.dataset)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0)
    if args.dataset.startswith("CIFAR"):
        pretrained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}-pretrained/{arch}/checkpoint.pth.tar".format(
            root=PY_ROOT, dataset=args.dataset, arch=args.arch)
        assert os.path.exists(pretrained_model_path), "{} does not exist!".format(pretrained_model_path)
        if args.arch == "resnet-50":
            classifier = ResNet(50, CLASS_NUM[args.dataset], block_name='BasicBlock')
        elif args.arch == "resnet-110":
            classifier = ResNet(110, CLASS_NUM[args.dataset], block_name='BasicBlock')
        elif args.arch == "resnet-56":
            classifier = ResNet(56, CLASS_NUM[args.dataset], block_name='BasicBlock')
        elif args.arch == "WRN-28-10-drop":
            classifier = WideResNet(28, CLASS_NUM[args.dataset], widen_factor=10, dropRate=0.3)
        elif args.arch == "WRN-40-10-drop":
            classifier = WideResNet(40, CLASS_NUM[args.dataset], widen_factor=10, dropRate=0.3)

    elif args.dataset == "TinyImageNet":
        pretrained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}@*.pth.tar".format(
            root=PY_ROOT, dataset=args.dataset, arch=args.arch)
        pretrained_model_path = list(glob.glob(pretrained_model_path))
        assert len(pretrained_model_path) > 0, "{} does not exist!".format(pretrained_model_path)
        pretrained_model_path = pretrained_model_path[0]
        if args.arch == "resnet50":
            classifier = resnet50(num_classes=CLASS_NUM[args.dataset], pretrained=False)
        elif args.arch == "resnet152":
            classifier = resnet152(num_classes=CLASS_NUM[args.dataset],pretrained=False)
        elif args.arch == "resnet101":
            classifier = resnet101(num_classes=CLASS_NUM[args.dataset],pretrained=False)
        elif args.arch == "WRN-28-10-drop":
            classifier = WideResNet(28, CLASS_NUM[args.dataset], widen_factor=10, dropRate=0.3)
        elif args.arch == "WRN-40-10-drop":
            classifier = WideResNet(40, CLASS_NUM[args.dataset], widen_factor=10, dropRate=0.3)
        num_classes = CLASS_NUM[args.dataset]
        if args.arch.startswith("resnet"):
            num_ftrs = classifier.fc.in_features
            classifier.fc = nn.Linear(num_ftrs, num_classes)
    load_weight_from_pth_checkpoint(classifier, pretrained_model_path)
    classifier.eval()
    log.info("Loaded pretrained {} from {}".format(args.arch, pretrained_model_path))
    net = Net(classifier, args.dataset, IMAGE_SIZE[args.dataset][0], IN_CHANNELS[args.dataset], 1, 0, False)
    net.cuda()
    save_dir = "{}/train_pytorch_model/adversarial_train/guided_denoiser".format(PY_ROOT)
    os.makedirs(save_dir, exist_ok=True)
    set_log_file(save_dir+"/train_{}_{}.log".format(args.dataset,args.arch))
    model_path = "{}/guided_denoiser_{}_{}_{}.pth.tar".format(save_dir, args.dataset, args.arch, args.GD)
    resume_epoch = 0
    # if os.path.exists(model_path):
    #     checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
    #     resume_epoch = checkpoint["epoch"]
    #     net.load_state_dict(checkpoint['state_dict'])
    #     log.info("Load resume model from {} (epoch:{})".format(model_path, resume_epoch))
    log.info("After trained over, the model will be saved to {}".format(model_path))
    cudnn.benchmark = True
    params = net.denoise.parameters()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr = args.lr, momentum = 0.9, weight_decay = args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr = args.lr, weight_decay = args.weight_decay)
    def get_lr(epoch):
        if epoch <= args.epochs * 0.6:
            return args.lr
        elif epoch <= args.epochs * 0.9:
            return args.lr * 0.1
        else:
            return args.lr * 0.01

    for epoch in range(resume_epoch, args.epochs + 1):
        requires_control = (epoch == resume_epoch)
        if args.GD == "FGD":
            loss_ids = [1]
        elif args.GD == "LGD":
            loss_ids = [2]
        train(epoch, net, train_data_loader, optimizer, get_lr, loss_ids, requires_control=requires_control)
        validate(net, test_data_loader, requires_control=requires_control)

        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'epoch': epoch+1,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            model_path)
        log.info("Epoch {} done, model was saved to {}".format(epoch, model_path))

if __name__ == '__main__':
    main()
