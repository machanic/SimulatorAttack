import argparse
import os
import sys
import time

from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageFolder
from torchvision.transforms import transforms


sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from adversarial_defense.model.feature_defense_model import FeatureDefenseModel

from advertorch.attacks import LinfPGDAttack
import glog as log
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from config import PY_ROOT, CLASS_NUM, IN_CHANNELS, IMAGE_SIZE, IMAGE_DATA_ROOT
from dataset.tiny_imagenet import TinyImageNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whether_denoising', action="store_true", help="whether to add denoising block")
    parser.add_argument('--filter_type', type=str, default='NonLocal_Filter', choices=["NonLocal_Filter","Gaussian_Filter","Mean_Filter", "Median_Filter"],
                        help="filter type")
    parser.add_argument('--ksize', type=int, default=3, help="kernel size of the filter")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning_rate")
    parser.add_argument('--epochs', type=int, default=200, help="epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
    parser.add_argument('--gpu', type=str, required=True, help="used GPU")
    # parser.add_argument('--perturbation_threshold', dest='threshold', type=float, default=1e-8, help='maximum threshold')
    parser.add_argument('--arch', type=str, required=True,
                        help="The arch used to generate adversarial images for testing")
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_interval",type=int, default=50)
    args = parser.parse_args()
    return args

# def load_model(in_channels, num_classes, basic_model, whether_denoising, filter_type, ksize):
#     if basic_model == 'DenoiseResNet18':
#         model = DenoiseResNet18(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
#     elif basic_model == 'DenoiseResNet34':
#         model = DenoiseResNet34(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
#     elif basic_model == 'DenoiseResNet50':
#         model = DenoiseResNet50(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
#     elif basic_model == 'DenoiseResNet101':
#         model = DenoiseResNet101(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
#     elif basic_model == 'DenoiseResNet152':
#         model = DenoiseResNet152(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
#     return model

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

def get_preprocessor(input_size=None, use_flip=True, center_crop=False):
    processors = []
    if input_size is not None:
        processors.append(transforms.Resize(size=input_size))
    if use_flip:
        processors.append(transforms.RandomHorizontalFlip())
    if center_crop:
        processors.append(transforms.CenterCrop(max(input_size)))
    processors.append(transforms.ToTensor())
    return transforms.Compose(processors)

def get_img_label_data_loader(datasetname, batch_size, is_train, image_size=None):
    workers = 0
    if image_size is None:
        image_size =  IMAGE_SIZE[datasetname]
    preprocessor = get_preprocessor(image_size, is_train)
    if datasetname == "CIFAR-10":
        train_dataset = CIFAR10(IMAGE_DATA_ROOT[datasetname], train=is_train, transform=preprocessor)
    elif datasetname == "CIFAR-100":
        train_dataset = CIFAR100(IMAGE_DATA_ROOT[datasetname], train=is_train, transform=preprocessor)
    elif datasetname == "MNIST":
        train_dataset = MNIST(IMAGE_DATA_ROOT[datasetname], train=is_train, transform=preprocessor)
    elif datasetname == "FashionMNIST":
        train_dataset = FashionMNIST(IMAGE_DATA_ROOT[datasetname], train=is_train, transform=preprocessor)
    elif datasetname == "TinyImageNet":
        train_dataset = TinyImageNet(IMAGE_DATA_ROOT[datasetname], preprocessor, train=is_train)
    elif datasetname == "ImageNet":
        preprocessor = get_preprocessor(image_size, is_train, center_crop=True)
        sub_folder = "/train" if is_train else "/validation"  # Note that ImageNet uses pretrainedmodels.utils.TransformImage to apply transformation
        train_dataset = ImageFolder(IMAGE_DATA_ROOT[datasetname] + sub_folder, transform=preprocessor)
        workers = 5
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers)
    return data_loader


def test(model, testloader, criterion):
    model.eval()
    correct, total, loss, counter = 0, 0, 0, 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
            counter += 1
    return loss / total, correct / total

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = [idx for idx, gpu in enumerate(args.gpu.split(","))]
    work_dir = '{}/train_pytorch_model/adversarial_train/feature_denoise/'.format(PY_ROOT)
    # pretrained_model_path = '{}/train_pytorch_model/adversarial_train/feature_denoise/{}@{}_{}_{}_{}.pth.tar'.format(
    #     PY_ROOT, args.dataset, args.arch, denoise_str, args.filter_type, args.ksize)
    assert os.path.exists(work_dir), "{} does not exist!".format(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    set_log_file(work_dir + "/adv_train_{}.log".format(args.dataset))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)
    model_path = '{}/train_pytorch_model/adversarial_train/feature_denoise/pgd_adv_train_{}@{}_{}_{}.pth.tar'.format(
        PY_ROOT, args.dataset, args.arch, args.filter_type, args.ksize)
    best_model_path = '{}/train_pytorch_model/adversarial_train/feature_denoise/best_pgd_adv_train_{}@{}_{}_{}.pth.tar'.format(
        PY_ROOT, args.dataset, args.arch, args.filter_type, args.ksize)

    model = FeatureDefenseModel(args.dataset, args.arch, no_grad=False)
    model = model.cuda()
    resume_epoch = 0
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=lambda storage, location: storage)
        model.load_state_dict(state_dict["state_dict"])
        resume_epoch = state_dict["epoch"]
        log.info("Load model from {} at epoch {}".format(model_path, resume_epoch))
    # model = model.to(args.gpu)
    if torch.cuda.is_available():
        model.cuda(gpus[0])


    log.info("After trained over, model will be saved to {}".format(model_path))
    train_loader = get_img_label_data_loader(args.dataset, args.batch_size, True)
    test_loader = get_img_label_data_loader(args.dataset, args.batch_size, False)
    if torch.cuda.device_count() > 1:
        criterion = torch.nn.DataParallel(nn.CrossEntropyLoss(), gpus).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    scheduler = MultiStepLR(optimizer,
                            milestones=[int(args.epochs / 2), int(args.epochs * 3 / 4), int(args.epochs * 7 / 8)],
                            gamma=0.1)

    total, correct, train_loss = 0, 0, 0
    # Record the best accuracy
    best_test_clean_acc, best_test_adv_acc, best_epoch = 0, 0, 0
    log.info(
        "basic model: {}, whether denoising: {}, filter type: {}, kernel size: {}".format(
            args.arch, args.whether_denoising, args.filter_type, args.ksize))
    for epoch in range(resume_epoch, args.epochs):
        if epoch % args.test_interval == 0:
            model.eval()
            test_total, test_correct, test_robustness = 0, 0, 0
            attack = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031372,nb_iter=30,
                                   eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
            start_time = time.time()

            for (images, labels) in test_loader:
                images = images.cuda()
                labels = labels.cuda().long()
                test_total += images.shape[0]
                with torch.no_grad():
                    test_correct += model(images).max(1)[1].eq(labels).float().sum().item()
                adv_images = attack.perturb(images, labels)
                with torch.no_grad():
                    test_robustness += model(adv_images).max(1)[1].eq(labels).float().sum().item()
            test_acc, test_adv_acc = test_correct / test_total, test_robustness / test_total
            # Record the time on the testset
            end_time = time.time()
            testset_total_time = end_time - start_time
            if test_adv_acc > best_test_adv_acc:
                best_epoch = epoch
                best_test_adv_acc = test_adv_acc
                best_test_clean_acc = test_acc
                torch.save({"state_dict":model.state_dict(), "epoch": epoch+1}, best_model_path)
            log.info("Present best adversarial model ----- best epoch: {} clean_test_acc: {:.3f} adv_test_acc: {:.3f}".format(
                best_epoch, best_test_clean_acc, best_test_adv_acc))
            log.info("Epoch:{} clean_test_acc: {:.3f}  adv_test_acc: {:.3f} during {} seconds".format(epoch, test_acc, test_adv_acc, testset_total_time))

        # Test and Train on the trainset
        train_total, train_correct, train_robustness = 0, 0, 0
        train_clean_loss, train_adv_loss, train_loss = 0, 0, 0
        start_time = time.time()
        attack = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031372, nb_iter=30,
                               eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.cuda().long()
            train_total += images.shape[0]
            with torch.no_grad():
                train_correct += model(images).max(1)[1].eq(labels).float().sum().item()
            model.eval()
            adv_images = attack.perturb(images, labels)
            model.train()
            adv_outputs = model(adv_images)
            train_robustness += adv_outputs.max(1)[1].eq(labels).float().sum().item()
            adv_loss = criterion(adv_outputs, labels)
            # clean_outputs = model(torch.from_numpy(images).cuda())  # 我认为对抗训练不需要真实图片的loss
            # clean_loss = criterion(clean_outputs, torch.from_numpy(labels).cuda())
            optimizer.zero_grad()
            adv_loss.backward()
            optimizer.step()
            train_adv_loss += adv_loss.item()
            train_loss = train_adv_loss
            model.eval()

        scheduler.step(epoch)
        # Record the time on the trainset
        end_time = time.time()
        trainset_total_time = end_time - start_time
        train_acc, train_adv_acc = train_correct / train_total, train_robustness / train_total
        log.info(
            "Epoch:{} train_clean_loss: {:.3f} train_adv_loss: {:.3f} train_total_loss: {:.3f}".format(epoch, train_clean_loss,
                                                                                                       train_adv_loss,
                                                                                                       train_loss))
        log.info("Epoch:{} clean_train_acc: {:.3f}  adv_train_acc: {:.3f}  Consumed time:{}".format(epoch, train_acc, train_adv_acc,trainset_total_time))
        torch.save({"state_dict":model.state_dict(), "epoch": epoch+1}, model_path)



if __name__ == "__main__":
    main()