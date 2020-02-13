import argparse
import random
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageFolder
from constant_enum import SPLIT_DATA_PROTOCOL
from dataset.tiny_imagenet import TinyImageNet
import os
import os.path as osp
import numpy as np
from cifar_models_myself import *
from config import IMAGE_SIZE, PY_ROOT, MODELS_TRAIN_STANDARD, IMAGE_DATA_ROOT, CLASS_NUM, \
    MODELS_TEST_STANDARD
import glog as log
from dataset.model_constructor import StandardModel
from dataset.dataset_loader_maker import DataLoaderMaker
from itertools import zip_longest # for Python 3.x

def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def get_log_path(dataset):
    dirname = 'image_gradient-{}.log'.format(dataset)
    return dirname

def set_log_file(fname):
    import subprocess
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

# def cw_loss(logit, label):
#     # untargeted cw loss: max_{i\neq y}logit_i - logit_y
#     _, argsort = logit.sort(dim=1, descending=True)
#     gt_is_max = argsort[:, 0].eq(label).long()
#     second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
#     gt_logit = logit[torch.arange(logit.shape[0]), label]
#     second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
#     return second_max_logit - gt_logit


def negative_cw_loss(logit, label):
    # untargeted cw loss: max_{i\neq y}logit_i - logit_y
    _, argsort = logit.sort(dim=1, descending=True)
    gt_is_max = argsort[:, 0].eq(label).long()
    second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
    gt_logit = logit[torch.arange(logit.shape[0]), label]
    second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
    return torch.clamp(gt_logit - second_max_logit, min=0.0)

def chunks(l, n, archs):
    n = max(1, n)
    batch_idx_arch_dict = dict()
    the_list = [l[i:i + n] for i in range(0, len(l), n)]
    for arch_idx, arch in enumerate(archs):
        for batch_idx in the_list[arch_idx % len(the_list)]:
            batch_idx_arch_dict[batch_idx] = arch
    return batch_idx_arch_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["CIFAR-10","CIFAR-100","MNIST","FashionMNIST","TinyImageNet","ImageNet"])
    parser.add_argument("--batch-size", type=int,default=200)
    parser.add_argument("--total-images",type=int, default=10000000)
    parser.add_argument("--protocol", required=True, type=SPLIT_DATA_PROTOCOL, choices=list(SPLIT_DATA_PROTOCOL),
                       help="split protocol of data")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    save_dir_path = "{}/data_grad_regression/{}".format(PY_ROOT, args.dataset)
    os.makedirs(save_dir_path, exist_ok=True)
    log_path = osp.join(save_dir_path, get_log_path(args.dataset))  # 随机产生一个目录用于实验
    set_log_file(log_path)
    log.info("All the data will be saved into {}".format(save_dir_path))
    log.info("Using GPU {}".format(args.gpu))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))

    if args.protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II:
        archs = MODELS_TRAIN_STANDARD[args.dataset]
    elif args.protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
        archs = MODELS_TRAIN_STANDARD[args.dataset] + MODELS_TEST_STANDARD[args.dataset]
    model_dict = {}
    for arch in archs:
        if StandardModel.check_arch(arch, args.dataset):
            model = StandardModel(args.dataset, arch, no_grad=False).eval()
            model_dict[arch] = model
    dataset = args.dataset
    is_train = True
    preprocessor = DataLoaderMaker.get_preprocessor(IMAGE_SIZE[dataset], is_train)
    if dataset == "CIFAR-10":
        train_dataset = CIFAR10(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
    elif dataset == "CIFAR-100":
        train_dataset = CIFAR100(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
    elif dataset == "MNIST":
        train_dataset = MNIST(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
    elif dataset == "FashionMNIST":
        train_dataset = FashionMNIST(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
    elif dataset == "TinyImageNet":
        train_dataset = TinyImageNet(IMAGE_DATA_ROOT[dataset], preprocessor, is_train=is_train)
    elif dataset == "ImageNet":
        preprocessor = DataLoaderMaker.get_preprocessor(IMAGE_SIZE[dataset], is_train, center_crop=True)
        sub_folder = "/train" if is_train else "/validation"  # Note that ImageNet uses pretrainedmodels.utils.TransformImage to apply transformation
        train_dataset = ImageFolder(IMAGE_DATA_ROOT[dataset] + sub_folder, transform=preprocessor)
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=0)
    num_batches = int(np.ceil(min(len(train_dataset), args.total_images) / batch_size))
    n = int(np.ceil(num_batches / len(model_dict)))
    assign_arch_list = chunks(np.arange(num_batches).tolist(), n, list(model_dict.keys()))
    last_arch = assign_arch_list[0]
    image_list = []
    grad_list = []
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx * batch_size > args.total_images:
            break
        arch = assign_arch_list.get(batch_idx, random.choice(list(assign_arch_list.values())))
        if os.path.exists(save_dir_path + "/{}_images.npy".format(arch)):
            log.info("skip {}".format(save_dir_path + "/{}_images.npy".format(arch)))
            continue
        if last_arch != arch and image_list:
            image_path = save_dir_path + "/{}_images.npy".format(last_arch)
            grad_path = save_dir_path + "/{}_gradients.npy".format(last_arch)
            image_list = np.concatenate(image_list, axis=0)
            grad_list = np.concatenate(grad_list, axis=0)
            fp = np.memmap(image_path, dtype='float32', mode='w+', shape=image_list.shape)
            fp[:, :, :, :] = image_list[:, :, :, :]
            del fp

            fp = np.memmap(grad_path, dtype="float32", mode="w+", shape=grad_list.shape)
            fp[:,:,:,:] = grad_list[:,:,:,:]
            del fp
            with open(image_path.replace(".npy", ".txt"), "w") as file_obj:
                file_obj.write(str(image_list.shape))
                file_obj.flush()
            log.info("save {} over".format(last_arch))
            image_list = []
            grad_list = []
            model.cpu()
            model_dict[last_arch].cpu()
            # model = model_dict[arch].cuda().eval()
        model = model_dict[arch].cuda().eval()
        last_arch = arch
        images, labels = images.cuda(), labels.cuda()
        images.requires_grad_()
        logits = model(images)
        assert logits.size(1) == CLASS_NUM[dataset]
        loss = negative_cw_loss(logits, labels).mean()
        model.zero_grad()
        loss.backward()
        grad_image = images.grad
        image_list.append(images.detach().cpu().numpy())
        grad_list.append(grad_image.detach().cpu().numpy())
    if image_list:
        image_path = save_dir_path + "/{}_images.npy".format(last_arch)
        grad_path = save_dir_path + "/{}_gradients.npy".format(last_arch)
        image_list = np.concatenate(image_list, axis=0)
        grad_list = np.concatenate(grad_list, axis=0)
        fp = np.memmap(image_path, dtype='float32', mode='w+', shape=image_list.shape)
        fp[:, :, :, :] = image_list[:, :, :, :]
        del fp

        fp = np.memmap(grad_path, dtype="float32", mode="w+", shape=grad_list.shape)
        fp[:, :, :, :] = grad_list[:, :, :, :]
        del fp
        with open(image_path.replace(".npy", ".txt"), "w") as file_obj:
            file_obj.write(str(image_list.shape))
            file_obj.flush()

