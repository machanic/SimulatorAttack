import argparse
import os
import sys

from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageFolder
from torchvision.transforms import transforms


sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import glog as log
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from dataset.tiny_imagenet import TinyImageNet

from adversarial_defense.model.denoise_resnet import DenoiseResNet18, DenoiseResNet34, DenoiseResNet50, DenoiseResNet101, DenoiseResNet152
from config import PY_ROOT, IN_CHANNELS, CLASS_NUM, IMAGE_SIZE, IMAGE_DATA_ROOT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whether_denoising', action="store_true", help="whether to add denoising block")
    parser.add_argument('--filter_type', type=str, default='NonLocal_Filter', help="filter type")
    parser.add_argument('--ksize', type=int, default=3, help="kernel size of the filter")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning_rate")
    parser.add_argument('--epochs', type=int, default=100, help="epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
    parser.add_argument('--gpu', type=str, required=True, help="used GPU")
    parser.add_argument('-a', '--arch', type=str, required=True,
                        help="The arch used to generate adversarial images for testing")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    return args

def load_model(in_channels, num_classes, basic_model, whether_denoising, filter_type, ksize):
    if basic_model == 'DenoiseResNet18':
        model = DenoiseResNet18(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
    elif basic_model == 'DenoiseResNet34':
        model = DenoiseResNet34(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
    elif basic_model == 'DenoiseResNet50':
        model = DenoiseResNet50(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
    elif basic_model == 'DenoiseResNet101':
        model = DenoiseResNet101(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
    elif basic_model == 'DenoiseResNet152':
        model = DenoiseResNet152(in_channels=in_channels, num_classes=num_classes, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
    return model

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
    normalize_mean = [0.4914, 0.4822, 0.4465]
    normalize_std = [0.2023, 0.1994, 0.2010]
    processors.append(transforms.Normalize(normalize_mean, normalize_std))
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
    model = load_model(in_channels=IN_CHANNELS[args.dataset], num_classes=CLASS_NUM[args.dataset], basic_model=args.arch,
                       whether_denoising=args.whether_denoising, filter_type=args.filter_type,
                       ksize=args.ksize)
    model = model.cuda()
    model_path = '{}/train_pytorch_model/adversarial_train/feature_denoise/{}@{}_{}_{}.pth.tar'.format(
        PY_ROOT, args.dataset, args.arch, args.filter_type, args.ksize)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    set_log_file(os.path.dirname(model_path) + "/train_{}_{}.log".format(args.dataset, args.arch))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)
    log.info("After trained over, model will be saved to {}".format(model_path))
    resume_epoch = 0
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=lambda storage, location: storage)
        model.load_state_dict(state_dict["state_dict"])
        resume_epoch = state_dict["epoch"]
        log.info("Load model from {} at epoch {}".format(model_path, resume_epoch))
    train_loader = get_img_label_data_loader(args.dataset, args.batch_size, True)
    test_loader = get_img_label_data_loader(args.dataset, args.batch_size, False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4, momentum=0.9, nesterov=True)
    scheduler = MultiStepLR(optimizer,
                            milestones=[int(args.epochs / 2), int(args.epochs * 3 / 4), int(args.epochs * 7 / 8)],
                            gamma=0.1)

    total, correct, train_loss = 0, 0, 0
    best_acc, best_epoch = 0, 0

    for epoch in range(resume_epoch, args.epochs):

        for data in train_loader:
            model.train()
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # count acc,loss on trainset
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        scheduler.step(epoch)
        acc = correct / total
        train_loss /= total
        val_loss, val_acc = test(model, test_loader, criterion)
        log.info(
            "epoch:{}, train_loss:{:.3f}, train_acc:{:.3f}, val_loss:{:.3f}, val_acc:{:.3f}".format(epoch, train_loss, acc,
                                                                                                val_loss, val_acc))
        correct, total, train_loss = 0, 0, 0
        # if best_acc < val_acc:
        #     best_acc = val_acc
        #     best_epoch = epoch
        torch.save({"state_dict":model.state_dict(), "epoch":epoch+1},  model_path)
        log.info("Best model at present: val_acc={:.3f}  best_epoch={}".format(best_acc, best_epoch))

if __name__ == "__main__":
    main()