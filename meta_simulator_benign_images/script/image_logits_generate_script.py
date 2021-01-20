import argparse
import os
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from dataset.tiny_imagenet import TinyImageNet
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from config import IMAGE_SIZE, IMAGE_DATA_ROOT, MODELS_TRAIN_STANDARD, PY_ROOT
from dataset.standard_model import StandardModel
from dataset.dataset_loader_maker import DataLoaderMaker
from meta_simulator_benign_images.script import save_image_logits_pairs

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("--dataset",type=str, required=True)
parser.add_argument("--batch_size",type=int,default=100)
parser.add_argument("--gpu",type=int,required=True)
parser.add_argument("--max_items",type=int, default=100000)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
train_preprocessor = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# test_preprocessor = transforms.ToTensor()
dataset = args.dataset
if dataset == "CIFAR-10":
    train_dataset = CIFAR10(IMAGE_DATA_ROOT[dataset], train=True, transform=train_preprocessor)
    # test_dataset =  CIFAR10(IMAGE_DATA_ROOT[dataset], train=False, transform=test_preprocessor)
elif dataset == "CIFAR-100":
    train_dataset = CIFAR100(IMAGE_DATA_ROOT[dataset], train=True, transform=train_preprocessor)
    # test_dataset = CIFAR100(IMAGE_DATA_ROOT[dataset], train=False, transform=test_preprocessor)
elif dataset == "ImageNet":
    train_preprocessor = DataLoaderMaker.get_preprocessor(IMAGE_SIZE[dataset], True, center_crop=False)
    # test_preprocessor = DataLoaderMaker.get_preprocessor(IMAGE_SIZE[dataset], False, center_crop=True)
    train_dataset = ImageFolder(IMAGE_DATA_ROOT[dataset] + "/train", transform=train_preprocessor)
elif dataset == "TinyImageNet":
    train_dataset = TinyImageNet(IMAGE_DATA_ROOT[dataset], train_preprocessor, train=True)


batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=0)

print('==> Building model..')
arch_list = MODELS_TRAIN_STANDARD[args.dataset]
model_dict = {}
for arch in arch_list:
    if StandardModel.check_arch(arch, args.dataset):
        print("begin use arch {}".format(arch))
        model = StandardModel(args.dataset, arch, no_grad=True)
        model_dict[arch] = model.eval()
        print("use arch {} done".format(arch))
print("==> Save gradient..")
for arch, model in model_dict.items():
    dump_path = "{}/benign_images_logits_pair/{}/{}_images.npy".format(PY_ROOT, args.dataset, arch)
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    model = model.cuda()
    save_image_logits_pairs(model, train_loader, dump_path, args.batch_size, args.max_items)
    model.cpu()
