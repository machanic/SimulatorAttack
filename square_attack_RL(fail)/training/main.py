import argparse
import os
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageFolder
import torchvision.datasets as datasets
from PIL import Image
from torch.utils import data
from square_attack_RL.training.FCN import MyFCN
from square_attack_RL.training.environment import Environment
from square_attack_RL.training.pixelwise_a3c import PixelWiseA3C
from square_attack_RL.training.state import State
from config import IN_CHANNELS, IMAGE_SIZE, IMAGE_DATA_ROOT, PY_ROOT
from torch.optim import Adam
import glog as log
import torch
from square_attack_RL.training.test import test
from dataset.standard_model import StandardModel


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class TinyImageNet(data.Dataset):
    def __init__(self, root, transform, train=True):
        self.is_train = train
        self.category_names = sorted(os.listdir(root + "/train/"))
        self.transform = transform
        if train:
            self.dataset = datasets.ImageFolder(root + "/train/", transform=transform)
        else:
            self.file_name_to_category_id = {}
            with open(root + "/val/val_annotations.txt", "r") as file_obj:
                for line in file_obj:
                    image_file_name, category_name, *_ = line.split()
                    self.file_name_to_category_id[root + "/val/images/{}".format(image_file_name)] \
                        = self.category_names.index(category_name)
    def __len__(self):
        if self.is_train:
            return len(self.dataset)
        else:
            return len(self.file_name_to_category_id)

    def __getitem__(self, item):
        if self.is_train:
            img, label = self.dataset[item]
        else:
            file_path = list(self.file_name_to_category_id.keys())[item]
            img = pil_loader(file_path)
            img = self.transform(img)
            label = self.file_name_to_category_id[file_path]
        return img, label

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

def get_img_label_data_loader(dataset, batch_size, is_train):
    workers = 0
    image_size =  IMAGE_SIZE[dataset]
    preprocessor = get_preprocessor(image_size, is_train)
    if dataset == "CIFAR-10":
        train_dataset = CIFAR10(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
    elif dataset == "CIFAR-100":
        train_dataset = CIFAR100(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
    elif dataset == "MNIST":
        train_dataset = MNIST(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
    elif dataset == "FashionMNIST":
        train_dataset = FashionMNIST(IMAGE_DATA_ROOT[dataset], train=is_train, transform=preprocessor)
    elif dataset == "TinyImageNet":
        train_dataset = TinyImageNet(IMAGE_DATA_ROOT[dataset], preprocessor, train=is_train)
        workers = 3
    elif dataset == "ImageNet":
        preprocessor = get_preprocessor(image_size, is_train, center_crop=True)
        sub_folder = "/train" if is_train else "/validation"  # Note that ImageNet uses pretrainedmodels.utils.TransformImage to apply transformation
        train_dataset = ImageFolder(IMAGE_DATA_ROOT[dataset] + sub_folder, transform=preprocessor)
        workers = 5
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=is_train,
                                              num_workers=workers)
    return data_loader

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",'-lr', type=float, default=0.001)
    parser.add_argument("--dataset",type=str, required=True, choices=["CIFAR-10","CIFAR-100","TinyImageNet","ImageNet"])
    parser.add_argument("--batch_size",type=int,default=100)
    parser.add_argument("--epochs", type=int,default=100)
    parser.add_argument("--n_episodes",type=int,default=30000)
    parser.add_argument("--model", type=str,required=True)
    parser.add_argument("--episode_len",type=int,default=10)
    # parser.add_argument("--test_episodes",type=int,default=3000)
    parser.add_argument("--gamma",type=float,default=0.95)
    parser.add_argument("--n_actions",type=int,default=3)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--norm", type=str, choices=["l2","linf"], required=True)
    parser.add_argument("--gpu",type=int,required=True)
    args = parser.parse_args()
    return args

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def adjust_learning_rate(optimizer, learning_rate, episode, n_episodes):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * ((1 - episode / n_episodes) ** 0.9)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    args = get_parse_args()
    if args.norm == "l2":
        args.epsilon = 4.6
    elif args.norm == "linf":
        args.epsilon = 0.031372
        if args.dataset == "ImageNet":
            args.epsilon = 0.05
    data_loader  = get_img_label_data_loader(args.dataset, args.batch_size, True)
    test_data_loader = get_img_label_data_loader(args.dataset, args.batch_size, False)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    target_model = StandardModel(args.dataset, args.model, no_grad=True)
    target_model = target_model.cuda()
    current_state = State((args.batch_size, IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset][0],
                           IMAGE_SIZE[args.dataset][1]), args.norm, args.epsilon)
    fcn = MyFCN((IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset][0], IMAGE_SIZE[args.dataset][1]), args.n_actions)
    fcn.apply(fcn.init_weights)
    optimizer = Adam(fcn.parameters(),lr=args.learning_rate)
    agent = PixelWiseA3C(fcn, optimizer, args.episode_len, args.gamma, act_deterministically=True)
    agent.model.cuda()
    agent.model.train()
    agent.shared_model.train()
    agent.shared_model.cuda()
    i = 0
    episode = 0
    save_model_path = "{}/train_pytorch_model/sign_player/2D_{}_untargeted_{}_attack_on_{}.pth.tar".format(PY_ROOT, args.dataset, args.norm,
                                                                                             args.model)
    os.makedirs(os.path.dirname(save_model_path),exist_ok=True)
    log_path =  "{}/train_pytorch_model/sign_player/train_2D_{}_untargeted_{}_attack_on_{}.log".format(PY_ROOT, args.dataset, args.norm,
                                                                                             args.model)
    set_log_file(log_path)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_path))
    log.info('Called with args:')
    print_args(args)

    log.info("The trained model file will be saved to {}".format(save_model_path))
    for epoch in range(args.epochs):
        for raw_x, true_labels in data_loader:
            episode += 1
            raw_x = raw_x.cuda()
            true_labels = true_labels.cuda()
            current_state.reset(raw_x)
            reward = torch.zeros(raw_x.size(0)).float()
            sum_reward = 0
            environment = Environment()
            environment.get_reward(target_model, raw_x, true_labels)  # initalize the first loss using clean images
            for t in range(0, args.episode_len):
                action = agent.act_and_train(current_state.image, reward)

                current_state.step(action)  # 得到的不是一张图，而是一个batch image
                reward = environment.get_reward(target_model, current_state.image, true_labels)  # shape= (batch_size,)
                # reward is a map

                sum_reward += reward.mean().item() * np.power(args.gamma, t)
            agent.stop_episode_and_train(current_state.image, reward, True)
            if episode % 100 == 0:
                log.info("{e}-th episode trained over, total reward {a}".format(e=episode, a=sum_reward))
                test(test_data_loader, agent, target_model, episode, args)

            # adjust_learning_rate(optimizer, args.learning_rate, episode, args.n_episodes)
        # test(test_data_loader, agent, target_model,epoch+1, args)
        torch.save({"epoch":epoch+1, "state_dict": agent.model.state_dict(),
                    "optimizer":agent.optimizer.state_dict()}, save_model_path)
        log.info("The {}-th epoch is trained over!".format(epoch))
    log.info("The training is completely over for {} epochs!".format(args.epochs))
        