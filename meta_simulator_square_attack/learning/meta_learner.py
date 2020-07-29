import os
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from meta_simulator_square_attack.learning.task_dataset import MetaTaskDataset
import random
from torch.utils.data import DataLoader
from cifar_models_myself import *
from meta_simulator_learning.meta_network import MetaNetwork
import numpy as np
from meta_simulator_square_attack.learning.inner_loop import InnerLoop
from torch.optim import Adam
from torchvision import models as torch_models
from meta_simulator_square_attack.learning.tinyimagenet_models import densenet161, densenet121, densenet169, densenet201,\
            resnext101_32x4d, resnext101_64x4d,inception_v3
from tiny_imagenet_models.wrn import tiny_imagenet_wrn
from cifar_models_myself.miscellaneous import Identity
import torchvision.models as vision_models
from config import IN_CHANNELS, IMAGE_SIZE, PY_ROOT

class MetaLearner(object):
    def __init__(self, dataset, arch, meta_batch_size, meta_step_size,
                 inner_step_size, lr_decay_itr, epoch, num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, data_loss_type, adv_norm, targeted, target_type, without_resnet):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_step_size = inner_step_size
        self.lr_decay_itr = lr_decay_itr
        self.epoch = epoch
        self.num_inner_updates = num_inner_updates
        self.load_task_mode  = load_task_mode
        backbone = self.construct_model(arch, dataset)
        self.network = MetaNetwork(backbone)
        self.network.cuda()
        trn_dataset = MetaTaskDataset(dataset, adv_norm, data_loss_type, tot_num_tasks, load_task_mode, protocol, targeted,
                                      target_type, without_resnet)
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.loss_fn = nn.MSELoss()
        self.fast_net = InnerLoop(self.network, self.num_inner_updates,self.inner_step_size, self.meta_batch_size)  # 并行执行每个task
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)

    def construct_model(self, arch, dataset):
        if dataset in ["CIFAR-10", "CIFAR-100","MNIST","FashionMNIST"]:
            network = self.construct_cifar_model(arch, dataset)
        elif dataset == "TinyImageNet":
            network = self.construct_tiny_imagenet_model(arch, dataset)
        elif dataset == "ImageNet":
            network = self.construct_imagenet_model(arch, dataset)
        return network

    def construct_tiny_imagenet_model(self, arch, dataset):
        if not arch.startswith("densenet") and not arch.startswith("resnext") and arch in torch_models.__dict__:
            network = torch_models.__dict__[arch](pretrained=False)
        num_classes = 1
        if arch.startswith("resnet"):
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, num_classes)
        elif arch.startswith("densenet"):
            if arch == "densenet161":
                network = densenet161(pretrained=False)
            elif arch == "densenet121":
                network = densenet121(pretrained=False)
            elif arch == "densenet169":
                network = densenet169(pretrained=False)
            elif arch == "densenet201":
                network = densenet201(pretrained=False)
        elif arch == "resnext32_4":
            network = resnext101_32x4d(pretrained=None)
        elif arch == "resnext64_4":
            network = resnext101_64x4d(pretrained=None)
        elif arch.startswith("inception"):
            network = inception_v3(pretrained=False)
        elif arch == "WRN-28-10-drop":
            network = tiny_imagenet_wrn(in_channels=IN_CHANNELS[dataset],depth=28,num_classes=1,widen_factor=10, dropRate=0.3)
        elif arch == "WRN-40-10-drop":
            network = tiny_imagenet_wrn(in_channels=IN_CHANNELS[dataset], depth=40, num_classes=1,
                                        widen_factor=10, dropRate=0.3)
        elif arch.startswith("vgg"):
            network.avgpool = Identity()
            network.classifier[0] = nn.Linear(512 * 2 * 2, 4096)  # 64 /2**5 = 2
            network.classifier[-1] = nn.Linear(4096, num_classes)
        return network

    def construct_imagenet_model(self, arch):
        os.environ["TORCH_HOME"] = "{}/train_pytorch_model/real_image_model/ImageNet-pretrained".format(PY_ROOT)
        model = vision_models.__dict__[arch](pretrained=False, num_classes=1)
        return model

    def construct_cifar_model(self, arch, dataset):
        if arch == "conv3":
            network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], 1)
        elif arch == "densenet121":
            network = DenseNet121(IN_CHANNELS[dataset], 1)
        elif arch == "densenet169":
            network = DenseNet169(IN_CHANNELS[dataset], 1)
        elif arch == "densenet201":
            network = DenseNet201(IN_CHANNELS[dataset], 1)
        elif arch == "googlenet":
            network = GoogLeNet(IN_CHANNELS[dataset], 1)
        elif arch == "mobilenet":
            network = MobileNet(IN_CHANNELS[dataset], 1)
        elif arch == "mobilenet_v2":
            network = MobileNetV2(IN_CHANNELS[dataset], 1)
        elif arch == "resnet18":
            network = ResNet18(IN_CHANNELS[dataset], 1)
        elif arch == "resnet34":
            network = ResNet34(IN_CHANNELS[dataset], 1)
        elif arch == "resnet50":
            network = ResNet50(IN_CHANNELS[dataset], 1)
        elif arch == "resnet101":
            network = ResNet101(IN_CHANNELS[dataset], 1)
        elif arch == "resnet152":
            network = ResNet152(IN_CHANNELS[dataset], 1)
        elif arch == "pnasnetA":
            network = PNASNetA(IN_CHANNELS[dataset], 1)
        elif arch == "pnasnetB":
            network = PNASNetB(IN_CHANNELS[dataset], 1)
        elif arch == "efficientnet":
            network = EfficientNetB0(IN_CHANNELS[dataset], 1)
        elif arch == "dpn26":
            network = DPN26(IN_CHANNELS[dataset], 1)
        elif arch == "dpn92":
            network = DPN92(IN_CHANNELS[dataset], 1)
        elif arch == "resnext29_2":
            network = ResNeXt29_2x64d(IN_CHANNELS[dataset], 1)
        elif arch == "resnext29_4":
            network = ResNeXt29_4x64d(IN_CHANNELS[dataset], 1)
        elif arch == "resnext29_8":
            network = ResNeXt29_8x64d(IN_CHANNELS[dataset], 1)
        elif arch == "resnext29_32":
            network = ResNeXt29_32x4d(IN_CHANNELS[dataset], 1)
        elif arch == "senet18":
            network = SENet18(IN_CHANNELS[dataset], 1)
        elif arch == "shufflenet_G2":
            network = ShuffleNetG2(IN_CHANNELS[dataset], 1)
        elif arch == "shufflenet_G3":
            network = ShuffleNetG3(IN_CHANNELS[dataset], 1)
        elif arch == "vgg11":
            network = vgg11(IN_CHANNELS[dataset], 1)
        elif arch == "vgg13":
            network = vgg13(IN_CHANNELS[dataset], 1)
        elif arch == "vgg16":
            network = vgg16(IN_CHANNELS[dataset], 1)
        elif arch == "vgg19":
            network = vgg19(IN_CHANNELS[dataset], 1)
        elif arch == "preactresnet18":
            network = PreActResNet18(IN_CHANNELS[dataset], 1)
        elif arch == "preactresnet34":
            network = PreActResNet34(IN_CHANNELS[dataset], 1)
        elif arch == "preactresnet50":
            network = PreActResNet50(IN_CHANNELS[dataset], 1)
        elif arch == "preactresnet101":
            network = PreActResNet101(IN_CHANNELS[dataset], 1)
        elif arch == "preactresnet152":
            network = PreActResNet152(IN_CHANNELS[dataset], 1)
        elif arch == "wideresnet28":
            network = wideresnet28(IN_CHANNELS[dataset], 1)
        elif arch == "wideresnet28drop":
            network = wideresnet28drop(IN_CHANNELS[dataset], 1)
        elif arch == "wideresnet34":
            network = wideresnet34(IN_CHANNELS[dataset], 1)
        elif arch == "wideresnet34drop":
            network = wideresnet34drop(IN_CHANNELS[dataset], 1)
        elif arch == "wideresnet40":
            network = wideresnet40(IN_CHANNELS[dataset], 1)
        elif arch == "wideresnet40drop":
            network = wideresnet40drop(IN_CHANNELS[dataset], 1)
        elif arch == "carlinet":
            network = carlinet(IN_CHANNELS[dataset], 1)
        return network


    def forward_pass(self, network, input, target):
        output = network.net_forward(input)
        loss = self.loss_fn(output, target)
        return loss, output

    def meta_update(self, grads, query_images, query_targets):
        dummy_input, dummy_target = query_images.cuda(), query_targets.cuda()  # B,C,H,W, # B, #class_num
        loss, output = self.forward_pass(self.network, dummy_input, dummy_target)  # 其实传谁无所谓，因为loss.backward调用的时候，会用外部更新的梯度的求和来替换掉loss.backward自己算出来的梯度值
        gradients = {k[len("network."):]: sum(d[k] for d in grads) for k in grads[0].keys()}  # 把N个task的grad加起来
        # Register a hook on each parameter in the net that replaces the current dummy grad with our grads accumulated across the meta-batch
        hooks = []
        for (k,v) in self.network.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))
        # Compute grads for current step, replace with summed gradients as defined by hook
        self.opt.zero_grad()
        loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()

    def train(self, model_path, resume_epoch=0):
        for epoch in range(resume_epoch, self.epoch):
            for i, (adv_images, gt_losses) in enumerate(self.train_loader):
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                seq_len = adv_images.size(1)
                meta_train_index_list = np.arange(seq_len//2).tolist()
                meta_test_index_list = np.arange(seq_len // 2, seq_len).tolist()
                meta_train_images = adv_images[:, meta_train_index_list, :, :, :]  #B,T,C,H,W
                meta_test_images = adv_images[:, meta_test_index_list, :, :, :]
                meta_train_losses = gt_losses[:, meta_train_index_list]   # B,T
                meta_test_losses = gt_losses[:, meta_test_index_list]
                for task_idx in range(meta_train_images.size(0)):  # 每个task的teacher model不同，所以
                    task_meta_train_images = meta_train_images[task_idx].cuda() # T, C, H, W
                    task_meta_test_images = meta_test_images[task_idx].cuda()
                    task_meta_train_losses = meta_train_losses[task_idx].unsqueeze(1).cuda()  # T,1
                    task_meta_test_losses = meta_test_losses[task_idx].unsqueeze(1).cuda()  # T,1
                    self.fast_net.copy_weights(self.network)
                    g = self.fast_net.forward(task_meta_train_images, task_meta_test_images, task_meta_train_losses,
                                              task_meta_test_losses)
                    grads.append(g)
                # Perform the meta update
                dummy_meta_test_images = meta_test_images[0].cuda()
                dummy_meta_test_losses = meta_test_losses[0].unsqueeze(1).cuda()
                self.meta_update(grads, dummy_meta_test_images, dummy_meta_test_losses)
                grads.clear()
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.network.state_dict(),
                'optimizer': self.opt.state_dict(),
            }, model_path)


    def adjust_learning_rate(self,itr, meta_lr, lr_decay_itr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if lr_decay_itr > 0:
            if int(itr % lr_decay_itr) == 0 and itr > 0:
                meta_lr = meta_lr / (10 ** int(itr / lr_decay_itr))
                self.fast_net.step_size = self.fast_net.step_size / 10
                for param_group in self.opt.param_groups:
                    param_group['lr'] = meta_lr
