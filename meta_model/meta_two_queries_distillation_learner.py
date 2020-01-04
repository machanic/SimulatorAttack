import sys

from optimizer.radam import RAdam

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from dataset.meta_two_queries_dataset import TwoQueriesMetaTaskDataset
import glob
import random
import os
from config import PY_ROOT, IN_CHANNELS, CLASS_NUM, IMAGE_SIZE
from torch.optim import Adam
from torch.utils.data import DataLoader
from cifar_models import *
from meta_model.tensorboard_helper import TensorBoardWriter
from meta_model.meta_network import MetaNetwork
import numpy as np
from torch.nn import functional as F
from inner_loop_pair_loss import InnerLoopPairLoss
from torchvision import models as torch_models
from tiny_imagenet_models.densenet import densenet121, densenet161, densenet169, densenet201
from tiny_imagenet_models.miscellaneous import Identity
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d

class MetaTwoQueriesLearner(object):
    def __init__(self, dataset, arch, meta_batch_size, meta_step_size,
                 inner_step_size, lr_decay_itr, epoch, num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, num_support, data_attack_type, tensorboard_data_prefix):
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
        model_load_path = "{}/train_pytorch_model/real_image_model/{}@{}@*.pth.tar".format(
            PY_ROOT, self.dataset, arch)
        model_load_path = glob.glob(model_load_path)[0]
        assert os.path.exists(model_load_path), model_load_path
        backbone.load_state_dict(torch.load(model_load_path,map_location=lambda storage, location: storage)["state_dict"])
        self.network = MetaNetwork(backbone)
        self.network.cuda()
        self.num_support = num_support
        trn_dataset = TwoQueriesMetaTaskDataset(data_attack_type, tot_num_tasks, dataset, load_mode=load_task_mode, protocol=protocol)
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.tensorboard = TensorBoardWriter("{0}/tensorboard/2q_distillation".format(PY_ROOT),
                                             tensorboard_data_prefix)
        os.makedirs("{0}/tensorboard/2q_distillation".format(PY_ROOT), exist_ok=True)
        self.loss_fn = nn.MSELoss()
        self.fast_net = InnerLoopPairLoss(self.network, self.num_inner_updates,
                                  self.inner_step_size, self.meta_batch_size)  # 并行执行每个task
        self.fast_net.cuda()
        self.opt = RAdam(self.network.parameters(), lr=meta_step_size)
        self.arch_pool = {}


    def construct_model(self, arch, dataset):
        if dataset != "TinyImageNet":
            if arch == "conv3":
                network = Conv3(IN_CHANNELS[dataset], IMAGE_SIZE[dataset], CLASS_NUM[dataset])
            elif arch == "densenet121":
                network = DenseNet121(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "densenet169":
                network = DenseNet169(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "densenet201":
                network = DenseNet201(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "googlenet":
                network = GoogLeNet(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "mobilenet":
                network = MobileNet(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "mobilenet_v2":
                network = MobileNetV2(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "resnet18":
                network = ResNet18(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "resnet34":
                network = ResNet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "resnet50":
                network = ResNet50(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "resnet101":
                network = ResNet101(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "resnet152":
                network = ResNet152(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "pnasnetA":
                network = PNASNetA(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "pnasnetB":
                network = PNASNetB(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "efficientnet":
                network = EfficientNetB0(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "dpn26":
                network = DPN26(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "dpn92":
                network = DPN92(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "resnext29_2":
                network = ResNeXt29_2x64d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "resnext29_4":
                network = ResNeXt29_4x64d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "resnext29_8":
                network = ResNeXt29_8x64d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "resnext29_32":
                network = ResNeXt29_32x4d(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "senet18":
                network = SENet18(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "shufflenet_G2":
                network = ShuffleNetG2(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "shufflenet_G3":
                network = ShuffleNetG3(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "vgg11":
                network = vgg11(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "vgg13":
                network = vgg13(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "vgg16":
                network = vgg16(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "vgg19":
                network = vgg19(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "preactresnet18":
                network = PreActResNet18(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "preactresnet34":
                network = PreActResNet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "preactresnet50":
                network = PreActResNet50(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "preactresnet101":
                network = PreActResNet101(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "preactresnet152":
                network = PreActResNet152(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "wideresnet28":
                network = wideresnet28(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "wideresnet34":
                network = wideresnet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
            elif arch == "wideresnet40":
                network = wideresnet40(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        else:
            if arch in torch_models.__dict__:
                network = torch_models.__dict__[arch](pretrained=True)
            num_classes = CLASS_NUM[dataset]
            if arch.startswith("resnet"):
                num_ftrs = network.fc.in_features
                network.fc = nn.Linear(num_ftrs, num_classes)
            elif arch.startswith("densenet"):
                if arch == "densenet161":
                    network = densenet161(pretrained=True)
                elif arch == "densenet121":
                    network = densenet121(pretrained=True)
                elif arch == "densenet169":
                    network = densenet169(pretrained=True)
                elif arch == "densenet201":
                    network = densenet201(pretrained=True)
            elif arch == "resnext32_4":
                network = resnext101_32x4d(pretrained="imagenet")
            elif arch == "resnext64_4":
                network = resnext101_64x4d(pretrained="imagenet")
            elif arch.startswith("vgg"):
                network.avgpool = Identity()
                network.classifier[0] = nn.Linear(512 * 2 * 2, 4096)  # 64 /2**5 = 2
                network.classifier[-1] = nn.Linear(4096, num_classes)
        return network

    def forward_pass(self, network, input, target):
        output = network.net_forward(input)
        loss = self.loss_fn(output, target)
        return loss, output

    def meta_update(self, grads, query_images, query_targets):
        dummy_input, dummy_target = query_images, query_targets  # B,C,H,W, # B, #class_num
        # We use a dummy forward / backward pass to get the correct grads into self.net
        loss, output = self.forward_pass(self.network, dummy_input, dummy_target)  # 其实传谁无所谓，因为loss.backward调用的时候，会用外部更新的梯度的求和来替换掉loss.backward自己算出来的梯度值
        # Unpack the list of grad dicts
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
            for i, (q1_images, q2_images, q1_logits, q2_logits, gt_label, _, _) in enumerate(self.train_loader):
                # q1_images shape = (Task_num, T, C, H, W)  q2_images shape = (Task_num, T, C, H, W)
                # q1_logits shape = (Task_num, T, #class), q2_logits shape = (Task_num, T, #class)
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                q1_images = q1_images.cuda()
                q2_images = q2_images.cuda()
                q1_logits = q1_logits.cuda()
                q2_logits = q2_logits.cuda()
                seq_len = q1_images.size(1)
                support_index_list = sorted(random.sample(range(seq_len // 2), self.num_support))
                query_index_list = np.arange(seq_len // 2, seq_len).tolist()
                support_q1_images = q1_images[:, support_index_list, :, :, :]
                support_q2_images = q2_images[:, support_index_list, :, :, :]
                query_q1_images = q1_images[:, query_index_list, :, :, :]
                query_q2_images = q2_images[:, query_index_list, :, :, :]  # (Task_num, T, C, H, W)
                support_q1_logits = q1_logits[:, support_index_list, :]
                support_q2_logits = q2_logits[:, support_index_list, :]  # (Task_num, T, #class)
                query_q1_logits = q1_logits[:, query_index_list, :]
                query_q2_logits = q2_logits[:, query_index_list,:]  # (Task_num, T, #class)
                all_tasks_accuracy = []
                all_tasks_mse_error = []
                for task_idx in range(q1_images.size(0)):  # 每个task的teacher model不同，所以
                    task_support_q1 = support_q1_images[task_idx] # T, C, H, W
                    task_support_q2 = support_q2_images[task_idx]
                    task_query_q1 = query_q1_images[task_idx]
                    task_query_q2 = query_q2_images[task_idx]
                    task_support_q1_logits = support_q1_logits[task_idx]
                    task_support_q2_logits = support_q2_logits[task_idx]
                    task_query_q1_logits = query_q1_logits[task_idx]
                    task_query_q2_logits = query_q2_logits[task_idx]
                    self.fast_net.copy_weights(self.network)
                    # fast_net only forward one task's data
                    g, accuracy, mse_error = self.fast_net.forward(task_support_q1, task_support_q2, task_query_q1, task_query_q2,
                                              task_support_q1_logits, task_support_q2_logits,
                                              task_query_q1_logits, task_query_q2_logits)
                    grads.append(g)
                    all_tasks_accuracy.append(accuracy)
                    all_tasks_mse_error.append(mse_error)
                # Perform the meta update
                dummy_query_images = query_q1_images[0]
                dummy_query_targets = query_q1_logits[0]
                self.meta_update(grads, dummy_query_images, dummy_query_targets)
                grads.clear()
                if itr % 100 == 0 and itr > 0:
                    self.tensorboard.record_trn_query_loss(torch.stack(all_tasks_mse_error).float().mean().detach().cpu(), itr)
                    # self.tensorboard.record_trn_query_acc(torch.stack(all_tasks_accuracy).float().mean().detach().cpu(), itr)
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
