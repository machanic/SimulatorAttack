import glob
import random
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import os
from config import PY_ROOT, IN_CHANNELS, CLASS_NUM, IMAGE_SIZE, ALL_MODELS
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.meta_img_grad_dataset import MetaTaskDataset
from cifar_models import *
from meta_model.inner_loop import InnerLoop
from meta_model.tensorboard_helper import TensorBoardWriter
from meta_model.meta_network import MetaNetwork
import numpy as np
from torch.nn import functional as F
from torchvision import models as torch_models
from tiny_imagenet_models.densenet import densenet121, densenet161, densenet169, densenet201
from tiny_imagenet_models.miscellaneous import Identity
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d

class MetaDistillationLearner(object):
    def __init__(self, dataset, arch, meta_batch_size, meta_step_size,
                 inner_step_size, lr_decay_itr,
                 epoch, num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, num_support, distill_loss_type, data_loss_type, tensorboard_data_prefix):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_step_size = inner_step_size
        self.lr_decay_itr = lr_decay_itr
        self.epoch = epoch
        self.num_inner_updates = num_inner_updates
        backbone = self.construct_model(arch, dataset)
        self.network = MetaNetwork(backbone)
        self.network.cuda()
        model_load_path = "{}/train_pytorch_model/real_image_model/{}@{}@epoch_*.pth.tar".format(
            PY_ROOT, self.dataset, arch)
        model_load_path = glob.glob(model_load_path)[0]
        assert os.path.exists(model_load_path), model_load_path
        self.network.network.load_state_dict(torch.load(model_load_path,
                                                        map_location=lambda storage, location: storage)["state_dict"])

        self.num_support = num_support
        trn_dataset = MetaTaskDataset(data_loss_type, tot_num_tasks, dataset, load_mode=load_task_mode, protocol=protocol)
        # task number per mini-batch is controlled by DataLoader
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.tensorboard = TensorBoardWriter("{0}/tensorboard/distillation".format(PY_ROOT),
                                             tensorboard_data_prefix)
        os.makedirs("{0}/tensorboard/distillation".format(PY_ROOT), exist_ok=True)
        self.loss_fn = self._make_criterion(mode=distill_loss_type)
        self.fast_net = InnerLoop(self.network, self.num_inner_updates,
                                  self.inner_step_size, self.meta_batch_size, self.loss_fn)
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)
        self.arch_pool = {}


    def _make_criterion(self, T=1.0, mode='CSE'):
        def criterion(outputs, targets):
            if mode == 'CSE':
                _p = F.log_softmax(outputs / T, dim=1)
                _q = F.softmax(targets / T, dim=1)
                _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
            elif mode == 'MSE':
                _p = F.softmax(outputs / T, dim=1)
                _q = F.softmax(targets / T, dim=1)
                _soft_loss = nn.MSELoss()(_p, _q) / 2
            else:
                raise NotImplementedError()

            _soft_loss = _soft_loss * T * T
            return _soft_loss

        return criterion

    def forward_pass(self, network, input, target):
        output = network.net_forward(input)
        loss = self.loss_fn(output, target)
        return loss, output

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

    def get_arch(self, arch):
        if arch in self.arch_pool:
            return self.arch_pool[arch]
        model = self.construct_model(arch, self.dataset)
        model_load_path = "{}/train_pytorch_model/real_image_model/{}@{}@epoch_*@lr_*@batch_*.pth.tar".format(
            PY_ROOT, self.dataset, arch)
        model_load_path = glob.glob(model_load_path)[0]
        assert os.path.exists(model_load_path), model_load_path
        model.load_state_dict(torch.load(model_load_path, map_location=lambda storage, location: storage)["state_dict"])
        model.eval()
        model.cuda()
        self.arch_pool[arch] = model
        return model

    def meta_update(self, grads, query_images, query_targets):
        dummy_input, dummy_target = query_images, query_targets  # B,T,C,H,W
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

    def evaluate_query_set(self, model, query_adv_images, query_target, fast_weights):  # B,C,H,W;  query_target = B,#class_num
        _, query_target_labels = torch.max(query_target, dim=1)
        query_output = model.net_forward(query_adv_images, fast_weights)
        query_predict = query_output.max(1)[1]
        accuracy = (query_target_labels.eq(query_predict.long())).sum() / float(query_target.size(0))
        accuracy = accuracy.float()
        mse_error = F.mse_loss(query_output, query_target)
        return accuracy.detach().cpu(), mse_error.detach().cpu()

    def train(self, model_path, resume_epoch=0):
        for epoch in range(resume_epoch, self.epoch):
            for i, (adv_images, _, archs) in enumerate(self.train_loader):
                # adv_images shape = (Task_num, T,C,H,W)  grad_images shape = (Task_num, T, C, H, W)
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                adv_images = adv_images.cuda()
                seq_len = adv_images.size(1)
                support_index_list = sorted(random.sample(range(seq_len // 2), self.num_support))
                query_index_list = np.arange(seq_len // 2, seq_len).tolist()
                support_adv_images = adv_images[:,support_index_list,:,:,:]
                query_adv_images = adv_images[:,query_index_list,:,:,:]  # (Task_num, T, C, H, W)
                _, _, C, H, W = support_adv_images.size()
                all_tasks_accuracy = []
                all_tasks_mse_error = []
                for task_idx in range(adv_images.size(0)):
                    arch_name = ALL_MODELS[archs[task_idx].item()]
                    teacher_model = self.get_arch(arch_name)
                    task_support_adv_images = support_adv_images[task_idx]  #  T, C, H, W
                    task_query_adv_images = query_adv_images[task_idx]
                    with torch.no_grad():
                        task_support_target = teacher_model.forward(task_support_adv_images).detach()
                        task_query_target = teacher_model.forward(task_query_adv_images).detach()
                    self.fast_net.copy_weights(self.network)
                    # fast_net only forward one task's data
                    g, fast_weights = self.fast_net.forward(task_support_adv_images, task_query_adv_images,
                                              task_support_target, task_query_target)
                    grads.append(g)
                    query_images = task_query_adv_images  # B, C,H,W
                    query_targets = task_query_target  # B, #class
                    with torch.no_grad():
                        accuracy, mse_error = self.evaluate_query_set(self.fast_net.network, query_images, query_targets,
                                                                      fast_weights)
                        fast_weights.clear()
                    all_tasks_accuracy.append(accuracy)
                    all_tasks_mse_error.append(mse_error)
                # Perform the meta update
                self.meta_update(grads, query_images, query_targets)
                grads.clear()
                # 不同的task teacher model不一样，不可以直接这么算准确度
                # accuracy, logits_mse_error = self.evaluate_query_set(self.network, query_images, query_targets)
                if itr % 100 == 0:
                    self.tensorboard.record_trn_query_loss(torch.stack(all_tasks_mse_error).mean().detach().cpu(), itr)
                    self.tensorboard.record_trn_query_acc(torch.stack(all_tasks_accuracy).mean().detach().cpu(), itr)
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