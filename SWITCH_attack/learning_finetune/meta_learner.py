import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from SWITCH_attack.learning_finetune.task_dataset import MetaTaskDataset
from torch.utils.data import DataLoader
from SWITCH_attack.learning_finetune.meta_network import MetaNetwork
from torch import nn
from SWITCH_attack.learning_finetune.inner_loop import InnerLoop
from torch.optim import Adam
from dataset.standard_model import MetaLearnerModelBuilder
import torch
from torch.nn import functional as F
import numpy as np

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
        self.targeted = targeted
        backbone = self.construct_model(arch, dataset)
        backbone.apply(self.remove_fc_bias)
        self.network = MetaNetwork(backbone)
        self.network.cuda()
        trn_dataset = MetaTaskDataset(dataset, adv_norm, data_loss_type, tot_num_tasks, load_task_mode, protocol, targeted,
                                      target_type, without_resnet)
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.loss_fn = nn.MSELoss()
        self.fast_net = InnerLoop(self.network, self.num_inner_updates,self.inner_step_size, self.meta_batch_size,data_loss_type)  # 并行执行每个task
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)

    def remove_fc_bias(self, module):
        if isinstance(module, nn.Linear):
            module.register_parameter('bias', None)
            module.reset_parameters()

    def construct_model(self, arch, dataset):
        if dataset in ["CIFAR-10", "CIFAR-100", "MNIST", "FashionMNIST"]:
            network = MetaLearnerModelBuilder.construct_cifar_model(arch, dataset)
        elif dataset == "TinyImageNet":
            network = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
        elif dataset == "ImageNet":
            network = MetaLearnerModelBuilder.construct_imagenet_model(arch, dataset)
        return network

    def xent_loss(self, logit, label, target=None):
        if target is not None:
            return -F.cross_entropy(logit, target, reduction='mean')
        else:
            return F.cross_entropy(logit, label, reduction='mean')

    def forward_pass(self, network, input, true_label, target_label):
        input.requires_grad_()
        output = network.net_forward(input)
        loss = self.xent_loss(output, true_label, target_label)  # fake loss !
        return loss, output

    def meta_update(self, grads, query_images,  query_true_labels):
        query_true_labels = query_true_labels.cuda()
        query_images = query_images.cuda()  # B,C,H,W, # B, #class_num
        loss, output = self.forward_pass(self.network, query_images, query_true_labels, None)  # 其实传谁无所谓，因为loss.backward调用的时候，会用外部更新的梯度的求和来替换掉loss.backward自己算出来的梯度值
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
        task_meta_train_target_labels = None
        task_meta_test_target_labels = None
        for epoch in range(resume_epoch, self.epoch):
            for i, data_tuple in enumerate(self.train_loader):
                if self.targeted:
                    adv_images, gt_gradients, gt_logits, true_labels, target_labels = data_tuple
                else:
                    adv_images, gt_gradients, gt_logits, true_labels = data_tuple
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                seq_len = adv_images.size(1)
                support_index_list = np.arange(seq_len//2).tolist()
                query_index_list = np.arange(seq_len // 2, seq_len).tolist()

                meta_train_images = adv_images[:, support_index_list, :, :, :]  # Task,T,C,H,W
                meta_test_images = adv_images[:, query_index_list, :, :, :] # Task,T,C,H,W
                meta_train_gt_gradients = gt_gradients[:, support_index_list, :, :, :]
                meta_test_gt_gradients= gt_gradients[:, query_index_list, :, :, :]
                meta_train_logits = gt_logits[:,support_index_list,:]
                meta_test_logits = gt_logits[:,query_index_list,:]
                meta_train_true_labels = true_labels[:,support_index_list]  # Task, B
                meta_test_true_labels = true_labels[:,query_index_list]  # Task, B
                if self.targeted:
                    meta_train_target_labels = target_labels[:,support_index_list]  # Task, B
                    meta_test_target_labels = target_labels[:,query_index_list]   # Task, B

                for task_idx in range(meta_train_images.size(0)):  # 每个task的teacher model不同，所以
                    task_meta_train_images = meta_train_images[task_idx].cuda() # B, C, H, W
                    task_meta_test_images = meta_test_images[task_idx].cuda()  # B, C, H, W
                    task_meta_train_gt_gradients = meta_train_gt_gradients[task_idx].cuda()  # B, C, H, W
                    task_meta_test_gt_gradients = meta_test_gt_gradients[task_idx].cuda()  # B, C, H, W
                    task_meta_train_logits = meta_train_logits[task_idx].cuda()
                    task_meta_test_logits = meta_test_logits[task_idx].cuda()
                    task_meta_train_true_labels = meta_train_true_labels[task_idx].cuda()
                    task_meta_test_true_labels = meta_test_true_labels[task_idx].cuda()
                    if self.targeted:
                        task_meta_train_target_labels = meta_train_target_labels[task_idx].cuda() # B
                        task_meta_test_target_labels = meta_test_target_labels[task_idx].cuda()  # B
                    self.fast_net.copy_weights(self.network)

                    g = self.fast_net.forward(task_meta_train_images, task_meta_train_gt_gradients,
                                              task_meta_train_logits, task_meta_train_true_labels, task_meta_train_target_labels,
                                              task_meta_test_images, task_meta_test_gt_gradients,
                                              task_meta_test_logits, task_meta_test_true_labels, task_meta_test_target_labels)
                    grads.append(g)
                self.meta_update(grads, meta_test_images[0].cuda(), meta_test_true_labels[0].cuda())
                grads.clear()
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.network.state_dict(),
                'optimizer': self.opt.state_dict(),
            }, model_path)


    def adjust_learning_rate(self,itr, meta_lr, lr_decay_itr):
        """Sets the learning_finetune rate to the initial LR decayed by 10 every 30 epochs"""
        if lr_decay_itr > 0:
            if int(itr % lr_decay_itr) == 0 and itr > 0:
                meta_lr = meta_lr / (10 ** int(itr / lr_decay_itr))
                self.fast_net.step_size = self.fast_net.step_size / 10
                for param_group in self.opt.param_groups:
                    param_group['lr'] = meta_lr
