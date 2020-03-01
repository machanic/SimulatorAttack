import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from dataset.meta_two_queries_dataset import TwoQueriesBatchImagesMetaTaskDataset
from torch.utils.data import DataLoader
from meta_simulator_model_batch_images.meta_network import MetaNetwork
from meta_simulator_model_batch_images.inner_loop_pair_loss import InnerLoopPairLoss
from torch.optim import Adam
from torch import nn
from dataset.model_constructor import MetaLearnerModelBuilder
import torch

class MetaTwoQueriesLearner(object):
    def __init__(self, dataset, arch, meta_batch_size, inner_batch_size, meta_step_size,
                 inner_step_size, lr_decay_itr, epoch, num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, num_support, data_loss_type, loss_type, adv_norm,targeted, target_type, use_softmax):
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
        self.num_support = num_support
        self.use_softmax = use_softmax
        trn_dataset = TwoQueriesBatchImagesMetaTaskDataset(dataset, inner_batch_size, adv_norm, data_loss_type,
                                                           tot_num_tasks, load_task_mode, protocol, targeted, target_type)
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.loss_fn = nn.MSELoss()
        self.fast_net = InnerLoopPairLoss(self.network, self.num_inner_updates,
                                  self.inner_step_size, self.meta_batch_size, loss_type, use_softmax)  # 并行执行每个task
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)

    def construct_model(self, arch, dataset):
        if dataset in ["CIFAR-10", "CIFAR-100","MNIST","FashionMNIST"]:
            network = MetaLearnerModelBuilder.construct_cifar_model(arch, dataset)
        elif dataset == "TinyImageNet":
            network = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
        elif dataset == "ImageNet":
            network = MetaLearnerModelBuilder.construct_imagenet_model(arch, dataset)
        return network

    def forward_pass(self, network, input, target):
        output = network.net_forward(input)
        loss = self.loss_fn(output, target)
        return loss, output

    def meta_update(self, grads, query_images, query_targets):
        dummy_input, dummy_target = query_images.cuda(), query_targets.cuda()  # B,C,H,W, # B, #class_num
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
            for i, (support_q1_images, query_q1_images, support_q2_images, query_q2_images, support_q1_logits, query_q1_logits,
                    support_q2_logits, query_q2_logits) in enumerate(self.train_loader):
                # q1_images shape = (Task_num, B, C, H, W)  q2_images shape = (Task_num, B, C, H, W)
                # q1_logits shape = (Task_num, B, #class), q2_logits shape = (Task_num, B, #class)
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                for task_idx in range(support_q1_images.size(0)):  # 每个task的teacher model不同，所以
                    task_support_q1 = support_q1_images[task_idx].cuda() # T, C, H, W
                    task_support_q2 = support_q2_images[task_idx].cuda()
                    task_query_q1 = query_q1_images[task_idx].cuda()
                    task_query_q2 = query_q2_images[task_idx].cuda()
                    task_support_q1_logits = support_q1_logits[task_idx].cuda()
                    task_support_q2_logits = support_q2_logits[task_idx].cuda()
                    task_query_q1_logits = query_q1_logits[task_idx].cuda()
                    task_query_q2_logits = query_q2_logits[task_idx].cuda()
                    self.fast_net.copy_weights(self.network)
                    g = self.fast_net.forward(task_support_q1, task_support_q2, task_query_q1, task_query_q2,
                                              task_support_q1_logits, task_support_q2_logits,
                                              task_query_q1_logits, task_query_q2_logits)
                    grads.append(g)
                    support_q1_images[task_idx].cpu()  # T, C, H, W
                    support_q2_images[task_idx].cpu()
                    query_q1_images[task_idx].cpu()
                    query_q2_images[task_idx].cpu()
                    support_q1_logits[task_idx].cpu()
                    support_q2_logits[task_idx].cpu()
                    query_q1_logits[task_idx].cpu()
                    query_q2_logits[task_idx].cpu()
                # Perform the meta update
                dummy_query_images = query_q1_images[0]
                dummy_query_targets = query_q1_logits[0]
                self.meta_update(grads, dummy_query_images, dummy_query_targets)
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
