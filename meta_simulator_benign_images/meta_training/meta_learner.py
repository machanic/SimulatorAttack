import sys

from torch.optim import Adam

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from meta_simulator_benign_images.meta_training.image_logits_dataset import MetaTaskDataset
from torch.utils.data import DataLoader
from cifar_models_myself import *
from meta_simulator_benign_images.meta_training.meta_network import MetaNetwork
from meta_simulator_benign_images.meta_training.inner_loop import InnerLoop
from dataset.standard_model import MetaLearnerModelBuilder

class MetaLearner(object):
    def __init__(self, dataset, arch, meta_batch_size, meta_step_size,
                 inner_step_size, lr_decay_itr, epoch, num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, num_support, num_query, without_resnet):
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
        self.num_query = num_query
        trn_dataset = MetaTaskDataset(dataset, tot_num_tasks, num_support+num_query, load_task_mode, protocol, without_resnet)
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.loss_fn = nn.MSELoss()
        self.fast_net = InnerLoop(self.network, self.num_inner_updates,
                                  self.inner_step_size, self.meta_batch_size)  # 并行执行每个task
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)
        self.arch_pool = {}

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
            for i, (benign_images, gt_logits) in enumerate(self.train_loader):
                # q1_images shape = (Task_num, T, C, H, W)  q2_images shape = (Task_num, T, C, H, W)
                # q1_logits shape = (Task_num, T, #class), q2_logits shape = (Task_num, T, #class)
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                support_images = benign_images[:, :self.num_support]
                support_target_logits = gt_logits[:,:self.num_support]
                query_images = benign_images[:, self.num_support:]
                query_target_logits = gt_logits[:, self.num_support:]

                for task_idx in range(support_images.size(0)):  # 每个task的teacher model不同，所以
                    task_support_images = support_images[task_idx].cuda() # T, C, H, W
                    task_query_images = query_images[task_idx].cuda()
                    task_support_target_logits = support_target_logits[task_idx].cuda()
                    task_query_target_logits = query_target_logits[task_idx].cuda()
                    self.fast_net.copy_weights(self.network)
                    g = self.fast_net.forward(task_support_images, task_query_images, task_support_target_logits,task_query_target_logits)
                    grads.append(g)
                # Perform the meta update
                dummy_query_images = query_images[0]
                dummy_query_targets = query_target_logits[0]
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
