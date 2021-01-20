import sys

from torch.optim import Adam

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from meta_simulator_benign_images.meta_training.image_logits_dataset import MetaTaskDataset
from torch.utils.data import DataLoader
from cifar_models_myself import *
from meta_simulator_benign_images.meta_training.meta_network import MetaNetwork
from meta_simulator_benign_images.meta_training.reptile_inner_loop import ReptileInnerLoop
from dataset.standard_model import MetaLearnerModelBuilder

class ReptileMetaLearner(object):
    def __init__(self, dataset, arch, meta_batch_size, meta_step_size,
                 inner_step_size, lr_decay_itr, epoch, num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, num_support, without_resnet):
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
        trn_dataset = MetaTaskDataset(dataset, tot_num_tasks, num_support, load_task_mode, protocol, without_resnet)
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.loss_fn = nn.MSELoss()
        self.fast_net = ReptileInnerLoop(self.network, self.num_inner_updates, self.inner_step_size, self.meta_batch_size)  # 并行执行每个task
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

    def compute_meta_weight(self, task_diff_weights):
        '''
        meta optim for reptile
        this method will update the self.net.parameters according to the
        reveived weights, which is the updating directions. The updata learning
        rate is self.update.lr
        '''
        dic = self.network.state_dict()
        for param_name, each in task_diff_weights[0].items():
            diff = torch.zeros_like(each)
            for j in range(len(task_diff_weights)):
                diff += task_diff_weights[j][param_name]  # j is task index
            diff /= len(task_diff_weights)
            dic[param_name] -= diff
        self.network.load_state_dict(dic)


    def train(self, model_path, resume_epoch=0):
        for epoch in range(resume_epoch, self.epoch):
            for i, (benign_images, gt_logits) in enumerate(self.train_loader):
                # q1_images shape = (Task_num, T, C, H, W)  q2_images shape = (Task_num, T, C, H, W)
                # q1_logits shape = (Task_num, T, #class), q2_logits shape = (Task_num, T, #class)
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                task_diff_weights = []
                support_images = benign_images
                support_target_logits = gt_logits
                for task_idx in range(support_images.size(0)):  # 每个task的teacher model不同，所以
                    task_support_images = support_images[task_idx].cuda() # T, C, H, W
                    task_support_target_logits = support_target_logits[task_idx].cuda()
                    self.fast_net.copy_weights(self.network)
                    current_task_diff = self.fast_net.forward(task_support_images, task_support_target_logits)
                    task_diff_weights.append(current_task_diff)
                self.compute_meta_weight(task_diff_weights)
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
