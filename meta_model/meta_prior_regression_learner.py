import random
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from dataset.meta_two_queries_dataset import TwoQueriesMetaTaskDataset
from meta_model.network.autoencoder import AutoEncoder
from meta_model.network.fcn_8s import FCN8s
from meta_model.network.unet import ResNetUNet
import os
from config import PY_ROOT, IN_CHANNELS, CLASS_NUM
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.meta_img_grad_dataset import MetaTaskDataset
from cifar_models import *
from meta_model.inner_loop import InnerLoop
from meta_model.tensorboard_helper import TensorBoardWriter
from meta_model.meta_network import MetaNetwork
import numpy as np

class MetaPriorRegressionLearner(object):
    def __init__(self, dataset, arch, meta_batch_size, meta_step_size,
                 inner_step_size, lr_decay_itr,
                 epoch, num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, num_support, data_attack_type, tensorboard_data_prefix):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_step_size = inner_step_size
        self.lr_decay_itr = lr_decay_itr
        self.epoch = epoch
        self.num_inner_updates = num_inner_updates
        self.test_finetune_updates = num_inner_updates
        if arch == "AE":
            backbone = AutoEncoder(IN_CHANNELS[self.dataset])
        elif arch == "UNet":
            backbone = ResNetUNet(self.dataset, IN_CHANNELS[self.dataset], CLASS_NUM[self.dataset])
        elif arch == "FCN":
            backbone = FCN8s(n_class=IN_CHANNELS[self.dataset])
        self.network = MetaNetwork(backbone)
        self.network.cuda()
        self.num_support = num_support
        trn_dataset = TwoQueriesMetaTaskDataset(data_attack_type, tot_num_tasks, dataset, load_mode=load_task_mode, protocol=protocol)
        # task number per mini-batch is controlled by DataLoader
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.tensorboard = TensorBoardWriter("{0}/tensorboard/prior_regression/".format(PY_ROOT),
                                             tensorboard_data_prefix)
        os.makedirs("{0}/tensorboard/prior_regression/".format(PY_ROOT), exist_ok=True)
        self.fast_net = InnerLoop(self.network, self.num_inner_updates,
                                  self.inner_step_size, self.meta_batch_size)
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)
        self.mse_loss = nn.MSELoss().cuda()

    def forward_pass(self, network, input, target):
        output = network.net_forward(input)
        loss = self.mse_loss(output, target)
        return loss, output

    def meta_update(self, grads, query_images, query_grads):
        in_, target = query_images[0], query_grads[0]  # T,C,H,W
        # We use a dummy forward / backward pass to get the correct grads into self.net
        loss, output = self.forward_pass(self.network, in_, target)  # 其实传谁无所谓，因为loss.backward调用的时候，会用外部更新的梯度的求和来替换掉loss.backward自己算出来的梯度值
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
        self.opt.zero_grad()  # 清空梯度
        loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()


    def evaluate_query_set(self, model, query_adv_images, query_target, fast_weights):  # B,C,H,W;  query_target = B,#class_num
        query_output = model.net_forward(query_adv_images, fast_weights)
        mse_error = F.mse_loss(query_output, query_target)
        return mse_error

    def train(self, model_path, resume_epoch=0):
        for epoch in range(resume_epoch, self.epoch):
            for i, (_, _, _, _, _, adv_images, grad_images) in enumerate(self.train_loader):
                # adv_images shape = (Task_num, T,C,H,W)  grad_images shape = (Task_num, T, C, H, W)
                # adv_images 去掉最后一个，priors去掉第一个，然后互相匹配
                adv_images = adv_images[:, :adv_images.size(1)-1, :, :, :]  # Task_num, T-1, C, H, W
                grad_images = grad_images[:, 1:, :, :, :]  # Task_num, T-1, C,H, W
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                adv_images, grad_images = adv_images.cuda(), grad_images.cuda()  # (Task_num, T, C, H, W)
                seq_len = adv_images.size(1)
                support_index_list = sorted(random.sample(range(seq_len//2), self.num_support))
                query_index_list = np.arange(seq_len//2, seq_len).tolist()
                support_adv_images, support_grad_images = adv_images[:,support_index_list,:,:,:], grad_images[:,support_index_list,:,:,:]
                query_adv_images, query_grad_images = adv_images[:,query_index_list,:,:,:], grad_images[:,query_index_list,:,:,:]
                all_tasks_mse_error = []
                for task_idx in range(adv_images.size(0)):
                    task_support_adv_images = support_adv_images[task_idx]  # T, C, H, W
                    task_support_grad_images = support_grad_images[task_idx]  # T, C, H, W
                    task_query_adv_images = query_adv_images[task_idx]
                    task_query_grad_images = query_grad_images[task_idx]
                    self.fast_net.copy_weights(self.network)
                    # fast_net only forward one task's data
                    g, fast_weights = self.fast_net.forward(task_support_adv_images, task_query_adv_images,
                                              task_support_grad_images, task_query_grad_images)
                    grads.append(g)
                    mse_error = self.evaluate_query_set(self.fast_net.network, task_query_adv_images,
                                                        task_query_grad_images, fast_weights)
                    fast_weights.clear()
                    all_tasks_mse_error.append(mse_error.detach().cpu())

                # Perform the meta update
                self.meta_update(grads, query_adv_images, query_grad_images)
                grads.clear()
                if itr % 100 == 0 and itr > 0:
                    self.tensorboard.record_trn_query_loss(torch.stack(all_tasks_mse_error).mean().detach().cpu(), itr)
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
