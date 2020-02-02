import random
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")

from meta_grad_regression_auto_encoder.network.autoencoder import AutoEncoder
from meta_simulator_model.network.fcn_8s import FCN8s
from meta_simulator_model.network.unet import ResNetUNet
import os
from config import PY_ROOT, IN_CHANNELS, CLASS_NUM
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.meta_img_grad_dataset import MetaImgOnlineGradTaskDataset
from cifar_models_myself import *
from meta_simulator_model.inner_loop import InnerLoop
from meta_simulator_model.tensorboard_helper import TensorBoardWriter
from meta_simulator_model.meta_network import MetaNetwork
import numpy as np

class MetaGradRegressionLearner(object):
    def __init__(self, dataset, arch, meta_batch_size, meta_step_size,
                 inner_step_size, lr_decay_itr,
                 epoch, num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, num_support, data_loss_type, tensorboard_data_prefix):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_step_size = inner_step_size
        self.lr_decay_itr = lr_decay_itr
        self.epoch = epoch
        self.num_inner_updates = num_inner_updates
        self.test_finetune_updates = num_inner_updates
        assert arch in ["AE","UNet", "FCN"]
        if arch == "AE":
            backbone = AutoEncoder(IN_CHANNELS[self.dataset])
        elif arch == "UNet":
            backbone = ResNetUNet(self.dataset, IN_CHANNELS[self.dataset], CLASS_NUM[self.dataset])
        elif arch == "FCN":
            backbone = FCN8s(n_class=IN_CHANNELS[self.dataset])
        self.network = MetaNetwork(backbone)
        self.network.cuda()
        self.num_support = num_support
        trn_dataset = MetaImgOnlineGradTaskDataset(data_loss_type, tot_num_tasks, dataset, load_mode=load_task_mode, protocol=protocol)
        # task number per mini-batch is controlled by DataLoader
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.tensorboard = TensorBoardWriter("{0}/tensorboard/grad_regression/".format(PY_ROOT),
                                             tensorboard_data_prefix)
        os.makedirs("{0}/tensorboard/grad_regression/".format(PY_ROOT), exist_ok=True)
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
            for i, (adv_images, grad_images, _) in enumerate(self.train_loader):
                # adv_images shape = (Task_num, T,C,H,W)  grad_images shape = (Task_num, T, C, H, W)
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                grads = []
                adv_images, grad_images = adv_images.cuda(), grad_images.cuda()  # (Task_num, T, C, H, W)
                seq_len = adv_images.size(1)
                # 无ConvLSTM版本,随机挑选num_support作为support set，从该序列中剩余元素随机挑选num_query作为query set
                # 有ConvLSTM版本,support set每帧都输入，每隔T帧拿一个grad_images作为监督信号。query set：同样sequence每帧都输入，每帧都监督！
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
                    with torch.no_grad():
                        mse_error = self.evaluate_query_set(self.fast_net.network, task_query_adv_images, task_query_grad_images,
                                                                      fast_weights)
                        all_tasks_mse_error.append(mse_error.detach().cpu())
                    fast_weights.clear()

                # Perform the meta update
                self.meta_update(grads, query_adv_images, query_grad_images)
                grads.clear()
                if itr % 100 == 0 and itr > 0:
                    self.tensorboard.record_trn_query_loss(torch.stack(all_tasks_mse_error).mean().detach().cpu(), itr)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.network.state_dict(),
                'optimizer': self.opt.state_dict(),
                "query_mse_error": torch.stack(all_tasks_mse_error).mean().detach().cpu(),
            }, model_path)


    def adjust_learning_rate(self,itr, meta_lr, lr_decay_itr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if lr_decay_itr > 0:
            if int(itr % lr_decay_itr) == 0 and itr > 0:
                meta_lr = meta_lr / (10 ** int(itr / lr_decay_itr))
                self.fast_net.step_size = self.fast_net.step_size / 10
                for param_group in self.opt.param_groups:
                    param_group['lr'] = meta_lr
