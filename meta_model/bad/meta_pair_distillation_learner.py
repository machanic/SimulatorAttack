import glob
import random
import sys

from constant_enum import PRIOR_MODE

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import os
from meta_model.inner_loop_pair_loss import InnerLoopPairLoss
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

class MetaPairDistillationLearner(object):
    def __init__(self, dataset, arch, meta_batch_size, meta_step_size,
                 inner_step_size, lr_decay_itr,
                 epoch, num_inner_updates, load_task_mode, protocol,
                 tot_num_tasks, num_support, sequence_num, data_loss_type, prior_mode, tensorboard_data_prefix):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_step_size = inner_step_size
        self.lr_decay_itr = lr_decay_itr
        self.epoch = epoch
        self.num_inner_updates = num_inner_updates
        self.num_classes = CLASS_NUM[self.dataset]
        backbone = self.construct_model(arch, dataset)
        self.network = MetaNetwork(backbone)
        self.network.cuda()
        model_load_path = "{}/train_pytorch_model/real_image_model/{}@{}@epoch_*@lr_*@batch_*.pth.tar".format(
            PY_ROOT, self.dataset, arch)
        model_load_path = glob.glob(model_load_path)[0]
        assert os.path.exists(model_load_path), model_load_path

        self.network.network.load_state_dict(torch.load(model_load_path,
                                                        map_location=lambda storage, location: storage)["state_dict"])

        self.num_support = num_support
        self.sequence_num = sequence_num   # 默认为1，每个task一个sequence
        trn_dataset = MetaTaskDataset(data_loss_type, tot_num_tasks, sequence_num, dataset, load_mode=load_task_mode,
                                      protocol=protocol)
        # task number per mini-batch is controlled by DataLoader
        self.train_loader = DataLoader(trn_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.tensorboard = TensorBoardWriter("{0}/tensorboard/pair_loss_distillation".format(PY_ROOT),
                                              tensorboard_data_prefix)
        os.makedirs("{0}/tensorboard/pair_loss_distillation".format(PY_ROOT), exist_ok=True)
        self.fast_net = InnerLoopPairLoss(self.network, self.num_inner_updates,
                                  self.inner_step_size, self.meta_batch_size)  # 并行执行每个task
        self.fast_net.cuda()
        self.opt = Adam(self.network.parameters(), lr=meta_step_size)
        self.arch_pool = {}
        self.exploration = 0.1
        self.fd_eta = 0.1
        self.online_lr = 100
        self.prior_mode = prior_mode

    def forward_pass(self, network, input, target):
        output = F.log_softmax(network.net_forward(input), dim=1)
        loss = F.mse_loss(output, F.log_softmax(target, dim=1))
        return loss

    def construct_model(self, arch, dataset):
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
        return network

    def get_arch(self, arch):
        if arch in self.arch_pool:
            model = self.arch_pool[arch]
            # assert (not next(model.parameters()).is_cuda)
            return model
        model = self.construct_model(arch, self.dataset)
        model_load_path = "{}/train_pytorch_model/real_image_model/{}@{}@epoch_*@lr_*@batch_*.pth.tar".format(
            PY_ROOT, self.dataset, arch)
        model_load_path = glob.glob(model_load_path)[0]
        assert os.path.exists(model_load_path), model_load_path
        model.load_state_dict(torch.load(model_load_path, map_location=lambda storage, location: storage)["state_dict"])
        model.eval()
        self.arch_pool[arch] = model
        return model

    def meta_update(self, grads, query_images, query_targets):
        # We use a dummy forward / backward pass to get the correct grads into self.net
        loss = self.forward_pass(self.network, query_images, query_targets)  # 其实传谁无所谓，因为loss.backward调用的时候，会用外部更新的梯度的求和来替换掉loss.backward自己算出来的梯度值
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
            hooks.append(v.register_hook(get_closure()))  # 这个register_hook在backward时替换梯度
        # Compute grads for current step, replace with summed gradients as defined by hook
        self.opt.zero_grad()
        loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()

    def norm(self, t: torch.Tensor):
        old_B1, old_B2, old_B3, C,H,W = t.size(0), t.size(1), t.size(2), t.size(-3), t.size(-2), t.size(-1)
        t = t.view(-1, C, H, W)
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(old_B1, old_B2, old_B3, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def construct_prior(self, images, true_labels, teacher_model):
        def norm(t):
            assert len(t.shape) == 4
            norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
            norm_vec += (norm_vec == 0).float() * 1e-8
            return norm_vec

        def eg_step(x, g, lr):
            real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
            pos = real_x * torch.exp(lr * g)
            neg = (1 - real_x) * torch.exp(-lr * g)
            new_x = pos / (pos + neg)
            return new_x * 2 - 1

        task_num, seq_num, T, C, H, W = images.size()
        batch_size = task_num * seq_num
        prior = torch.zeros(batch_size, C, H, W).cuda()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        all_time_step_priors = [prior]
        for seq_idx in range(seq_num):
            current_images = images[:,:,seq_idx,:,:,:]
            current_images = current_images.view(batch_size, C, H, W)
            exp_noise = self.exploration * torch.randn_like(prior) / ((C*H*W) ** 0.5)
            exp_noise = exp_noise.cuda()
            q1 = prior + exp_noise
            q2 = prior - exp_noise
            images1 = current_images + self.fd_eta * q1 / norm(q1)
            images2 = current_images + self.fd_eta * q2 / norm(q2)
            with torch.no_grad():
                all_images = torch.cat((images1, images2), dim=0)
                output = teacher_model(all_images)
                output1, output2 = torch.split(output, (images1.size(0), images2.size(0)),dim=0)
                loss1 = criterion(output1, true_labels)
                loss2 = criterion(output2, true_labels)
            est_deriv = (loss1 - loss2) / (self.fd_eta * self.exploration)
            est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise
            prior = eg_step(prior, est_grad, self.online_lr)
            all_time_step_priors.append(prior)

    def train_simulate_grad_mode(self, model_path, resume_epoch=0):
        for epoch in range(resume_epoch, self.epoch):
            for i, (adv_images, _, archs) in enumerate(self.train_loader):
                # adv_images shape = (Task_num, Seq_num, T,C,H,W)  grad_images shape = (Task_num, Seq_num, T, C, H, W)
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                adv_images = adv_images.cuda()
                seq_len = adv_images.size(2)
                support_index_list = sorted(random.sample(range(seq_len // 2), self.num_support))
                query_index_list = np.arange(seq_len // 2, seq_len).tolist()
                support_adv_images = adv_images[:,:,support_index_list,:,:,:]
                query_adv_images = adv_images[:,:,query_index_list,:,:,:]
                prior = torch.zeros_like(adv_images).cuda()
                exp_noise = self.exploration * torch.randn_like(prior) / (
                        (prior.size(-3) * prior.size(-2) * prior.size(-1)) ** 0.5)
                exp_noise = exp_noise.cuda()
                q1 = prior + exp_noise  # shape = (Task_num, Seq_num, T, C, H, W)
                q2 = prior - exp_noise
                q1_support = q1[:,:, support_index_list, :, :, :]
                q1_query = q1[:,:, query_index_list, :, :, :]
                q2_support = q2[:, :, support_index_list, :, :, :]
                q2_query = q2[:,:, query_index_list, :, :, :]
                support_adv_images_1 = support_adv_images + self.fd_eta * q1_support / self.norm(q1_support)
                support_adv_images_2 = support_adv_images + self.fd_eta * q2_support / self.norm(q2_support)
                query_adv_images_1 = query_adv_images + self.fd_eta * q1_query / self.norm(q1_query)
                query_adv_images_2 = query_adv_images + self.fd_eta * q2_query / self.norm(q2_query)
                grads = []
                huber_losses = []
                mse_losses = []
                for task_idx in range(adv_images.size(0)):
                    arch_name = ALL_MODELS[archs[task_idx].item()]
                    teacher_model = self.get_arch(arch_name)
                    teacher_model.cuda()
                    task_support_images_1 = support_adv_images_1[task_idx]  # Seq_num, T, C, H, W
                    _,_,C,H,W = task_support_images_1.size()
                    task_support_images_1 = task_support_images_1.view(-1, C, H, W) # B, C, H, W
                    task_support_images_2 = support_adv_images_2[task_idx]
                    task_support_images_2 = task_support_images_2.view(-1, C, H, W) # B, C, H, W
                    task_query_images_1 = query_adv_images_1[task_idx]
                    task_query_images_1 = task_query_images_1.view(-1, C, H, W)
                    task_query_images_2 = query_adv_images_2[task_idx]
                    task_query_images_2 = task_query_images_2.view(-1, C, H, W)

                    B1,B2,B3,B4 = task_support_images_1.size(0), task_support_images_2.size(0), \
                                  task_query_images_1.size(0), task_query_images_2.size(0)
                    teacher_model_input = torch.cat((task_support_images_1, task_support_images_2, task_query_images_1, task_query_images_2), dim=0).detach()
                    # teacher_model_input_query = torch.cat((task_query_images_1, task_query_images_2), dim=0).detach().cuda(1)
                    with torch.no_grad():
                        teacher_output = teacher_model(teacher_model_input)

                    teacher_output = teacher_output.detach().cuda(0)
                    task_support_target_1, task_support_target_2, task_query_target_1, task_query_target_2 = \
                        torch.split(teacher_output, (B1, B2,B3,B4), dim=0)
                    # teacher_model.cpu()
                    # torch.cuda.empty_cache()
                    self.fast_net.copy_weights(self.network)
                    g, huber_loss, query_mse_loss1, query_mse_loss2 = self.fast_net.forward(task_support_images_1, task_support_images_2,
                                                                                            task_query_images_1, task_query_images_2,
                                              task_support_target_1, task_support_target_2, task_query_target_1, task_query_target_2)
                    huber_losses.append(huber_loss.detach().cpu())
                    mse_losses.append(query_mse_loss1.detach().cpu())
                    mse_losses.append(query_mse_loss2.detach().cpu())
                    grads.append(g)

                dummy_query_images = query_adv_images[0].view(-1, query_adv_images.size(-3), query_adv_images.size(-2),
                                                              query_adv_images.size(-1))
                arch_name = ALL_MODELS[archs[random.randint(0,len(archs)-1)]]
                teacher_model = self.get_arch(arch_name)
                teacher_model.cuda()
                with torch.no_grad():
                    dummy_logits = teacher_model(dummy_query_images).detach()
                # teacher_model.cpu()
                self.meta_update(grads, dummy_query_images, dummy_logits)
                grads.clear()
                mse_losses = torch.stack(mse_losses).mean()
                huber_losses = torch.stack(huber_losses).mean()
                if itr % 100 == 0:
                    query_output = self.network(dummy_query_images)
                    query_predict = query_output.max(1)[1]
                    teacher_target = dummy_logits.max(1)[1]
                    teacher_target = teacher_target.long().cuda()
                    accuracy = (teacher_target.eq(query_predict.long())).sum() / teacher_target.size(0)
                    self.tensorboard.record_trn_query_distance_loss(huber_losses, itr)
                    self.tensorboard.record_trn_query_output_logits_loss(mse_losses, itr)
                    self.tensorboard.record_trn_query_acc(accuracy, itr)

            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.network.state_dict(),
                'optimizer': self.opt.state_dict(),
            }, model_path)


    def train(self, model_path, resume_epoch=0):
        if self.prior_mode == PRIOR_MODE.PGD_GRAD:
            self.train_pgd_grad_mode(model_path, resume_epoch)
        elif self.prior_mode == PRIOR_MODE.SIMULATE_GRAD:
            self.train_simulate_grad_mode(model_path, resume_epoch)

    def train_pgd_grad_mode(self, model_path, resume_epoch=0):
        for epoch in range(resume_epoch, self.epoch):
            for i, (adv_images, grad_images, archs) in enumerate(self.train_loader):
                # adv_images shape = (Task_num, Seq_num, T,C,H,W)  grad_images shape = (Task_num, Seq_num, T, C, H, W)
                itr = epoch * len(self.train_loader) + i
                self.adjust_learning_rate(itr, self.meta_step_size, self.lr_decay_itr)
                adv_images = adv_images.cuda()
                grad_images = grad_images.cuda()
                seq_len = adv_images.size(2)
                support_index_list = sorted(random.sample(range(seq_len // 2), self.num_support))
                query_index_list = np.arange(seq_len // 2, seq_len).tolist()
                support_adv_images = adv_images[:,:,support_index_list,:,:,:]  # 前一半随机采用num_support个
                query_adv_images = adv_images[:,:,query_index_list,:,:,:]  # (Task_num, Seq_num, T, C, H, W)
                exp_noise = self.exploration * torch.randn_like(grad_images) / (
                        (grad_images.size(-3) * grad_images.size(-2) * grad_images.size(-1)) ** 0.5)
                exp_noise = exp_noise.cuda()
                q1 = grad_images + exp_noise  # shape = (Task_num, Seq_num, T, C, H, W)
                q2 = grad_images - exp_noise
                q1_support = q1[:,:, support_index_list, :, :, :]
                q1_query = q1[:,:, query_index_list, :, :, :]
                q2_support = q2[:, :, support_index_list, :, :, :]
                q2_query = q2[:,:, query_index_list, :, :, :]
                support_adv_images_1 = support_adv_images + self.fd_eta * q1_support / self.norm(q1_support)
                support_adv_images_2 = support_adv_images + self.fd_eta * q2_support / self.norm(q2_support)
                query_adv_images_1 = query_adv_images + self.fd_eta * q1_query / self.norm(q1_query)
                query_adv_images_2 = query_adv_images + self.fd_eta * q2_query / self.norm(q2_query)
                grads = []
                huber_losses = []
                mse_losses = []
                for task_idx in range(adv_images.size(0)):
                    arch_name = ALL_MODELS[archs[task_idx].item()]
                    teacher_model = self.get_arch(arch_name)
                    teacher_model.cuda()
                    task_support_images_1 = support_adv_images_1[task_idx]  # Seq_num, T, C, H, W
                    _,_,C,H,W = task_support_images_1.size()
                    task_support_images_1 = task_support_images_1.view(-1, C, H, W) # B, C, H, W
                    task_support_images_2 = support_adv_images_2[task_idx]
                    task_support_images_2 = task_support_images_2.view(-1, C, H, W) # B, C, H, W
                    task_query_images_1 = query_adv_images_1[task_idx]
                    task_query_images_1 = task_query_images_1.view(-1, C, H, W)
                    task_query_images_2 = query_adv_images_2[task_idx]
                    task_query_images_2 = task_query_images_2.view(-1, C, H, W)

                    B1,B2,B3,B4 = task_support_images_1.size(0), task_support_images_2.size(0), \
                                  task_query_images_1.size(0), task_query_images_2.size(0)
                    teacher_model_input = torch.cat((task_support_images_1, task_support_images_2, task_query_images_1, task_query_images_2), dim=0).detach()
                    # teacher_model_input_query = torch.cat((task_query_images_1, task_query_images_2), dim=0).detach().cuda(1)
                    with torch.no_grad():
                        teacher_output = teacher_model(teacher_model_input)

                    teacher_output = teacher_output.detach().cuda(0)
                    task_support_target_1, task_support_target_2, task_query_target_1, task_query_target_2 = \
                        torch.split(teacher_output, (B1, B2,B3,B4), dim=0)
                    # teacher_model.cpu()
                    # torch.cuda.empty_cache()
                    self.fast_net.copy_weights(self.network)
                    g, huber_loss, query_mse_loss1, query_mse_loss2 = self.fast_net.forward(task_support_images_1, task_support_images_2,
                                                                                            task_query_images_1, task_query_images_2,
                                              task_support_target_1, task_support_target_2, task_query_target_1, task_query_target_2)
                    huber_losses.append(huber_loss.detach().cpu())
                    mse_losses.append(query_mse_loss1.detach().cpu())
                    mse_losses.append(query_mse_loss2.detach().cpu())
                    grads.append(g)

                dummy_query_images = query_adv_images[0].view(-1, query_adv_images.size(-3), query_adv_images.size(-2),
                                                              query_adv_images.size(-1))
                arch_name = ALL_MODELS[archs[random.randint(0,len(archs)-1)]]
                teacher_model = self.get_arch(arch_name)
                teacher_model.cuda()
                with torch.no_grad():
                    dummy_logits = teacher_model(dummy_query_images).detach()
                # teacher_model.cpu()
                self.meta_update(grads, dummy_query_images, dummy_logits)
                grads.clear()
                mse_losses = torch.stack(mse_losses).mean()
                huber_losses = torch.stack(huber_losses).mean()
                if itr % 100 == 0:
                    query_output = self.network(dummy_query_images)
                    query_predict = query_output.max(1)[1]
                    teacher_target = dummy_logits.max(1)[1]
                    teacher_target = teacher_target.long().cuda()
                    accuracy = (teacher_target.eq(query_predict.long())).sum() / teacher_target.size(0)
                    self.tensorboard.record_trn_query_distance_loss(huber_losses, itr)
                    self.tensorboard.record_trn_query_output_logits_loss(mse_losses, itr)
                    self.tensorboard.record_trn_query_acc(accuracy, itr)

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
