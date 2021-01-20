import glob
import os
import re

import glog as log
import torch
from torch import nn
from torch.optim import Adam

from config import PY_ROOT
from constant_enum import SPLIT_DATA_PROTOCOL
from meta_simulator_bandits.learning.meta_network import MetaNetwork
from dataset.standard_model import MetaLearnerModelBuilder


class MetaModelFinetune(object):
    def __init__(self, dataset,
                 simulator_type,
                 batch_size, meta_train_type, distill_loss, data_loss, norm, targeted, use_softmax, without_resnet):
        target_str = "targeted_attack_random" if targeted else "untargeted_attack"
        # 2Q_DISTILLATION@CIFAR-100@TRAIN_I_TEST_II@model_resnet34@loss_pair_mse@dataloss_cw_l2_untargeted_attack@epoch_4@meta_batch_size_30@num_support_50@num_updates_12@lr_0.001@inner_lr_0.01.pth.tar
        self.meta_model_path = "{root}/train_pytorch_model/meta_simulator/{meta_train_type}@{dataset}@{split}@model_{meta_arch}@loss_{loss}@dataloss_{data_loss}_{norm}_{target_str}*".format(
            root=PY_ROOT, meta_train_type=meta_train_type.upper(), dataset=dataset,
            split=SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II,
            meta_arch="resnet34", loss=distill_loss, data_loss=data_loss, norm=norm, target_str=target_str)
        if without_resnet:
            self.meta_model_path = "{root}/train_pytorch_model/meta_simulator/{meta_train_type}@{dataset}@{split}@model_{meta_arch}@loss_{loss}@dataloss_{data_loss}_{norm}_{target_str}*@without_resnet.pth.tar".format(
                root=PY_ROOT, meta_train_type=meta_train_type.upper(), dataset=dataset,
                split=SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II,
                meta_arch="resnet34", loss=distill_loss, data_loss=data_loss, norm=norm, target_str=target_str)
        self.meta_model_path = glob.glob(self.meta_model_path)
        pattern = re.compile(".*model_(.*?)@.*inner_lr_(\d\.*?).*\.pth.*")
        assert len(self.meta_model_path) > 0
        self.meta_model_path = self.meta_model_path[0]
        log.info("load meta model {}".format(self.meta_model_path))
        ma = pattern.match(os.path.basename(self.meta_model_path))
        arch = ma.group(1)
        self.inner_lr = float(ma.group(2))
        self.arch = arch
        self.dataset = dataset
        self.need_pair_distance = (distill_loss.lower()=="pair_mse")
        # self.need_pair_distance = False
        self.softmax = nn.Softmax(dim=1)
        self.mse_loss = nn.MSELoss()
        self.pair_wise_distance = nn.PairwiseDistance(p=2)
        self.use_softmax = use_softmax
        if simulator_type=="meta_simulator":
            self.pretrained_weights = torch.load(self.meta_model_path, map_location=lambda storage, location: storage)[
                "state_dict"]
            meta_backbone = self.construct_model(arch, dataset)
            self.meta_network = MetaNetwork(meta_backbone)
            self.meta_network.load_state_dict(self.pretrained_weights)
            self.meta_network.eval()
            self.meta_network.cuda()
        elif simulator_type == "vanilla_ensemble":
            self.meta_model_path = "{root}/train_pytorch_model/vanilla_simulator/{dataset}@{norm}_norm_{target_str}@{meta_arch}*.tar".format(
                root=PY_ROOT, dataset=dataset, meta_arch="resnet34", norm=norm, target_str=target_str)
            self.meta_model_path = glob.glob(self.meta_model_path)
            assert len(self.meta_model_path) > 0
            self.meta_model_path = self.meta_model_path[0]
            self.inner_lr = 0.01
            self.meta_network = self.construct_model("resnet34", dataset)
            self.pretrained_weights = torch.load(self.meta_model_path, map_location=lambda storage, location: storage)
            log.info("Load model from {} in epoch {}.".format(self.meta_model_path, self.pretrained_weights["epoch"]))
            self.pretrained_weights = self.pretrained_weights["state_dict"]

        self.batch_weights = {}
        self.batch_size = batch_size
        for i in range(batch_size):
            self.batch_weights[i] = self.pretrained_weights

    def construct_model(self, arch, dataset):
        if dataset in ["CIFAR-10", "CIFAR-100", "MNIST", "FashionMNIST"]:
            network = MetaLearnerModelBuilder.construct_cifar_model(arch, dataset)
        elif dataset == "TinyImageNet":
            network = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
        elif dataset == "ImageNet":
            network = MetaLearnerModelBuilder.construct_imagenet_model(arch, dataset)
        return network

    def finetune(self, q1_images, q2_images, q1_gt_logits, q2_gt_logits, finetune_times, is_first_finetune):
        '''
        :param q1_images: shape of (B,T,C,H,W) where T is sequence length
        :param q2_images: shape of (B,T,C,H,W)
        :param q1_gt_logits: shape of (B, T, #class)
        :param q2_gt_logits: shape of (B, T, #class)
        :return:
        '''
        log.info("begin finetune images")
        if is_first_finetune:
            for i in range(self.batch_size):
                self.batch_weights[i] = self.pretrained_weights
            self.meta_network.load_state_dict(self.pretrained_weights)
        for img_idx, (q1_images_tensor,q2_images_tensor, each_q1_gt_logits,each_q2_gt_logits) in enumerate(zip(q1_images,
                                                                                q2_images,q1_gt_logits,q2_gt_logits)):
            self.meta_network.load_state_dict(self.batch_weights[img_idx])
            # meta_network.copy_weights(self.master_network) # delete this line, only fine-tune 1 time for later iterations
            # self.meta_network.train()
            optimizer = Adam(self.meta_network.parameters(), lr=self.inner_lr)
            for _ in range(finetune_times):
                q1_output = self.meta_network.forward(q1_images_tensor)
                q2_output = self.meta_network.forward(q2_images_tensor)
                if self.use_softmax:
                    q1_output, q2_output = self.softmax(q1_output), self.softmax(q2_output)
                    each_q1_gt_logits, each_q2_gt_logits = self.softmax(each_q1_gt_logits), self.softmax(each_q2_gt_logits)
                mse_error_q1 = self.mse_loss(q1_output, each_q1_gt_logits)
                mse_error_q2 = self.mse_loss(q2_output, each_q2_gt_logits)
                if self.need_pair_distance:
                    predict_distance = self.pair_wise_distance(q1_output, q2_output)
                    target_distance = self.pair_wise_distance(each_q1_gt_logits, each_q2_gt_logits)
                    distance_loss = self.mse_loss(predict_distance, target_distance)
                    tot_loss = distance_loss + 0.1 * mse_error_q1 + 0.1 * mse_error_q2
                else:
                    tot_loss =  mse_error_q1 + mse_error_q2
                optimizer.zero_grad()
                tot_loss.backward()
                optimizer.step()
            self.meta_network.eval()
            self.batch_weights[img_idx] = self.meta_network.state_dict().copy()
        log.info("finetune images done")

    def predict(self, q1_images, q2_images):
        '''
        :param q1_images: shape of (B,C,H,W)
        :param q2_images: shape of (B,C,H,W)
        :return:
        '''
        log.info("predict from meta model")
        q1_output = []
        q2_output = []
        for img_idx, (q1_img, q2_img) in enumerate(zip(q1_images, q2_images)):
            self.meta_network.load_state_dict(self.batch_weights[img_idx])
            self.meta_network.eval()
            q1_img = torch.unsqueeze(q1_img, 0)
            q2_img=  torch.unsqueeze(q2_img, 0)
            q1_logits = self.meta_network.forward(q1_img)
            q2_logits = self.meta_network.forward(q2_img)
            q1_output.append(q1_logits)
            q2_output.append(q2_logits)
        q1_output = torch.cat(q1_output, 0)
        q2_output = torch.cat(q2_output, 0)
        return q1_output, q2_output



class MemoryEfficientMetaModelFinetune(object):
    def __init__(self, dataset,  batch_size, meta_arch, meta_train_type, distill_loss, data_loss, norm, targeted, simulator_type,
                 use_softmax, without_resnet):
        target_str = "targeted_attack_random" if targeted else "untargeted_attack"
        # 2Q_DISTILLATION@CIFAR-100@TRAIN_I_TEST_II@model_resnet34@loss_pair_mse@dataloss_cw_l2_untargeted_attack@epoch_4@meta_batch_size_30@num_support_50@num_updates_12@lr_0.001@inner_lr_0.01.pth.tar
        self.meta_model_path = "{root}/train_pytorch_model/meta_simulator/{meta_train_type}@{dataset}@{split}@model_{meta_arch}@loss_{loss}@dataloss_{data_loss}_{norm}_{target_str}*".format(
            root=PY_ROOT, meta_train_type=meta_train_type.upper(), dataset=dataset, split=SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II,
            meta_arch=meta_arch, loss=distill_loss, data_loss=data_loss, norm=norm, target_str=target_str)
        if without_resnet:
            self.meta_model_path = "{root}/train_pytorch_model/meta_simulator/{meta_train_type}@{dataset}@{split}@model_{meta_arch}@loss_{loss}@dataloss_{data_loss}_{norm}_{target_str}*@without_resnet.pth.tar".format(
                root=PY_ROOT, meta_train_type=meta_train_type.upper(), dataset=dataset,
                split=SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II,
                meta_arch="resnet34", loss=distill_loss, data_loss=data_loss, norm=norm, target_str=target_str)
        log.info("start using {}".format(self.meta_model_path))
        self.meta_model_path = glob.glob(self.meta_model_path)
        pattern = re.compile(".*model_(.*?)@.*inner_lr_(\d\.*?).*\.pth.*")
        assert len(self.meta_model_path) > 0
        self.meta_model_path = self.meta_model_path[0]

        ma = pattern.match(os.path.basename(self.meta_model_path))
        arch = ma.group(1)
        self.inner_lr = float(ma.group(2))
        self.arch = arch
        self.dataset = dataset
        self.need_pair_distance = (distill_loss.lower()=="pair_mse")
        # self.need_pair_distance = False
        self.softmax = nn.Softmax(dim=1)
        self.mse_loss = nn.MSELoss()
        self.pair_wise_distance = nn.PairwiseDistance(p=2)
        self.use_softmax = use_softmax
        if simulator_type == "vanilla_ensemble":
            target_str = "targeted" if targeted else "untargeted"
            self.meta_model_path = "{root}/train_pytorch_model/vanilla_simulator/{dataset}@{norm}_norm_{target_str}@{meta_arch}*.tar".format(
                root=PY_ROOT, dataset=dataset, meta_arch="resnet34", norm=norm, target_str=target_str)
            print(self.meta_model_path)
            self.meta_model_path = glob.glob(self.meta_model_path)

            assert len(self.meta_model_path) > 0
            self.meta_model_path = self.meta_model_path[0]
            self.inner_lr = 0.01
            self.meta_network = self.construct_model(arch, dataset)
            self.pretrained_weights = torch.load(self.meta_model_path, map_location=lambda storage, location: storage)
            log.info("Load model from {} in epoch {}.".format(self.meta_model_path, self.pretrained_weights["epoch"]))
            self.pretrained_weights = self.pretrained_weights["state_dict"]
        elif simulator_type == "meta_simulator":
            meta_backbone = self.construct_model(arch, dataset)
            self.meta_network = MetaNetwork(meta_backbone)
            loaded = torch.load(self.meta_model_path, map_location=lambda storage, location: storage)
            self.pretrained_weights = loaded["state_dict"]
            log.info("load meta model {} epoch({})".format(self.meta_model_path, loaded["epoch"]))
        self.meta_network.load_state_dict(self.pretrained_weights)
        self.meta_network.eval()
        self.meta_network.cuda()
        self.batch_weights = {}
        self.batch_size = batch_size
        for i in range(batch_size):
            self.batch_weights[i] = self.pretrained_weights

    def construct_model(self, arch, dataset):
        if dataset in ["CIFAR-10", "CIFAR-100", "MNIST", "FashionMNIST"]:
            network = MetaLearnerModelBuilder.construct_cifar_model(arch, dataset)
        elif dataset == "TinyImageNet":
            network = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
        elif dataset == "ImageNet":
            network = MetaLearnerModelBuilder.construct_imagenet_model(arch, dataset)
        return network

    def finetune(self, q1_images, q2_images, q1_gt_logits, q2_gt_logits, finetune_times, is_first_finetune, img_idx_to_batch_idx):
        '''
        :param q1_images: shape of (B,T,C,H,W) where T is sequence length
        :param q2_images: shape of (B,T,C,H,W)
        :param q1_gt_logits: shape of (B, T, #class)
        :param q2_gt_logits: shape of (B, T, #class)
        :return:
        '''
        log.info("begin finetune images")
        if is_first_finetune:
            for i in range(self.batch_size):
                self.batch_weights[i] = self.pretrained_weights
            self.meta_network.load_state_dict(self.pretrained_weights)
        for img_idx, (q1_images_tensor,q2_images_tensor, each_q1_gt_logits,each_q2_gt_logits) in enumerate(zip(q1_images,
                                                                                q2_images,q1_gt_logits,q2_gt_logits)):
            self.meta_network.load_state_dict(self.batch_weights[img_idx_to_batch_idx[img_idx]])
            # meta_network.copy_weights(self.master_network) # delete this line, only fine-tune 1 time for later iterations
            # self.meta_network.train()
            optimizer = Adam(self.meta_network.parameters(), lr=self.inner_lr)
            for _ in range(finetune_times):
                q1_output = self.meta_network.forward(q1_images_tensor)
                q2_output = self.meta_network.forward(q2_images_tensor)
                if self.use_softmax:
                    q1_output, q2_output = self.softmax(q1_output), self.softmax(q2_output)
                    each_q1_gt_logits, each_q2_gt_logits = self.softmax(each_q1_gt_logits), self.softmax(each_q2_gt_logits)
                mse_error_q1 = self.mse_loss(q1_output, each_q1_gt_logits)
                mse_error_q2 = self.mse_loss(q2_output, each_q2_gt_logits)
                if self.need_pair_distance:
                    predict_distance = self.pair_wise_distance(q1_output, q2_output)
                    target_distance = self.pair_wise_distance(each_q1_gt_logits, each_q2_gt_logits)
                    distance_loss = self.mse_loss(predict_distance, target_distance)
                    tot_loss = distance_loss + 0.1 * mse_error_q1 + 0.1 * mse_error_q2
                else:
                    tot_loss =  mse_error_q1 + mse_error_q2
                optimizer.zero_grad()
                tot_loss.backward()
                optimizer.step()
            # self.batch_weights[img_idx_to_batch_idx[img_idx]] ={k:v.to('cpu') for k, v in self.meta_network.state_dict().items()}
            self.batch_weights[img_idx_to_batch_idx[img_idx]] = self.meta_network.state_dict().copy()
        log.info("finetune images done")

    def predict(self, q1_images, q2_images, img_idx_to_batch_idx):
        '''
        :param q1_images: shape of (B,C,H,W)
        :param q2_images: shape of (B,C,H,W)
        :return:
        '''
        log.info("predict from meta model")
        q1_output = []
        q2_output = []
        for img_idx, (q1_img, q2_img) in enumerate(zip(q1_images, q2_images)):
            self.meta_network.load_state_dict(self.batch_weights[img_idx_to_batch_idx[img_idx]])
            self.meta_network.eval()
            q1_img = torch.unsqueeze(q1_img, 0)
            q2_img=  torch.unsqueeze(q2_img, 0)
            q1_logits = self.meta_network.forward(q1_img)
            q2_logits = self.meta_network.forward(q2_img)
            q1_output.append(q1_logits)
            q2_output.append(q2_logits)
        q1_output = torch.cat(q1_output, 0)
        q2_output = torch.cat(q2_output, 0)
        return q1_output, q2_output