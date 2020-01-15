import glob
import os
import re

import glog as log
import torch
from torch import nn
from torch.optim import SGD, Adam
from torchvision import models as torch_models

from cifar_models import ResNet34
from config import PY_ROOT, IN_CHANNELS, CLASS_NUM
from constant_enum import SPLIT_DATA_PROTOCOL
from meta_simulator_model.meta_network import MetaNetwork
from optimizer.radam import RAdam


class MetaModelFinetune(object):
    def __init__(self, dataset, batch_size, meta_train_type, meta_train_data, distillation_loss):
        # FIXME 修改
        self.meta_model_path = "{}/train_pytorch_model/cross_arch_attack_2q/{}@{}@{}*data_{}@distill_loss_{}*".format(
            PY_ROOT,
            meta_train_type.upper(), dataset, SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II,
            meta_train_data, distillation_loss)
        self.meta_model_path = glob.glob(self.meta_model_path)
        pattern = re.compile(".*model_(.*?)@data.*inner_lr_(.*?)\..*")
        assert len(self.meta_model_path) > 0
        self.meta_model_path = self.meta_model_path[0]
        log.info("load meta model {}".format(self.meta_model_path))
        ma = pattern.match(os.path.basename(self.meta_model_path))
        arch = ma.group(1)
        self.inner_lr = float(ma.group(2))
        self.arch = arch
        self.dataset = dataset
        self.need_pair_distance = (meta_train_type == "2q_distillation") #FIXME
        # self.need_pair_distance = False
        meta_backbone = self.construct_model(arch, dataset)
        self.softmax = nn.Softmax(dim=1)
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.pair_wise_distance = nn.PairwiseDistance(p=2)
        # self.master_network = MetaNetwork(meta_backbone)
        # self.master_network.load_state_dict(
        #     torch.load(self.meta_model_path, map_location=lambda storage, location: storage)["state_dict"])
        # self.master_network.eval()
        # self.master_network.cuda()
        self.meta_model_pool = {}
        for img_idx in range(batch_size):
            meta_backbone = self.construct_model(arch, dataset)
            meta_network = MetaNetwork(meta_backbone)
            meta_network.load_state_dict(
                torch.load(self.meta_model_path, map_location=lambda storage, location: storage)["state_dict"])
            meta_network.eval()
            self.meta_model_pool[img_idx] = meta_network

    def construct_model(self, arch, dataset):
        if dataset in ["CIFAR-10", "MNIST", "FashionMNIST"]:
            network = ResNet34(IN_CHANNELS[dataset], CLASS_NUM[dataset])
        elif dataset == "TinyImageNet":
            if arch in torch_models.__dict__:
                network = torch_models.__dict__[arch](pretrained=True)
            num_classes = CLASS_NUM[dataset]
            if arch.startswith("resnet"):
                num_ftrs = network.fc.in_features
                network.fc = nn.Linear(num_ftrs, num_classes)
        return network

    def finetune(self, q1_images, q2_images, q1_gt_logits, q2_gt_logits, finetune_times):
        '''
        :param q1_images: shape of (B,T,C,H,W) where T is sequence length
        :param q2_images: shape of (B,T,C,H,W)
        :param q1_gt_logits: shape of (B, T, #class)
        :param q2_gt_logits: shape of (B, T, #class)
        :return:
        '''
        log.info("begin finetune images")
        for img_idx, (q1_images_tensor,q2_images_tensor, each_q1_gt_logits,each_q2_gt_logits) in enumerate(zip(q1_images,
                                                                                q2_images,q1_gt_logits,q2_gt_logits)):
            meta_network = self.meta_model_pool[img_idx]
            meta_network.cuda()
            # meta_network.copy_weights(self.master_network) # delete this line, only fine-tune 1 time for later iterations
            meta_network.train()
            optimizer = RAdam(meta_network.parameters(), lr=self.inner_lr)
            for _ in range(finetune_times):
                q1_output = meta_network.forward(q1_images_tensor)
                q2_output = meta_network.forward(q2_images_tensor)
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
            meta_network.eval()
            self.meta_model_pool[img_idx] = meta_network
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
            meta_network = self.meta_model_pool[img_idx]
            meta_network.eval()
            meta_network.cuda()
            q1_img = torch.unsqueeze(q1_img, 0)
            q2_img=  torch.unsqueeze(q2_img, 0)
            q1_logits = meta_network.forward(q1_img)
            q2_logits = meta_network.forward(q2_img)
            q1_output.append(q1_logits)
            q2_output.append(q2_logits)
        q1_output = torch.cat(q1_output, 0)
        q2_output = torch.cat(q2_output, 0)
        return q1_output, q2_output
