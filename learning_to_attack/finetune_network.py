from torch import nn
import glog as log
import torch
from torch.optim import Adam
from dataset.standard_model import MetaLearnerModelBuilder
from torch.nn import functional as F


class FinetuneModel(object):
    def __init__(self, dataset, batch_size, arch, lr):
        self.lr = lr
        self.arch = arch
        self.dataset = dataset
        # self.targeted = targeted
        self.mse_loss = nn.MSELoss()
        self.backbone = self.construct_model(arch, dataset)
        self.pretrained_weights = self.backbone.state_dict()
        self.backbone.eval()
        self.backbone.cuda()
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

    # def cw_loss(self, logits, label, target=None):
    #     if target is not None:
    #         # targeted cw loss: logit_t - max_{i\neq t}logit_i
    #         _, argsort = logits.sort(dim=1, descending=True)
    #         target_is_max = argsort[:, 0].eq(target).long()
    #         second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
    #         target_logit = logits[torch.arange(logits.shape[0]), target]
    #         second_max_logit = logits[torch.arange(logits.shape[0]), second_max_index]
    #         return second_max_logit - target_logit
    #     else:
    #         # untargeted cw loss: max_{i\neq y}logit_i - logit_y
    #         _, argsort = logits.sort(dim=1, descending=True)
    #         gt_is_max = argsort[:, 0].eq(label).long()
    #         second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
    #         gt_logit = logits[torch.arange(logits.shape[0]), label]
    #         second_max_logit = logits[torch.arange(logits.shape[0]), second_max_index]
    #         return gt_logit - second_max_logit
    #
    # def xent_loss(self, logits, true_labels, target_labels):
    #     if target_labels is not None:
    #         return F.cross_entropy(logits, target_labels, reduction='none')
    #     else:
    #         return -F.cross_entropy(logits, true_labels, reduction='none')
    #
    #
    # def loss(self, logits, true_labels, target_labels, loss_type):
    #     if loss_type == "xent_loss":
    #         if self.targeted:
    #             return self.xent_loss(logits,true_labels, target_labels)
    #         else:
    #             return self.xent_loss(logits,true_labels, target_labels)
    #     elif loss_type == "cw_loss":
    #         if self.targeted:
    #             return self.cw_loss(logits, true_labels, target_labels)
    #         else:
    #             return self.cw_loss(logits,true_labels, target_labels)


    def finetune(self, images, losses, finetune_times, is_first_finetune, img_idx_to_batch_idx):
        '''
        :param q1_images: shape of (B,T,C,H,W) where T is sequence length
        :param q1_gt_logits: shape of (B, T, #class)
        :return:
        '''
        log.info("begin finetune images")
        if is_first_finetune:
            for i in range(self.batch_size):
                self.batch_weights[i] = self.pretrained_weights
            self.backbone.load_state_dict(self.pretrained_weights)
        for img_idx, (images, losses) in enumerate(zip(images, losses)):
            self.backbone.load_state_dict(self.batch_weights[img_idx_to_batch_idx[img_idx]])
            # meta_network.copy_weights(self.master_network) # delete this line, only fine-tune 1 time for later iterations
            # self.meta_network.train()
            optimizer = Adam(self.backbone.parameters(), lr=self.lr)
            for _ in range(finetune_times):
                predict_loss = self.backbone.forward(images)
                tot_loss = self.mse_loss(predict_loss, losses)
                optimizer.zero_grad()
                tot_loss.backward()
                optimizer.step()
            self.batch_weights[img_idx_to_batch_idx[img_idx]] = self.backbone.state_dict().copy()
        log.info("finetune images done")

    def predict(self, images, img_idx_to_batch_idx):
        '''
        :param images: shape of (B,C,H,W)
        :return:
        '''
        log.info("predict from meta model")
        output_list = []
        for img_idx, img in enumerate(images):
            self.backbone.load_state_dict(self.batch_weights[img_idx_to_batch_idx[img_idx]])
            self.backbone.eval()
            img =  torch.unsqueeze(img, 0)
            logits = self.backbone.forward(img)
            output_list.append(logits)
        output_list = torch.cat(output_list, 0)
        return output_list