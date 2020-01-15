import json
import os
import time
from abc import ABCMeta, abstractmethod

import glog as log
import numpy as np
import torch
from torch.nn import functional as F

from config import CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker


class Attacker(metaclass=ABCMeta):
    def __init__(self, model, targeted, target_type, dataset, batch_size):
        self.model = model
        self.targeted = targeted
        self.target_type = target_type
        assert self.target_type in ["least_likely", "random"]
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_img_label_data_loader(dataset, batch_size, False)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)

    def xent_loss(self, logit, label, target=None):
        if target is not None:
            return -F.cross_entropy(logit, target, reduction='none')
        else:
            return F.cross_entropy(logit, label, reduction='none')

    @abstractmethod
    def make_adv_examples_iteration(self, step_index, adv_images, true_labels, target_labels, args):
        pass

    def make_adv_examples_minibatch(self,batch_index, max_queries, images, true_labels, target_labels, args):
        if not self.targeted:
            assert target_labels is None
        batch_size = images.size(0)
        with torch.no_grad():
            logits = self.model(images)
        pred = logits.argmax(dim=1)
        query = torch.zeros(batch_size).cuda()
        correct = pred.eq(true_labels).float()  # shape = (batch_size,)
        not_done = correct.clone()  # shape = (batch_size,)
        selected = torch.arange(batch_index * batch_size,
                                (batch_index + 1) * batch_size)  # 选择这个batch的所有图片的index
        adv_images = images.clone()
        for step_index in range(max_queries//2):
            adv_images = self.make_adv_examples_iteration(step_index, adv_images, true_labels, target_labels, args)
            with torch.no_grad():
                adv_logits = self.model(adv_images)
            adv_pred = adv_logits.argmax(dim=1)
            adv_prob = F.softmax(adv_logits, dim=1)
            adv_loss = self.xent_loss(adv_logits, true_labels, target_labels)
            query = query + 2 * not_done
            if self.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels)).float()   # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels).float() # 只要是跟原始label相等的，就还需要query，还没有成功
            success = (1 - not_done) * correct
            success_query = success * query
            not_done_loss = adv_loss * not_done
            not_done_prob = adv_prob[torch.arange(batch_size), true_labels] * not_done
            log.info('Attacking image {} - {} / {} , step {}, max query {}'.format(
                batch_index * batch_size, (batch_index + 1) * batch_size, self.total_images, step_index + 1,
                int(query.max().item())
            ))
            log.info('        correct: {:.4f}'.format(correct.mean().item()))
            log.info('       not_done: {:.4f}'.format(not_done.mean().item()))
            if success.sum().item() > 0:
                log.info('     mean_query: {:.4f}'.format(success_query[success.byte()].mean().item()))
                log.info('   median_query: {:.4f}'.format(success_query[success.byte()].median().item()))
            if not_done.sum().item() > 0:
                log.info('  not_done_loss: {:.4f}'.format(not_done_loss[not_done.byte()].mean().item()))
                log.info('  not_done_prob: {:.4f}'.format(not_done_prob[not_done.byte()].mean().item()))
            if not not_done.byte().any(): # all success
                break
        for key in ['query', 'correct', 'not_done',
                    'success', 'success_query', 'not_done_loss', 'not_done_prob']:
            value_all = getattr(self, key + "_all")
            value = eval(key)
            value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来
        if not_done.mean().item() > 0  :
            return adv_images[not_done].detach().cpu(), images[not_done].detach().cpu(), not_done.detach().cpu()
        return None, None, None

    def attack_dataset(self, args,  max_queries, result_dump_path):
        all_notdone_adv_images = []
        all_notdone_real_images = []
        all_notdone_true_labels = []
        # all_notdone_imageid = []
        for batch_idx, (images, true_labels) in enumerate(self.dataset_loader):
            images, true_labels = images.cuda(), true_labels.cuda()
            batch_size = images.size(0)
            if batch_idx * batch_size >= self.total_images:
                break

            if self.targeted:
                if self.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                  size=true_labels.size()).long().cuda()
                elif self.target_type == 'least_likely':
                    with torch.no_grad():
                        logits = self.model(images)
                    target_labels = logits.argmin(dim=1)
            else:
                target_labels = None

            adv_images, images, not_done_indexes = self.make_adv_examples_minibatch(batch_idx, max_queries, images,
                                                                                    true_labels, target_labels, args)
            if adv_images is not None:
                all_notdone_adv_images.append(adv_images)
                all_notdone_real_images.append(images)
                all_notdone_true_labels.append(true_labels)
                # all_notdone_imageid.append(image_id[not_done_indexes])
        all_notdone_adv_images = torch.cat(all_notdone_adv_images, 0)
        all_notdone_real_images = torch.cat(all_notdone_real_images, 0)
        all_notdone_true_labels = torch.cat(all_notdone_true_labels, 0)
        # all_notdone_imageid = torch.cat(all_notdone_imageid, 0)

        log.info('Attack finished ({} images)'.format(self.total_images))
        log.info('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        log.info('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            log.info(
                '     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            log.info(
                '   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
        if self.not_done_all.sum().item() > 0:
            log.info(
                '  avg not_done_loss: {:.4f}'.format(self.not_done_loss_all[self.not_done_all.byte()].mean().item()))
            log.info(
                '  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        log.info('Saving results to {}'.format(result_dump_path))
        result_dict = {"notdone_adv_images": all_notdone_adv_images.detach().numpy(),
                       "notdone_real_images": all_notdone_real_images.detach().numpy(), "notdone_true_labels": all_notdone_true_labels}
        meta_info_dict = {}
        for key in ['query', 'correct', 'not_done',
                    'success', 'success_query', 'not_done_loss', 'not_done_prob']:
            value_all = getattr(self, key + "_all").numpy()
            meta_info_dict[key] = value_all
        meta_info_dict['args'] = vars(args)
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, indent=4, sort_keys=True)
        result_dict.update(meta_info_dict)
        save_npz_path = os.path.dirname(result_dump_path) + "/not_done_images.npz"
        np.savez(save_npz_path, **result_dict)
        log.info("done, write stats info to {}".format(result_dump_path))

