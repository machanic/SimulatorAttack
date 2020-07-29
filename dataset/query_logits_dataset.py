import glob
import os
import pickle
import random
import re

import copy
import numpy as np
import torch
from torch.utils import data

from config import IMAGE_SIZE, IN_CHANNELS, PY_ROOT, CLASS_NUM, \
    MODELS_TRAIN_STANDARD, MODELS_TEST_STANDARD, MODELS_TRAIN_WITHOUT_RESNET
from constant_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE


class QueryLogitsDataset(data.Dataset):
    def __init__(self, dataset, adv_norm, data_loss_type, protocol, targeted, target_type="random", without_resnet=False):
        """
        Args:
            num_samples_per_class: num samples to generate "per class" in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.dataset = dataset
        if not without_resnet:
            if protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II:
                self.model_names = MODELS_TRAIN_STANDARD[dataset]
            elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_II_TEST_I:
                self.model_names = MODELS_TEST_STANDARD[dataset]
            elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
                self.model_names = MODELS_TRAIN_STANDARD[dataset] + MODELS_TEST_STANDARD[dataset]
        else:
            if protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II:
                self.model_names = MODELS_TRAIN_WITHOUT_RESNET[dataset]
            elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_II_TEST_I:
                self.model_names = MODELS_TEST_STANDARD[dataset]
            elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
                self.model_names = MODELS_TRAIN_WITHOUT_RESNET[dataset] + MODELS_TEST_STANDARD[dataset]
        self.data_root_dir = "{}/data_bandit_attack/{}/{}".format(PY_ROOT, dataset, "targeted_attack" if targeted else "untargeted_attack")
        self.pattern = re.compile(".*arch_(.*?)@.*")
        self.train_files = []
        self.targeted = targeted
        print("visit : {}".format(self.data_root_dir + "/dataset_{dataset}@attack_{norm}*loss_{loss_type}@{target_str}@images.npy".format(
                dataset=dataset, norm=adv_norm, loss_type=data_loss_type, target_str="targeted_" + target_type if targeted else "untargeted")))
        for img_file_path in glob.glob(self.data_root_dir + "/dataset_{dataset}@attack_{norm}*loss_{loss_type}@{target_str}@images.npy".format(
                dataset=dataset, norm=adv_norm, loss_type=data_loss_type, target_str="targeted_" + target_type if targeted else "untargeted")):
            file_name = os.path.basename(img_file_path)
            ma = self.pattern.match(file_name)
            model_name = ma.group(1)
            if model_name in self.model_names:
                print("read the data of model {} as training data".format(model_name))
                q1_path = img_file_path.replace("images.npy","q1.npy")
                q2_path = img_file_path.replace("images.npy", "q2.npy")
                logits_q1_path = img_file_path.replace("images.npy","logits_q1.npy")
                logits_q2_path = img_file_path.replace("images.npy","logits_q2.npy")
                shape_path = img_file_path.replace("images.npy", "shape.txt")
                gt_labels_path = img_file_path.replace("images.npy","gt_labels.npy")
                with open(shape_path, "r") as file_obj:
                    shape = eval(file_obj.read().strip())
                count = shape[0]
                seq_len = shape[1]
                gt_labels = np.load(gt_labels_path)
                each_file_json = {"count": count, "seq_len":seq_len, "image_path":img_file_path, "q1_path":q1_path,
                                  "q2_path":q2_path, "logits_q1_path":logits_q1_path,  "logits_q2_path":logits_q2_path,
                                  "gt_labels":gt_labels, "arch":model_name}
                if self.targeted:
                    targets_path = img_file_path.replace("images.npy","targets.npy")
                    each_file_json["targets"] = np.load(targets_path)
                self.train_files.append(each_file_json)
        self.data_attack_type = adv_norm
        self.train_data = self.generate_train_data_index(self.train_files)

    def generate_train_data_index(self, train_files):
        train_data = []
        for train_file in train_files:
            count = train_file["count"]
            for i in range(count):
                train_file_clone = copy.deepcopy(train_file)
                train_file_clone["index"] = i
                train_data.append(train_file_clone)
        return train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, task_index):
        task_data = self.train_data[task_index]
        seq_len = task_data["seq_len"]
        index = task_data["index"]
        seq_index = random.randint(0, seq_len-1)
        with open(task_data["image_path"], "rb") as file_obj:
            adv_images = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                               IMAGE_SIZE[self.dataset][0],
                                                                               IMAGE_SIZE[self.dataset][1]),
                                   offset=(index * seq_len + seq_index) * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                          IMAGE_SIZE[self.dataset][1] * 32 // 8)
        with open(task_data["q1_path"], "rb") as file_obj:
            q1 = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                               IMAGE_SIZE[self.dataset][0],
                                                                               IMAGE_SIZE[self.dataset][1]),
                                   offset=(index * seq_len + seq_index) * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                          IMAGE_SIZE[self.dataset][1] * 32 // 8)
        with open(task_data["q2_path"], "rb") as file_obj:
            q2 = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                              IMAGE_SIZE[self.dataset][0],
                                                                              IMAGE_SIZE[self.dataset][1]),
                                  offset=(index * seq_len + seq_index) * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                         IMAGE_SIZE[self.dataset][1] * 32 // 8)
        with open(task_data["logits_q1_path"], "rb") as file_obj:
            q1_logits = np.memmap(file_obj, dtype='float32', mode='r', shape=CLASS_NUM[self.dataset],
                                  offset=(index * seq_len + seq_index) * CLASS_NUM[self.dataset] * 32 // 8)
        with open(task_data["logits_q2_path"], "rb") as file_obj:
            q2_logits = np.memmap(file_obj, dtype='float32', mode='r', shape=CLASS_NUM[self.dataset],
                                  offset=(index * seq_len + seq_index) * CLASS_NUM[self.dataset] * 32 // 8)
        q1_images = adv_images + q1
        q2_images = adv_images + q2

        q1_images = torch.from_numpy(q1_images)
        q2_images = torch.from_numpy(q2_images)
        q1_logits = torch.from_numpy(q1_logits)
        q2_logits = torch.from_numpy(q2_logits)
        return q1_images, q2_images, q1_logits, q2_logits



