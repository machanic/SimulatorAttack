import glob
import os
import pickle
import random
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils import data

from config import IMAGE_SIZE, IN_CHANNELS, PY_ROOT, ALL_MODELS, MODELS_TRAIN, MODELS_TEST
from constant_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE


class MetaTaskDataset(data.Dataset):
    """
    Support 和query数据用同样的PGD 40 sequence, support数据是指用0~20 PGD的前一半迭代，或指定几个监督信号， query数据是指用20~40的后一半
    """
    def __init__(self, data_loss_type, num_tot_tasks, dataset, load_mode, protocol):
        """
        Args:
            num_samples_per_class: num samples to generate "per class" in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.img_size = IMAGE_SIZE[dataset]
        self.dataset = dataset
        self.sequence_len = 40
        if protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II:
            self.model_names = MODELS_TRAIN
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_II_TEST_I:
            self.model_names = MODELS_TEST
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
            self.model_names = ALL_MODELS
        if self.dataset == "TinyImageNet":
            self.data_root_dir = "{}/data_PGD_without_sep_labels/{}/".format(PY_ROOT, data_loss_type)
            pattern_str = ".*_model_(.*?)_images.*"
        else:
            self.data_root_dir = "{}/data_PGD_40/{}/".format(PY_ROOT, data_loss_type)
            pattern_str = ".*_model_(.*?)_label.*"
        self.pattern = re.compile(pattern_str)
        self.train_files = []

        for img_file_path in glob.glob(self.data_root_dir + "{}/{}*_images.npy".format(self.dataset, self.dataset)):
            file_name = os.path.basename(img_file_path)
            ma = self.pattern.match(file_name)
            model_name = ma.group(1)
            if model_name in self.model_names:
                grad_file_path = img_file_path.replace("_images.npy","_grad.npy")
                with open(img_file_path.replace(".npy",".txt"), "r") as file_obj:
                    count = int(file_obj.read().strip())
                each_file_json = {"count": count, "arch":model_name, "grad_file":grad_file_path, "img_file":img_file_path}
                self.train_files.append(each_file_json)
        self.num_tot_trn_tasks = num_tot_tasks
        self.task_dump_txt_path = "{}/task/{}_{}/{}_data_loss_{}_tot_num_tasks_{}.pkl".format(PY_ROOT, protocol,
                                            dataset, dataset, data_loss_type, num_tot_tasks)
        self.store_tasks(load_mode, self.task_dump_txt_path, self.train_files)

    def store_tasks(self, load_mode, task_dump_txt_path, train_files):
        self.all_tasks = dict()
        if load_mode == LOAD_TASK_MODE.LOAD and os.path.exists(task_dump_txt_path):
            with open(task_dump_txt_path, "rb") as file_obj:
                self.all_tasks = pickle.load(file_obj)
            return

        for i in range(self.num_tot_trn_tasks):  # 总共的训练任务个数，每次迭代都从这些任务去取
            if i % 1000 == 0:
                print("store {} tasks".format(i))
            file_entry = random.choice(train_files)
            img_whole_path, grad_path = self.get_image_paths(file_entry)
            arch = file_entry["arch"]
            img_path, index = img_whole_path.split("#")
            index = int(index)
            self.all_tasks[i] = {"task_idx":i, "arch": arch, "img_path": img_path, "grad_path": grad_path,
                                          "seq_index":index}

        self.dump_task(self.all_tasks, task_dump_txt_path)

    def dump_task(self, all_tasks, task_dump_txt_path):
        os.makedirs(os.path.dirname(task_dump_txt_path),exist_ok=True)
        with open(task_dump_txt_path, "wb") as file_obj:
            pickle.dump(all_tasks, file_obj, protocol=True)

    def get_image_paths(self, file_entry):
        img_file_path = file_entry["img_file"]
        grad_file_path = file_entry["grad_file"]
        count = file_entry["count"]
        all_index_list = np.arange(count).tolist()
        seq_index = random.choice(all_index_list)
        img_whole_path = "{}#{}".format(img_file_path, seq_index)
        return img_whole_path, grad_file_path

    def __getitem__(self, task_index):
        data_json = self.all_tasks[task_index]
        image_path = data_json["img_path"]
        seq_index = data_json["seq_index"]
        arch_name = data_json["arch"]
        fobj = open(image_path, "rb")
        im = np.memmap(fobj, dtype='float32', mode='r', shape=(
            1, self.sequence_len, IN_CHANNELS[self.dataset], IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1]),
                       offset=seq_index * self.sequence_len * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1]
                              * 32 // 8).copy()
        adv_image = im.reshape(self.sequence_len,  IN_CHANNELS[self.dataset], IMAGE_SIZE[self.dataset][0],
                               IMAGE_SIZE[self.dataset][1])
        fobj.close()
        grad_path = data_json["grad_path"]
        fobj = open(grad_path, "rb")
        im = np.memmap(fobj, dtype='float32', mode='r', shape=(
            1, self.sequence_len, IN_CHANNELS[self.dataset], IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1]),
                       offset=seq_index * self.sequence_len * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1]
                              * 32 // 8).copy()
        grad_image = im.reshape(self.sequence_len, IN_CHANNELS[self.dataset], IMAGE_SIZE[self.dataset][0],
                                IMAGE_SIZE[self.dataset][1])
        fobj.close()
        adv_image = torch.from_numpy(adv_image)  # T, C, H, W
        grad_image = torch.from_numpy(grad_image)  # T, C, H, W
        return adv_image, grad_image, ALL_MODELS.index(arch_name)

    def __len__(self):
        return len(self.all_tasks)

