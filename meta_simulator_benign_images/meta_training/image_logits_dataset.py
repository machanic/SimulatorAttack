import glob
import os
import pickle
import random
import re

import numpy as np
import torch
from torch.utils import data

from config import IMAGE_SIZE, IN_CHANNELS, PY_ROOT, CLASS_NUM, \
    MODELS_TRAIN_STANDARD, MODELS_TEST_STANDARD, MODELS_TRAIN_WITHOUT_RESNET
from constant_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE


class MetaTaskDataset(data.Dataset):
    def __init__(self, dataset, tot_num_tasks, support_query_set_len, load_mode, protocol, without_resnet=False):
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
        self.data_root_dir = "{}/benign_images_logits_pair/{}/".format(PY_ROOT, dataset)
        self.pattern = re.compile("^(.*)_images.npy")
        self.train_files = []
        self.tot_num_trn_tasks = tot_num_tasks
        self.support_query_set_len = support_query_set_len
        print("dataset root is {}".format(self.data_root_dir))
        print("all models are {}".format(" , ".join(self.model_names)))
        for img_file_path in glob.glob(self.data_root_dir + "/*images.npy"):
            file_name = os.path.basename(img_file_path)
            ma = self.pattern.match(file_name)
            model_name = ma.group(1)
            if model_name in self.model_names:
                logits_path = img_file_path.replace("_images.npy", "_logits_labels.npz")
                logits_data = np.load(logits_path)
                logits_data = logits_data['logits']
                each_file_json = {"count": logits_data.shape[0], "image_path":img_file_path, "gt_logits":logits_data,
                                  "arch":model_name}
                self.train_files.append(each_file_json)
        self.task_dump_txt_path = "{}/task_benign_images_grads/{}_{}/{}_tot_num_tasks_{}.pkl".format(PY_ROOT, protocol,
                                               dataset, dataset, tot_num_tasks)
        self.store_tasks(load_mode, self.task_dump_txt_path, self.train_files)

    def store_tasks(self, load_mode, task_dump_txt_path, train_files):
        self.all_tasks = {}
        if load_mode == LOAD_TASK_MODE.LOAD and os.path.exists(task_dump_txt_path):
            with open(task_dump_txt_path, "rb") as file_obj:
                self.all_tasks = pickle.load(file_obj)
            return
        for i in range(self.tot_num_trn_tasks):  # 总共的训练任务个数，每次迭代都从这些任务去取
            if i % 1000 == 0:
                print("store {} tasks".format(i))
            file_entry = random.choice(train_files)
            entry = self.get_one_entry(file_entry)
            entry["task_idx"] = i
            self.all_tasks[i] = entry
        self.dump_task(self.all_tasks, task_dump_txt_path)

    def dump_task(self, all_tasks, task_dump_txt_path):
        os.makedirs(os.path.dirname(task_dump_txt_path),exist_ok=True)
        with open(task_dump_txt_path, "wb") as file_obj:
            pickle.dump(all_tasks, file_obj, protocol=True)

    def get_one_entry(self, file_entry):
        dict_with_img_index = {}
        dict_with_img_index.update(file_entry)
        count = file_entry["count"]
        dict_with_img_index["indexes"] = [random.randint(0, count-1) for _ in range(self.support_query_set_len)]
        return dict_with_img_index

    def __len__(self):
        return self.tot_num_trn_tasks

    def __getitem__(self, task_index):
        task_data = self.all_tasks[task_index]
        gt_logits = []
        for index in task_data["indexes"]:
            gt_logits.append(task_data["gt_logits"][index])
        gt_logits = torch.from_numpy(np.stack(gt_logits)).float()
        benign_images = []
        with open(task_data["image_path"], "rb") as file_obj:
            for index in task_data["indexes"]:
                benign_image = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                                   IMAGE_SIZE[self.dataset][0],
                                                                                   IMAGE_SIZE[self.dataset][1]),
                                       offset=index  * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                              IMAGE_SIZE[self.dataset][1] * 32 // 8)
                benign_images.append(benign_image)
        benign_images = torch.from_numpy(np.stack(benign_images))
        return benign_images, gt_logits

