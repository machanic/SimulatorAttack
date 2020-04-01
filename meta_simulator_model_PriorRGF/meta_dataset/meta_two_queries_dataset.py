import glob
import os
import pickle
import random
import re

import numpy as np
import torch
from torch.utils import data

from config import IMAGE_SIZE, IN_CHANNELS, PY_ROOT, CLASS_NUM, \
    MODELS_TRAIN_STANDARD, MODELS_TEST_STANDARD
from constant_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE


class TwoQueriesMetaTaskDataset(data.Dataset):
    def __init__(self, dataset, adv_norm, tot_num_tasks, protocol, targeted, target_type="random"):
        """
        Args:
            num_samples_per_class: num samples to generate "per class" in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.dataset = dataset
        if protocol == SPLIT_DATA_PROTOCOL.TRAIN_I_TEST_II:
            self.model_names = MODELS_TRAIN_STANDARD[dataset]
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_II_TEST_I:
            self.model_names = MODELS_TEST_STANDARD[dataset]
        elif protocol == SPLIT_DATA_PROTOCOL.TRAIN_ALL_TEST_ALL:
            self.model_names = MODELS_TRAIN_STANDARD[dataset] + MODELS_TEST_STANDARD[dataset]
        self.data_root_dir = "{}/data_prior_RGF_attack/{}/{}".format(PY_ROOT, dataset, "targeted_attack" if targeted else "untargeted_attack")
        self.pattern = re.compile(".*arch_(.*?)@.*")
        self.train_files = []
        self.targeted = targeted
        self.tot_num_trn_tasks = tot_num_tasks
        print("dataset root is {}".format(self.data_root_dir))
        print("all models are {}".format(" , ".join(self.model_names)))
        for q1_path in glob.glob(self.data_root_dir + "/dataset_{dataset}*@norm_{norm}@{target_str}@q1.npy".format(
                dataset=dataset, norm=adv_norm,  target_str="targeted_" + target_type if targeted else "untargeted")):
            file_name = os.path.basename(q1_path)
            ma = self.pattern.match(file_name)
            model_name = ma.group(1)
            if model_name in self.model_names:
                q2_path = q1_path.replace("q1.npy", "q2.npy")
                logits_q1_path = q1_path.replace("q1.npy","logits_q1.npy")
                logits_q2_path = q1_path.replace("q1.npy","logits_q2.npy")
                shape_q1_path = q1_path.replace("q1.npy", "q1_shape.txt")
                shape_q2_path = q1_path.replace("q1.npy","q2_shape.txt")
                with open(shape_q1_path, "r") as file_obj:
                    q1_shape = eval(file_obj.read().strip()) # N, 2, 10, C, H, W
                with open(shape_q2_path, "r") as file_obj:
                    q2_shape = eval(file_obj.read().strip()) # N, 2, 50, C, H, W

                each_file_json = {"q1_shape": q1_shape, "q2_shape": q2_shape,
                                  "q1_path":q1_path,  "q2_path":q2_path, "logits_q1_path":logits_q1_path,
                                  "logits_q2_path":logits_q2_path,
                                   "arch":model_name}
                self.train_files.append(each_file_json)
        self.adv_norm = adv_norm
        # target_str = "untargeted" if not targeted else "target_{}".format(target_type)
        # self.task_dump_txt_path = "{}/prior_RGF_task/{}_{}/{}_norm_{}_{}_tot_num_tasks_{}.pkl".format(PY_ROOT, protocol,
        #                                        dataset, dataset, adv_norm, target_str, tot_num_tasks)
        self.construct_tasks(self.train_files)

    def construct_tasks(self, train_files):
        self.all_tasks = {}
        for i in range(self.tot_num_trn_tasks):  # 总共的训练任务个数，每次迭代都从这些任务去取
            if i % 1000 == 0:
                print("store {} tasks".format(i))
            file_entry = random.choice(train_files) # 一个文件就是一个网络的数据
            entry = self.get_one_entry(file_entry)
            entry["task_idx"] = i
            self.all_tasks[i] = entry

    def get_one_entry(self, file_entry):
        dict_with_img_index = {}
        dict_with_img_index.update(file_entry)
        q1_shape = file_entry["q1_shape"]
        q2_shape = file_entry["q2_shape"]
        assert q1_shape[0] == q2_shape[0]
        img_idx = random.randint(0, q1_shape[0] - 1)
        dict_with_img_index["img_idx"] = img_idx
        return dict_with_img_index

    def __len__(self):
        return self.tot_num_trn_tasks

    def __getitem__(self, task_index):
        task_data = self.all_tasks[task_index]
        q1_shape = task_data["q1_shape"]  # N, 2, 10, C, H, W
        q2_shape = task_data["q2_shape"]  # N, 2, 50, C, H, W
        img_idx = task_data["img_idx"]

        with open(task_data["q1_path"], "rb") as file_obj:
            q1 = np.memmap(file_obj, dtype='float32', mode='r', shape=(q1_shape[1],q1_shape[2],q1_shape[3],q1_shape[4],q1_shape[5]),
                                   offset=img_idx * q1_shape[1] * q1_shape[2] *  q1_shape[3] *  q1_shape[4] * q1_shape[5] * 32 // 8)
        with open(task_data["q2_path"], "rb") as file_obj:
            q2 = np.memmap(file_obj, dtype='float32', mode='r', shape=(q2_shape[1],q2_shape[2],q2_shape[3],q2_shape[4],q2_shape[5]),
                                   offset=img_idx * q2_shape[1] * q2_shape[2] * q2_shape[3] * q2_shape[4] * q2_shape[5] * 32 // 8)
        with open(task_data["logits_q1_path"], "rb") as file_obj: # # B, 2, 10, #class
            q1_logits = np.memmap(file_obj, dtype='float32', mode='r', shape=(q1_shape[1], q1_shape[2], CLASS_NUM[self.dataset]),
                                  offset=img_idx * q1_shape[1] * q1_shape[2] * CLASS_NUM[self.dataset] * 32 // 8)
        with open(task_data["logits_q2_path"], "rb") as file_obj:  # B, 2, 50, #class
            q2_logits = np.memmap(file_obj, dtype='float32', mode='r',
                                  shape=(q2_shape[1], q2_shape[2], CLASS_NUM[self.dataset]),
                                  offset=img_idx * q2_shape[1] * q2_shape[2] * CLASS_NUM[self.dataset] * 32 // 8)

        q1_images = torch.from_numpy(q1) # 2, 10, C, H, W
        q2_images = torch.from_numpy(q2)  # 2, 50, C, H, W
        q1_logits = torch.from_numpy(q1_logits) # 2, 10, #class
        q2_logits = torch.from_numpy(q2_logits) # 2, 50, #class
        return q1_images, q2_images, q1_logits, q2_logits

