import glob
import os
import pickle
import random
import re

import numpy as np
import torch
from torch.utils import data

from config import IMAGE_SIZE, IN_CHANNELS, PY_ROOT, \
    MODELS_TRAIN_STANDARD, MODELS_TEST_STANDARD, MODELS_TRAIN_WITHOUT_RESNET
from constant_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE


class MetaTaskDataset(data.Dataset):
    def __init__(self, dataset, adv_norm, data_loss_type, tot_num_tasks, load_mode, protocol, targeted, target_type="increment",
                 without_resnet=False):
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
        self.data_root_dir = "{}/data_surroage_gradient_targeted_increment/{}/{}".format(PY_ROOT, dataset, "targeted_attack" if targeted else "untargeted_attack")
        self.pattern = re.compile(".*arch_(.*?)@.*")
        self.train_files = []
        self.targeted = targeted
        self.tot_num_trn_tasks = tot_num_tasks

        for img_file_path in glob.glob(self.data_root_dir + "/dataset_{dataset}@*@norm_{norm}@loss_{loss_type}@{target_str}@images.npy".format(
                dataset=dataset, norm=adv_norm, loss_type=data_loss_type, target_str="targeted" if targeted else "untargeted")):
            file_name = os.path.basename(img_file_path)
            ma = self.pattern.match(file_name)
            model_name = ma.group(1)
            if model_name in self.model_names:
                gradient_path = img_file_path.replace("images.npy", "gradients.npy")
                logits_loss_path = img_file_path.replace("images.npy","logits_loss.npz")
                shape_path = img_file_path.replace("images.npy", "shape.txt")
                with open(shape_path, "r") as file_obj:
                    shape = eval(file_obj.read().strip())
                count = shape[0]
                seq_len = shape[1]
                # logits_loss_data = np.load(logits_loss_path)  # logits shape = B,T,#class,  loss shape = (B,T), target_label shape = (B,T)
                each_file_json = {"count": count, "seq_len":seq_len, "image_path":img_file_path,
                                  "logits_loss_path":logits_loss_path, "gradient_path": gradient_path,
                                   "arch":model_name}
                self.train_files.append(each_file_json)
        self.data_attack_type = adv_norm
        target_str = "untargeted" if not targeted else "target_{}".format(target_type)
        self.task_dump_txt_path = "{}/task_surrogate_grad_finetune_attack/{}_{}/{}_data_loss_type_{}_norm_{}_{}_tot_num_tasks_{}.pkl".format(PY_ROOT, protocol,
                                               dataset, dataset, data_loss_type, adv_norm, target_str, tot_num_tasks)
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
        image_index = random.randint(0, count-1)
        dict_with_img_index["image_index"] = image_index
        return dict_with_img_index

    def __len__(self):
        return self.tot_num_trn_tasks

    def __getitem__(self, task_index):
        task_data = self.all_tasks[task_index]
        image_index = task_data["image_index"]
        seq_len = task_data["seq_len"]
        with open(task_data["image_path"], "rb") as file_obj:
            adv_images = np.memmap(file_obj, dtype='float32', mode='r', shape=(seq_len, IN_CHANNELS[self.dataset],
                                            IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1]),
                                   offset= image_index * seq_len * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                          IMAGE_SIZE[self.dataset][1] * 32 // 8)
            adv_images = torch.from_numpy(adv_images)
        with open(task_data["gradient_path"], "rb") as file_obj:
            gradients = np.memmap(file_obj, dtype='float32', mode='r', shape=(seq_len, IN_CHANNELS[self.dataset],
                                            IMAGE_SIZE[self.dataset][0], IMAGE_SIZE[self.dataset][1]),
                                   offset= image_index * seq_len * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                          IMAGE_SIZE[self.dataset][1] * 32 // 8)
            gradients = torch.from_numpy(gradients)

        data = np.load(task_data["logits_loss_path"])
        logits = torch.from_numpy(data["logits"][image_index])  # T,#class
        true_labels = torch.from_numpy(data["true_labels"][image_index])  # T
        if self.targeted:
            target_labels = torch.from_numpy(data["target_labels"][image_index])
            return adv_images, gradients, logits, true_labels, target_labels
        return adv_images, gradients, logits, true_labels

