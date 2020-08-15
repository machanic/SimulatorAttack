import glob
import os
import pickle
import random
import re
import copy
import numpy as np
from torch.utils import data

from config import IMAGE_SIZE, IN_CHANNELS, PY_ROOT, \
    MODELS_TRAIN_STANDARD, MODELS_TEST_STANDARD, MODELS_TRAIN_WITHOUT_RESNET
from constant_enum import SPLIT_DATA_PROTOCOL, LOAD_TASK_MODE


class MetaTaskDataset(data.Dataset):
    def __init__(self, dataset, adv_norm, data_loss_type, tot_num_tasks,
                 num_support, num_query,
                 load_mode, protocol, targeted, target_type="random",
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
        self.data_root_dir = "{}/data_surroage_gradient/{}/{}".format(PY_ROOT, dataset, "targeted_attack" if targeted else "untargeted_attack")
        self.pattern = re.compile(".*arch_(.*?)@.*")
        self.model_train_files = {}
        self.targeted = targeted
        self.tot_num_trn_tasks = tot_num_tasks
        self.num_support = num_support
        self.num_query = num_query
        print("Load {}".format(self.data_root_dir))
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
                                  "logits_loss_path":logits_loss_path, "gradient_path": gradient_path, "arch":model_name}
                                  # "logits":logits_loss_data["logits"],"loss":logits_loss_data['loss'],
                                  # "true_labels":logits_loss_data['true_labels'], "}
                # if self.targeted:
                #     each_file_json["target_labels"] = logits_loss_data["target_labels"]
                self.model_train_files[model_name] = each_file_json
                print("read {}".format(img_file_path))
        self.data_attack_type = adv_norm
        target_str = "untargeted" if not targeted else "target_{}".format(target_type)
        self.task_dump_txt_path = "{}/task_surrogate_gradient/{}/{}_{}_data_loss_type_{}_norm_{}_{}_tot_num_tasks_{}.pkl".format(PY_ROOT,
                                               dataset, dataset, protocol, data_loss_type, adv_norm, target_str, tot_num_tasks)
        self.store_tasks(load_mode, self.task_dump_txt_path, self.model_train_files)

    def store_tasks(self, load_mode, task_dump_txt_path, model_train_files):
        self.all_tasks = {}
        if load_mode == LOAD_TASK_MODE.LOAD and os.path.exists(task_dump_txt_path):
            with open(task_dump_txt_path, "rb") as file_obj:
                self.all_tasks = pickle.load(file_obj)
            return
        for i in range(self.tot_num_trn_tasks):  # 总共的训练任务个数，每次迭代都从这些任务去取
            if i % 1000 == 0:
                print("store {} tasks".format(i))
            test_model_name = random.choice(list(model_train_files.keys()))
            train_model_names = list(model_train_files.keys())
            train_model_names.remove(test_model_name)

            train_files_entries, test_file_entry = self.get_entry(train_model_names, test_model_name, self.num_support, self.num_query)
            self.all_tasks[i] = train_files_entries, test_file_entry
        self.dump_task(self.all_tasks, task_dump_txt_path)

    def dump_task(self, all_tasks, task_dump_txt_path):
        os.makedirs(os.path.dirname(task_dump_txt_path),exist_ok=True)
        with open(task_dump_txt_path, "wb") as file_obj:
            pickle.dump(all_tasks, file_obj, protocol=True)

    def get_entry(self, train_model_names, test_model_name, num_support, num_query):
        train_files = []
        for train_model in train_model_names:
            file_entry = copy.deepcopy(self.model_train_files[train_model])
            img_indexes = np.random.randint(0, file_entry["count"], size=num_support)
            file_entry["image_indexes"] = img_indexes
            train_files.append(file_entry)
        test_file_entry = copy.deepcopy(self.model_train_files[test_model_name])
        img_indexes = np.random.randint(0, test_file_entry["count"], size=num_query)
        test_file_entry["image_indexes"] = img_indexes

        return train_files, test_file_entry

    def __len__(self):
        return self.tot_num_trn_tasks

    def __getitem__(self, task_index):
        train_files_entries, test_file_entry = self.all_tasks[task_index]

        train_adv_images = []
        train_gradients = []
        train_logits = []
        train_true_labels = []
        train_target_labels = []
        for file_entry in train_files_entries:
            logits_loss_path = file_entry["logits_loss_path"]
            logits_loss_data = np.load(logits_loss_path)
            image_indexes = file_entry["image_indexes"]
            seq_len = file_entry["seq_len"]
            seq_index = random.randint(0, seq_len - 1)

            with open(file_entry["image_path"], "rb") as file_obj:
                for index in image_indexes:
                    adv_image = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                     IMAGE_SIZE[self.dataset][0],  IMAGE_SIZE[self.dataset][1]),
                                          offset=(index * seq_len + seq_index) * IN_CHANNELS[self.dataset] *
                                                 IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1] * 32 // 8)
                    train_adv_images.append(adv_image)
            with open(file_entry["gradient_path"], "rb") as file_obj:
                for index in image_indexes:
                    gradient = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                     IMAGE_SIZE[self.dataset][0],  IMAGE_SIZE[self.dataset][1]),
                                          offset=(index * seq_len + seq_index) * IN_CHANNELS[self.dataset] *
                                                 IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1] * 32 // 8)
                    train_gradients.append(gradient)
            for index in image_indexes:
                logits = logits_loss_data["logits"][index][seq_index]
                train_logits.append(logits)
                train_true_labels.append(logits_loss_data["true_labels"][index][seq_index])
                if self.targeted:
                    train_target_labels.append(logits_loss_data["target_labels"][index][seq_index])

        train_adv_images = np.stack(train_adv_images)
        train_gradients = np.stack(train_gradients)
        train_logits = np.stack(train_logits)
        train_true_labels = np.stack(train_true_labels)
        if self.targeted:
            train_target_labels = np.stack(train_target_labels)

        test_adv_images = []
        test_gradients = []
        test_logits = []
        test_true_labels = []
        test_target_labels = []
        image_indexes = test_file_entry["image_indexes"]
        seq_len = test_file_entry["seq_len"]
        seq_index = random.randint(0, seq_len - 1)
        logits_loss_path = test_file_entry["logits_loss_path"]
        logits_loss_data = np.load(logits_loss_path)
        with open(test_file_entry["image_path"], "rb") as file_obj:
            for index in image_indexes:
                adv_image = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                                  IMAGE_SIZE[self.dataset][0],
                                                                                  IMAGE_SIZE[self.dataset][1]),
                                      offset=(index * seq_len + seq_index) * IN_CHANNELS[self.dataset] *
                                             IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1] * 32 // 8)
                test_adv_images.append(adv_image)
        with open(test_file_entry["gradient_path"], "rb") as file_obj:
            for index in image_indexes:
                gradient = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                                 IMAGE_SIZE[self.dataset][0],
                                                                                 IMAGE_SIZE[self.dataset][1]),
                                     offset=(index * seq_len + seq_index) * IN_CHANNELS[self.dataset] *
                                            IMAGE_SIZE[self.dataset][0] * IMAGE_SIZE[self.dataset][1] * 32 // 8)
                test_gradients.append(gradient)
        for index in image_indexes:
            logits = logits_loss_data["logits"][index][seq_index]
            test_logits.append(logits)
            test_true_labels.append(logits_loss_data["true_labels"][index][seq_index])
            if self.targeted:
                test_target_labels.append(logits_loss_data["target_labels"][index][seq_index])
        test_adv_images = np.stack(test_adv_images)
        test_gradients = np.stack(test_gradients)
        test_logits = np.stack(test_logits)
        test_true_labels = np.stack(test_true_labels)
        if self.targeted:
            test_target_labels = np.stack(test_target_labels)
        if self.targeted:
            return train_adv_images, train_gradients, train_logits, train_true_labels, train_target_labels, \
                   test_adv_images, test_gradients, test_logits, test_true_labels, test_target_labels
        return train_adv_images, train_gradients, train_logits, train_true_labels, test_adv_images, test_gradients,\
               test_logits, test_true_labels


