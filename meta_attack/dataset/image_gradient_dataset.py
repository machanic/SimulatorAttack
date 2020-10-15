import os.path as osp
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np

from config import IN_CHANNELS, CLASS_NUM, IMAGE_SIZE

class ImageGradientDataset(Dataset):
    def __init__(self, root, arch, dataset, k_shot, k_query,  shuffle=True):
        """
        :param root: root path of gradients file
        :param arch: the architecture name
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        """
        self.dataset = dataset
        self.num_channels = IN_CHANNELS[dataset]
        self.data_batch_size = 100
        self.batchsz = 50000 // self.data_batch_size if dataset != "ImageNet" else 10000 // self.data_batch_size # batch of set, not batch of imgs
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.root = root

        self.resize = IMAGE_SIZE[self.dataset][0]  # resize to
        self.transform = transforms.Compose([torch.from_numpy])
        # rescale makes the output to be within(0 - 1)
        self.n_way = 1
        # load the data
        self.image_path = osp.join(root, "{}_images.npy".format(arch))
        self.grad_path = osp.join(root, "{}_grads.npy".format(arch))
        self.label_path = osp.join(root, "{}_labels.npy".format(arch))

        self.idx_list = []
        for batch in range(self.batchsz):
            for data_batch in range(self.data_batch_size):
                self.idx_list.append(batch * self.data_batch_size + data_batch)  # all the index that we use
        self.idx_list = np.array(self.idx_list).astype(np.int32)
        if shuffle:
            order = np.arange(len(self.idx_list))
            np.random.shuffle(order)
            self.idx_list = self.idx_list[order]
        self.tot_images = self.batchsz * self.data_batch_size
        self.cutoff = self.k_shot / (self.k_shot + self.k_query) * self.tot_images
        self.cutoff = int(self.cutoff)
        self.maximum_index = self.cutoff // self.k_shot
        self.labels = np.load(self.label_path)


    def get_image_data(self, file_path, all_index_list):
        return_data = []
        with open(file_path, "rb") as file_obj:
            for index in all_index_list:
                real_index = self.idx_list[index].item()
                data = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                               IMAGE_SIZE[self.dataset][0],
                                                                               IMAGE_SIZE[self.dataset][1]),
                                   offset=real_index * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                          IMAGE_SIZE[self.dataset][1] * 32 // 8)
                return_data.append(data)
        return np.stack(return_data, 0)

    def get_grad_data(self, file_path, all_index_list):
        return_data = []
        with open(file_path, "rb") as file_obj:
            for index in all_index_list:
                real_index = self.idx_list[index].item()
                data = np.memmap(file_obj, dtype='float32', mode='r', shape=(IN_CHANNELS[self.dataset],
                                                                               IMAGE_SIZE[self.dataset][0],
                                                                               IMAGE_SIZE[self.dataset][1]),
                                   offset=real_index * IN_CHANNELS[self.dataset] * IMAGE_SIZE[self.dataset][0] * \
                                          IMAGE_SIZE[self.dataset][1] * 32 // 8)
                return_data.append(data)
        grads = np.stack(return_data, 0)
        std = grads.std(axis=(1, 2, 3))
        std = std.reshape((-1, 1, 1, 1))
        grads = grads / (std + 1e-23)
        return grads

    def get_label_data(self, all_index_list):
        return_labels = []
        for index in all_index_list:
            real_index = self.idx_list[index].item()
            return_labels.append(self.labels[real_index])
        return np.stack(return_labels, 0)

    def __getitem__(self, index):
        all_support_index_list = []
        for i in range(self.k_shot):
            support_index = i + index * self.k_shot
            all_support_index_list.append(support_index)
        all_query_index_list = []
        bias = self.cutoff
        for i in range(self.k_query):
            query_index = i + bias + index * self.k_query
            all_query_index_list.append(query_index)
        images_support = self.get_image_data(self.image_path, all_support_index_list)
        grads_support  = self.get_grad_data(self.grad_path, all_support_index_list)
        labels_support = self.get_label_data(all_support_index_list)

        images_query = self.get_image_data(self.image_path,all_query_index_list)
        grads_query = self.get_grad_data(self.grad_path, all_query_index_list)
        labels_query = self.get_label_data(all_query_index_list)

        support_x = torch.zeros(self.k_shot, self.num_channels, self.resize, self.resize)
        support_y = torch.zeros(self.k_shot, self.num_channels, self.resize, self.resize)
        query_x = torch.zeros(self.k_query, self.num_channels, self.resize, self.resize)
        query_y = torch.zeros(self.k_query, self.num_channels, self.resize, self.resize)
        support_label = torch.zeros(self.k_shot)
        query_label = torch.zeros(self.k_query)

        for i in range(self.k_shot):
            support_x[i] = self.transform(images_support[i])
            support_y[i] = self.transform(grads_support[i])
            support_label[i] = torch.tensor(labels_support[i]).long()
        for i in range(self.k_query):
            query_x[i] = self.transform(images_query[i])
            query_y[i] = self.transform(grads_query[i])
            query_label[i] = torch.tensor(labels_query[i]).long()
        return support_x, support_y, support_label, query_x, query_y, query_label

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.maximum_index
