import os
import socket

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from adversarial_defense.mixup_inference.utils.checkpoints import build_logger
from adversarial_defense.mixup_inference.utils.checkpoints import plot_image, save_context
from adversarial_defense.mixup_inference.utils import flags
import adversarial_defense.mixup_inference.torture
from adversarial_defense.mixup_inference.torture.loss_function import soft_cross_entropy
import torchvision
from adversarial_defense.mixup_inference.MI import pgd
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel

EVALUATE_EPOCH = 1
SAVE_EPOCH = 10
EPOCH_TOTAL = 200
HYPERPARAMETERS = None
DEFAULT_RESULTS_FOLDER_ARGUMENT = "Not Valid"
DEFAULT_RESULTS_FOLDER = "./results/"
FILES_TO_BE_SAVED = ["./", "./Torture", "./Torture/Models", "./Utils"]
KEY_ARGUMENTS = ["batch_size", "model", "data", "adv_ratio", "mixup_alpha"]
config = {
    "DEFAULT_RESULTS_FOLDER": DEFAULT_RESULTS_FOLDER,
    "FILES_TO_BE_SAVED": FILES_TO_BE_SAVED,
    "KEY_ARGUMENTS": KEY_ARGUMENTS
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
parser.add_argument("--gpu",type=str,required=True)
parser.add_argument("--results-folder",default=DEFAULT_RESULTS_FOLDER_ARGUMENT)
parser.add_argument("-k", "-key", "--key", default="")
parser.add_argument("-data", "--data", default="Caltech101")
parser.add_argument("-o", "--overwrite-results", action="store_true")
parser.add_argument("-bs", "--batch_size", type=int,default=50)
parser.add_argument("-lr", "--learning_rate",type=float, default=0.01)
parser.add_argument("-nw", "--num_workers", type=int,default=64)
parser.add_argument("-ar", "--adv_ratio", type=float,default=10)
parser.add_argument("-ma", "--mixup_alpha", type=float, default=0.)
parser.add_argument("--dataset", type=str,required=True)
parser.add_argument("--save_epoch",type=int, default=SAVE_EPOCH)
args = parser.parse_args()
logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, args, config)
logger.info("build dataloader")
with open("TotalList.txt", "a") as f:
    f.write(socket.gethostname() + ":" + args.results_folder + "\n")

def onehot(ind, num_classes):
    vector = np.zeros([num_classes])
    vector[ind] = 1
    return vector.astype(np.float32)

logger.info("build dataloader")
train_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, True)
val_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, False)
model = StandardModel(args.dataset, args.arch, no_grad=False)
model.cuda()
model.train()

def anneal_lr(epoch):
    if epoch < 100:
        return 1.
    elif epoch < 150:
        return 0.1
    else:
        return 0.01


pgd_kwargs = {
    "eps": 16. / 255.,
    "eps_iter": 4. / 255.,
    "nb_iter": 10,
    "norm": np.inf,
    "clip_min": -1,
    "clip_max": 1,
    "loss_fn": None,
}

def shuffle_minibatch(inputs, targets):
    if args.mixup_alpha == 0.:
        return inputs, targets
    mixup_alpha = args.mixup_alpha

    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]

    ma = np.random.beta(mixup_alpha, mixup_alpha, [batch_size, 1])
    ma_img = ma[:, :, None, None]

    inputs1 = inputs1 * torch.from_numpy(ma_img).cuda().float()
    inputs2 = inputs2 * torch.from_numpy(1 - ma_img).cuda().float()

    targets1 = targets1.float() * torch.from_numpy(ma).cuda().float()
    targets2 = targets2.float() * torch.from_numpy(1 - ma).cuda().float()

    inputs_shuffle = (inputs1 + inputs2).cuda()
    targets_shuffle = (targets1 + targets2).cuda()

    return inputs_shuffle, targets_shuffle

criterion = soft_cross_entropy
optimizer = optim.SGD(model.parameters(),
                      lr=args.learning_rate, momentum=0.9)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, [anneal_lr])

for epoch in range(EPOCH_TOTAL):  # loop over the dataset multiple times
    logger.info("Start Epoch {}".format(epoch))
    running_loss_1, running_loss_2 = 0.0, 0.0
    lr_scheduler.step(epoch)

    for i, data_batch in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        # if i == 19:
        #     break
        img_batch, label_batch = data_batch
        img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

        train_img_batch, train_label_batch = [], []
        model.eval()
        if args.adv_ratio > 0.:
            adv_x = pgd.projected_gradient_descent(model, img_batch,  **pgd_kwargs)
            adv_x_s, adv_label_batch_s = shuffle_minibatch(adv_x, label_batch)
            train_img_batch.append(adv_x_s)
            train_label_batch.append(adv_label_batch_s)

        if args.adv_ratio < 1.:
            img_batch_s, label_batch_s = shuffle_minibatch(
                img_batch, label_batch)
            train_img_batch.append(img_batch_s)
            train_label_batch.append(label_batch_s)

        train_img_batch = torch.cat(train_img_batch, dim=0)
        model.train()
        output_batch = model(train_img_batch)

        if len(train_label_batch) == 1:
            loss = criterion(output_batch, train_label_batch[0])
            if args.adv_ratio == 0.:
                running_loss_2 += loss.item()
            else:
                running_loss_1 += loss.item()
        else:
            output_batch1 = output_batch[:args.batch_size]
            output_batch2 = output_batch[args.batch_size:]
            loss1 = criterion(output_batch1, train_label_batch[0]) * args.adv_ratio
            loss2 = criterion(output_batch2, train_label_batch[1]) * (1. - args.adv_ratio)
            running_loss_1 += loss1.item()
            running_loss_2 += loss2.item()
            loss = loss1 + loss2

        model.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info('[%d] train loss: adv: %.3f, clean: %.3f' %
                (epoch + 1, running_loss_1 / i, running_loss_2 / i))

    if epoch % EVALUATE_EPOCH == 0:
        running_loss, correct, total = 0.0, 0.0, 0.0
        model.eval()
        for i, data_batch in enumerate(val_loader):
            # get the inputs; data is a list of [inputs, labels]
            img_batch, label_batch = data_batch
            img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
            output_batch = model(img_batch)
            loss = criterion(output_batch, label_batch)
            running_loss += loss.item()

            _, predicted = torch.max(output_batch.data, 1)
            _, label_ind = torch.max(label_batch.data, 1)
            correct += (predicted == label_ind).sum().item()
            total += label_batch.size(0)
        logger.info('[%d] test loss: %.3f, accuracy: %.3f' %
                    (epoch + 1, running_loss / i, correct / total))

    if epoch % args.save_epoch == 0 or epoch == EPOCH_TOTAL - 1:
        torch.save(model.state_dict(),
                   os.path.join(MODELS_FOLDER, "eopch{}.ckpt".format(epoch)))

logger.info('Finished Training')

