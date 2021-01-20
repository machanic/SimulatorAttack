import argparse
import glob
import random
import re
import sys

sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
from dataset.standard_model import StandardModel
import glog as log
from config import PY_ROOT, MODELS_TRAIN_STANDARD, MODELS_TEST_STANDARD
from cifar_models_myself import *
from dataset.dataset_loader_maker import DataLoaderMaker
from advertorch.attacks import LinfPGDAttack, L2PGDAttack, FGSM, MomentumIterativeAttack
import numpy as np
import os

def generate_and_save_adv_examples(dataset, data_loader, attackers, save_dir_path):
    all_adv_images = []
    all_orig_labels = []
    all_clean_images = []
    for idx, (images, labels) in enumerate(data_loader):
        images = images.cuda()
        labels = labels.cuda()
        attacker = random.choice(attackers)
        adv_images = attacker.perturb(images, labels)
        all_adv_images.append(adv_images.detach().cpu().numpy())
        all_orig_labels.append(labels.detach().cpu().numpy())
        all_clean_images.append(images.detach().cpu().numpy())
        log.info("process data {}/{} done".format(idx, len(data_loader)))

    all_adv_images = np.concatenate(all_adv_images, 0)
    all_clean_images = np.concatenate(all_clean_images, 0)
    all_orig_labels = np.concatenate(all_orig_labels, 0).astype(np.int32)
    save_adv_images_path = save_dir_path + "/{}_adv_images.npy".format(dataset)
    save_clean_images_path = save_dir_path + "/{}_clean_images.npy".format(dataset)
    save_labels_path = save_dir_path + "/{}_labels.npy".format(dataset)
    shape_path = save_dir_path + "/{}_shape.txt".format(dataset)
    fp = np.memmap(save_adv_images_path, dtype='float32', mode='w+', shape=all_adv_images.shape)
    fp[:, :, :, :] = all_adv_images[:, :, :, :]
    del fp
    fp = np.memmap(save_clean_images_path, dtype='float32', mode='w+', shape=all_clean_images.shape)
    fp[:, :, :, :] = all_clean_images[:, :, :, :]
    del fp

    np.save(save_labels_path, all_orig_labels)
    with open(shape_path, "w") as file_obj:
        file_obj.write(str(all_adv_images.shape))
        file_obj.flush()
    log.info("Write to directory {} done".format(save_dir_path))

def get_already_gen_models(loss_type, save_path, datasetname):
    dir_path = "{}/{}/{}/{}_model_*_images.npy".format(save_path, loss_type, datasetname, datasetname)
    pattern = re.compile(".*{}_model_(.*?)_label.*".format(datasetname))
    model_names = set()
    for model_path in glob.glob(dir_path):
        ma = pattern.match(os.path.basename(model_path))
        arch = ma.group(1)
        model_names.add(arch)
    return model_names


def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def generate(datasetname, batch_size):
    save_dir_path = "{}/data_adv_defense/guided_denoiser".format(PY_ROOT)
    os.makedirs(save_dir_path, exist_ok=True)
    set_log_file(save_dir_path + "/generate_{}.log".format(datasetname))
    data_loader = DataLoaderMaker.get_img_label_data_loader(datasetname, batch_size, is_train=True)
    attackers = []
    for model_name in MODELS_TRAIN_STANDARD[datasetname] + MODELS_TEST_STANDARD[datasetname]:
        model = StandardModel(datasetname, model_name, no_grad=False)
        model = model.cuda().eval()
        linf_PGD_attack =LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031372, nb_iter=30,
                      eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
        l2_PGD_attack = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=1.0,
                                    nb_iter=30,clip_min=0.0, clip_max=1.0, targeted=False)
        FGSM_attack = FGSM(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"))
        momentum_attack = MomentumIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031372, nb_iter=30,
                      eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False)
        attackers.append(linf_PGD_attack)
        attackers.append(l2_PGD_attack)
        attackers.append(FGSM_attack)
        attackers.append(momentum_attack)
        log.info("Create model {} done!".format(model_name))

    generate_and_save_adv_examples(datasetname, data_loader, attackers, save_dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--dataset", type=str, default='CIFAR-10')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument("--batch_size",type=int,default=10)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    generate(args.dataset,  args.batch_size)