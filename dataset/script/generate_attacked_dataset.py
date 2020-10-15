import glob
import os
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import torch
import numpy as np
import glog as log
from config import PY_ROOT, MODELS_TRAIN_STANDARD, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel, MetaLearnerModelBuilder


def generate_attacked_dataset(dataset, num_sample, models):
    selected_images = []
    selected_images_big = []
    selected_true_labels = []
    selected_img_id = []
    total_count = 0
    data_loader = DataLoaderMaker.get_imgid_img_label_data_loader(dataset, 500, False, seed=1234)
    print("begin select")
    if dataset != "ImageNet":
        for image_id, images, labels in data_loader:
            images_gpu = images.cuda()
            pred_eq_true_label = []
            for model in models:
                with torch.no_grad():
                    logits = model(images_gpu)
                pred = logits.max(1)[1]
                correct = pred.detach().cpu().eq(labels).long()
                pred_eq_true_label.append(correct.detach().cpu().numpy())
            pred_eq_true_label = np.stack(pred_eq_true_label).astype(np.uint8) # M, B
            pred_eq_true_label = np.bitwise_and.reduce(pred_eq_true_label, axis=0)  # 1,0,1,1,1
            current_select_count = len(np.where(pred_eq_true_label)[0])
            total_count += current_select_count
            selected_image = images.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]]
            selected_images.append(selected_image)
            selected_true_labels.append(labels.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            selected_img_id.append(image_id.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            if total_count >= num_sample:
                break
    else:
        for image_id, images, images_big, labels in data_loader:
            images_gpu = images.cuda()
            pred_eq_true_label = []
            for model in models:
                with torch.no_grad():
                    logits = model(images_gpu)
                pred = logits.max(1)[1]
                correct = pred.detach().cpu().eq(labels).long()
                pred_eq_true_label.append(correct.detach().cpu().numpy())
            pred_eq_true_label = np.stack(pred_eq_true_label).astype(np.uint8) # M, B
            pred_eq_true_label = np.bitwise_and.reduce(pred_eq_true_label, axis=0)  # 1,0,1,1,1
            current_select_count = len(np.where(pred_eq_true_label)[0])
            total_count += current_select_count
            selected_image = images.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]]
            selected_images.append(selected_image)

            selected_image_big = images_big.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]]
            selected_images_big.append(selected_image_big)
            selected_true_labels.append(labels.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            selected_img_id.append(image_id.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            if total_count >= num_sample:
                break
    selected_images = np.concatenate(selected_images, 0)
    if dataset=="ImageNet":
        selected_images_big = np.concatenate(selected_images_big, 0)
        selected_images_big = selected_images_big[:num_sample]
    selected_true_labels = np.concatenate(selected_true_labels, 0)
    selected_img_id = np.concatenate(selected_img_id, 0)

    selected_images = selected_images[:num_sample]
    selected_true_labels = selected_true_labels[:num_sample]
    selected_img_id = selected_img_id[:num_sample]
    return selected_images, selected_images_big, selected_true_labels, selected_img_id

def save_selected_images(dataset, selected_images, selected_big_images, selected_true_labels, selected_img_id, save_path):
    if dataset != "ImageNet":
        np.savez(save_path, images=selected_images, labels=selected_true_labels, image_id=selected_img_id)
    else:
        np.savez(save_path, images_224x224=selected_images, images_299x299=selected_big_images,
                 labels=selected_true_labels, image_id=selected_img_id)


def load_models(dataset):
    archs = []
    model_path_list = []
    if dataset == "CIFAR-10" or dataset == "CIFAR-100":
        for arch in ["resnet-110","WRN-28-10","WRN-34-10","resnext-8x64d","resnext-16x64d"]:
            test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                PY_ROOT, dataset, arch)
            if os.path.exists(test_model_path):
                archs.append(arch)
                model_path_list.append(test_model_path)
            else:
                log.info(test_model_path + " does not exist!")
    elif dataset == "TinyImageNet":
        # for arch in ["vgg11_bn","resnet18","vgg16_bn","resnext64_4","densenet121"]:
        for arch in MODELS_TEST_STANDARD[dataset]:
            test_model_path = "{}/train_pytorch_model/real_image_model/{}@{}@*.pth.tar".format(
                PY_ROOT, dataset, arch)
            test_model_path = list(glob.glob(test_model_path))[0]
            if os.path.exists(test_model_path):
                archs.append(arch)
                model_path_list.append(test_model_path)
            else:
                log.info(test_model_path + "does not exist!")
    else:
        for arch in ["inceptionv3","inceptionv4", "inceptionresnetv2","resnet101", "resnet152"]:
            test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
                PY_ROOT, dataset, arch)
            test_model_path = list(glob.glob(test_model_list_path))
            if len(test_model_path) == 0:  # this arch does not exists in args.dataset
                continue
            archs.append(arch)
            model_path_list.append(test_model_path[0])
    models = []
    print("begin construct model")
    if dataset == "TinyImageNet":
        for idx, arch in enumerate(archs):
            model = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
            model_path = model_path_list[idx]
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage)["state_dict"])
            model.cuda()
            model.eval()
            models.append(model)
    else:
        for arch in archs:
            model = StandardModel(dataset, arch, no_grad=True)
            model.cuda()
            model.eval()
            models.append(model)
    print("end construct model")
    return models

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    dataset = "ImageNet"
    models = load_models(dataset)
    selected_images, selected_images_big, selected_true_labels, selected_img_id = generate_attacked_dataset(dataset, 30000, models)
    save_path = "{}/attacked_images/{}/{}_images_for_candidate.npz".format(PY_ROOT, dataset, dataset)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_selected_images(dataset, selected_images, selected_images_big, selected_true_labels, selected_img_id, save_path)
