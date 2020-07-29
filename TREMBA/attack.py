import argparse

import json
import os
from types import SimpleNamespace

import glob

from torchvision.transforms import Normalize

from TREMBA.FCN import ImagenetEncoder, ImagenetDecoder
from torch import nn
import glog as log
import torch

from TREMBA.imagenet_model.resnet import resnet152_denoise
from dataset.dataset_loader_maker import DataLoaderMaker
from TREMBA.utils import Function, MarginLossSingle
from config import CLASS_NUM, PY_ROOT, MODELS_TEST_STANDARD
from dataset.standard_model import StandardModel
import numpy as np

from defensive_model import DefensiveModel


def EmbedBA(function, encoder, decoder, image, label, config, latent=None):
    device = image.device
    if latent is None:
        latent = encoder(image.unsqueeze(0)).squeeze().view(-1)
    momentum = torch.zeros_like(latent)
    dimension = len(latent)
    noise = torch.empty((dimension, config.sample_size), device=device)
    origin_image = image.clone()
    last_loss = []
    lr = config.lr
    for iter in range(config.num_iters + 1):
        perturbation = torch.clamp(decoder(latent.unsqueeze(0)).squeeze(0) * config.epsilon,
                                   -config.epsilon, config.epsilon)
        logit, loss = function(torch.clamp(image + perturbation, 0, 1), label)
        if config.targeted:
            success = torch.argmax(logit, dim=1) == label
        else:
            success = torch.argmax(logit, dim=1) != label
        last_loss.append(loss.item())

        if function.current_counts > config.max_queries:
            break

        if bool(success.item()):
            return True, torch.clamp(image + perturbation, 0, 1), function.current_counts

        nn.init.normal_(noise)
        noise[:, config.sample_size // 2:] = -noise[:, :config.sample_size // 2]
        latents = latent.repeat(config.sample_size, 1) + noise.transpose(0, 1) * config.sigma
        perturbations = torch.clamp(decoder(latents) * config.epsilon, -config.epsilon, config.epsilon)
        _, losses = function(torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1), label)

        grad = torch.mean(losses.expand_as(noise) * noise, dim=1)

        # if iter % config.log_interval == 0 and config.print_log:
        #     log.info("iteration: {} loss: {}, l2_deviation {}".format(iter, float(loss.item()),
        #                                                            float(torch.norm(perturbation))))

        momentum = config.momentum * momentum + (1 - config.momentum) * grad

        latent = latent - lr * momentum

        last_loss = last_loss[-config.plateau_length:]
        if (last_loss[-1] > last_loss[0] + config.plateau_overhead or last_loss[-1] > last_loss[0] and last_loss[
            -1] < 0.6) and len(last_loss) == config.plateau_length:
            if lr > config.lr_min:
                lr = max(lr / config.lr_decay, config.lr_min)
            last_loss = []

    return False,function.current_counts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='config file')
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--norm", type=str, choices=["linf","l2"], required=True)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--test_archs", action="store_true")
    parser.add_argument("--dataset",type=str, required=True)
    parser.add_argument("--max_queries", type=int, default=10000, help="Maximum number of queries.")
    parser.add_argument("--OSP", action="store_true", help="whether to use optimal starting point")
    args = parser.parse_args()
    return args

def get_exp_dir_name(dataset, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'TREMBA_attack_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        dirname = 'TREMBA_attack-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def main():
    args = get_args()
    with open(args.config) as config_file:
        state = json.load(config_file)["attack"][args.targeted]
        state = SimpleNamespace(**state)
    if args.save_prefix is not None:
        state.save_prefix = args.save_prefix
    if args.arch is not None:
        state.arch = args.arch
    if args.test_archs is not None:
        state.test_archs = args.test_archs
    state.OSP = args.OSP
    state.targeted = args.targeted

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    targeted_str = "untargeted" if not state.targeted else "targeted"
    if state.targeted:
        save_name = "{}/train_pytorch_model/TREMBA/{}_{}_generator.pth.tar".format(PY_ROOT, args.dataset, targeted_str)
    else:
        save_name = "{}/train_pytorch_model/TREMBA/{}_{}_generator.pth.tar".format(PY_ROOT, args.dataset, targeted_str)
    weight = torch.load(save_name, map_location=device)["state_dict"]
    data_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size)
    encoder_weight = {}
    decoder_weight = {}
    for key, val in weight.items():
        if key.startswith('0.'):
            encoder_weight[key[2:]] = val
        elif key.startswith('1.'):
            decoder_weight[key[2:]] = val
    archs = []
    if args.test_archs:
        if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT, args.dataset, arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
                else:
                    log.info(test_model_path + " does not exists!")
        elif args.dataset == "TinyImageNet":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(
                    root=PY_ROOT, dataset=args.dataset, arch=arch)
                test_model_path = list(glob.glob(test_model_list_path))
                if test_model_path and os.path.exists(test_model_path[0]):
                    archs.append(arch)
        else:
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
                    PY_ROOT, args.dataset, arch)
                test_model_list_path = list(glob.glob(test_model_list_path))
                if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
                    continue
                archs.append(arch)
        args.arch = ",".join(archs)
    else:
        archs.append(args.arch)

    args.exp_dir = get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args)
    for arch in archs:
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        if args.OSP:
            if state.source_model_name == "Adv_Denoise_Resnet152":
                source_model = resnet152_denoise()
                loaded_state_dict = torch.load((os.path.join('{}/train_pytorch_model/TREMBA'.format(PY_ROOT),
                                                             state.source_model_name +".pth.tar")))
                source_model.load_state_dict(loaded_state_dict)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                # FIXME 仍然要改改
                source_model = nn.Sequential(
                    Normalize(mean, std),
                    source_model
                )
                source_model.to(device)
                source_model.eval()

        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)

        model.eval()
        encoder = ImagenetEncoder()
        decoder = ImagenetDecoder(args.dataset)
        encoder.load_state_dict(encoder_weight)
        decoder.load_state_dict(decoder_weight)
        model.to(device)
        encoder.to(device)
        encoder.eval()
        decoder.to(device)
        decoder.eval()
        F = Function(model, state.batch_size, state.margin, CLASS_NUM[args.dataset], state.targeted)
        total_success = 0
        count_total = 0
        queries = []
        not_done = []
        correct_all = []

        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            correct = torch.argmax(logits, dim=1).eq(labels).item()
            correct_all.append(int(correct))
            if correct:
                if args.targeted:
                    if args.target_type == 'random':
                        target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset], size=labels.size()).long().cuda()
                        invalid_target_index = target_labels.eq(labels)
                        while invalid_target_index.sum().item() > 0:
                            target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                                size=target_labels[
                                                                                    invalid_target_index].shape).long().cuda()
                            invalid_target_index = target_labels.eq(labels)
                    elif args.target_type == 'least_likely':
                        with torch.no_grad():
                            logit = model(images)
                        target_labels = logit.argmin(dim=1)
                    elif args.target_type == "increment":
                        target_labels = torch.fmod(labels + 1, CLASS_NUM[args.dataset]).cuda()
                    labels = target_labels[0].item()
                else:
                    labels = labels[0].item()
                if args.OSP:
                    hinge_loss = MarginLossSingle(state.white_box_margin, state.target)
                    images.requires_grad = True
                    latents = encoder(images)
                    for k in range(state.white_box_iters):
                        perturbations = decoder(latents) * state.epsilon
                        logits = source_model(torch.clamp(images + perturbations, 0, 1))
                        loss = hinge_loss(logits, labels)
                        grad = torch.autograd.grad(loss, latents)[0]
                        latents = latents - state.white_box_lr * grad
                    with torch.no_grad():
                        success, adv, query_count = EmbedBA(F, encoder, decoder, images[0], labels, state, latents.view(-1))
                else:
                    with torch.no_grad():
                        success, adv, query_count = EmbedBA(F, encoder, decoder, images[0], labels, state)
                not_done.append(1 - int(success))
                total_success += int(success)
                count_total += int(correct)
                if success:
                    queries.append(query_count)
                else:
                    queries.append(args.max_queries)

                log.info("image: {} eval_count: {} success: {} average_count: {} success_rate: {}".format(i, F.current_counts,
                                                                                                       success, F.get_average(),
                                                                                        float(total_success) / float(count_total)))
                F.new_counter()
            else:
                queries.append(0)
                not_done.append(1)
                log.info("The {}-th image is already classified incorrectly.".format(i))
        correct_all = np.concatenate(correct_all, axis=0).astype(np.int32)
        query_all = np.array(queries).astype(np.int32)
        not_done_all = np.array(not_done).astype(np.int32)
        success = (1 - not_done_all) * correct_all
        success_query = success * query_all
        meta_info_dict = {"query_all": query_all.tolist(), "not_done_all": not_done_all.tolist(),
                          "correct_all": correct_all.tolist(),
                          "mean_query": np.mean(success_query[np.nonzero(success)[0]]).item(),
                          "max_query": np.max(success_query[np.nonzero(success)[0]]).item(),
                          "median_query": np.median(success_query[np.nonzero(success)[0]]).item(),
                          "avg_not_done": np.mean(not_done_all[np.nonzero(correct_all)[0]].astype(np.float32)).item(),
                          "args": vars(args)}

        with open(save_result_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("Done, write stats info to {}".format(save_result_path))
