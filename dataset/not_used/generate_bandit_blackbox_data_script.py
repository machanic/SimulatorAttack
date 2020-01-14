import argparse
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import glob
import json
import os
import re
from types import SimpleNamespace
import numpy as np
import torch as ch
from torch.nn import DataParallel
from torch.nn.modules import Upsample
from cifar_models import *
from config import IN_CHANNELS, IMAGE_SIZE, PY_ROOT
from dataset_loader_maker import DataLoaderMaker
from model_constructor import ModelConstructor


class BanditAttack(object):
    @staticmethod
    def norm(t):
        assert len(t.shape) == 4
        norm_vec = ch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    ###
    # Below is different optimization steps
    # All take the form of func(x, g, lr)
    # eg: exponentiated gradients
    # l2/linf: projected gradient descent
    @staticmethod
    def eg_step(x, g, lr):
        real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
        pos = real_x * ch.exp(lr * g)
        neg = (1 - real_x) * ch.exp(-lr * g)
        new_x = pos / (pos + neg)
        return new_x * 2 - 1

    @staticmethod
    def linf_step(x, g, lr):
        return x + lr * ch.sign(g)

    @staticmethod
    def l2_prior_step(x, g, lr):
        new_x = x + lr * g / BanditAttack.norm(g)
        norm_new_x = BanditAttack.norm(new_x)
        norm_mask = (norm_new_x < 1.0).float()
        return new_x * norm_mask + (1 - norm_mask) * new_x / norm_new_x

    @staticmethod
    def gd_prior_step(x, g, lr):
        return x + lr * g

    @staticmethod
    def l2_image_step(x, g, lr):
        return x + lr * g / BanditAttack.norm(g)

    ##
    # Projection steps for l2 and linf constraints:
    # All take the form of func(new_x, old_x, epsilon)
    ##
    @staticmethod
    def l2_proj(image, eps):
        orig = image.clone()

        def proj(new_x):
            delta = new_x - orig
            out_of_bounds_mask = (BanditAttack.norm(delta) > eps).float()
            x = (orig + eps * delta / BanditAttack.norm(delta)) * out_of_bounds_mask
            x += new_x * (1 - out_of_bounds_mask)
            return x

        return proj

    @staticmethod
    def linf_proj(image, eps):
        orig = image.clone()

        def proj(new_x):
            return orig + ch.clamp(new_x - orig, -eps, eps)

        return proj

    @classmethod
    def make_adversarial_examples(cls, image, true_label, args, model_to_fool):
        '''
        The attack process for generating adversarial examples with priors.
        '''
        # Initial setup
        prior_size = IMAGE_SIZE[args.dataset][0] if not args.tiling else args.tile_size
        upsampler = Upsample(size=(IMAGE_SIZE[args.dataset][0], IMAGE_SIZE[args.dataset][1]))
        total_queries = ch.zeros(args.batch_size).cuda()
        prior = ch.zeros(args.batch_size, IN_CHANNELS[args.dataset], prior_size, prior_size).cuda()
        dim = prior.nelement() / args.batch_size  # nelement() --> total number of elements
        prior_step = BanditAttack.gd_prior_step if args.mode == 'l2' else BanditAttack.eg_step
        image_step = BanditAttack.l2_image_step if args.mode == 'l2' else BanditAttack.linf_step
        proj_maker = BanditAttack.l2_proj if args.mode == 'l2' else BanditAttack.linf_proj  # 调用proj_maker返回的是一个函数
        proj_step = proj_maker(image, args.epsilon)
        # Loss function
        criterion = ch.nn.CrossEntropyLoss(reduction='none')

        # Original classifications
        orig_images = image.clone()
        orig_classes = model_to_fool(image).argmax(1).cuda()
        correct_classified_mask = (orig_classes == true_label).float()
        not_dones_mask = correct_classified_mask.clone()  # 分类分对的mask
        t = 0
        normalized_q1 = []
        normalized_q2 = []
        images = []
        priors = []
        logits_q1_list = []
        logits_q2_list = []

        while not ch.any(total_queries > args.max_queries):  # 如果所有的为True,则整体是True,只要有一项是False,整体是False
            t += args.gradient_iters * 2
            if t >= args.max_queries:
                break
            if not args.nes:
                ## Updating the prior:
                # Create noise for exporation, estimate the gradient, and take a PGD step
                exp_noise = args.exploration * ch.randn_like(prior) / (dim ** 0.5)  # parameterizes the exploration to be done around the prior
                exp_noise = exp_noise.cuda()
                # Query deltas for finite difference estimator
                q1 = upsampler(prior + exp_noise)  # 这就是Finite Difference算法， prior相当于论文里的v，这个prior也会更新，把梯度累积上去
                q2 = upsampler(prior - exp_noise)   # prior 相当于累积的更新量，用这个更新量，再去修改image，就会变得非常准
                # Loss points for finite difference estimator
                images.append(image.detach().cpu().numpy())
                logits_q1 = model_to_fool(image + args.fd_eta * q1 / BanditAttack.norm(q1))
                logits_q2 = model_to_fool(image + args.fd_eta * q2 / BanditAttack.norm(q2))
                l1 = criterion(logits_q1, true_label)
                l2 = criterion(logits_q2, true_label)
                normalized_q1.append((args.fd_eta * q1 / BanditAttack.norm(q1)).detach().cpu().numpy())
                normalized_q2.append((args.fd_eta * q2 / BanditAttack.norm(q2)).detach().cpu().numpy())
                priors.append(prior.detach().cpu().numpy())  # 第一个prior是全0,没用
                logits_q1_list.append(logits_q1.detach().cpu().numpy())
                logits_q2_list.append(logits_q2.detach().cpu().numpy())
                # Finite differences estimate of directional derivative
                est_deriv = (l1 - l2) / (args.fd_eta * args.exploration)  # 方向导数 , l1和l2是loss
                # 2-query gradient estimate
                est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise  # B, C, H, W,
                # Update the prior with the estimated gradient
                prior = prior_step(prior, est_grad, args.online_lr)  # 注意，修正的是prior,这就是bandit算法的精髓
            else:  # NES方法
                prior = ch.zeros_like(image).cuda()
                for grad_iter_t in range(args.gradient_iters):
                    exp_noise = ch.randn_like(image) / (dim ** 0.5)
                    images.append(image.detach().cpu().numpy())
                    logits_q1 = model_to_fool(image + args.fd_eta * exp_noise)
                    logits_q2 = model_to_fool(image - args.fd_eta * exp_noise)
                    l1 = criterion(logits_q1, true_label)
                    l2 = criterion(logits_q2, true_label)
                    est_deriv = (l1-l2) / args.fd_eta
                    normalized_q1.append((args.fd_eta * exp_noise).detach().cpu().numpy())
                    normalized_q2.append((-args.fd_eta * exp_noise).detach().cpu().numpy())
                    priors.append(prior.detach().cpu().numpy())
                    logits_q1_list.append(logits_q1.detach().cpu().numpy())
                    logits_q2_list.append(logits_q2.detach().cpu().numpy())
                    prior += est_deriv.view(-1, 1, 1, 1) * exp_noise

                # Preserve images that are already done,
                # Unless we are specifically measuring gradient estimation
                prior = prior * not_dones_mask.view(-1, 1, 1, 1).cuda()

            ## Update the image:
            # take a pgd step using the prior
            new_im = image_step(image, upsampler(prior * correct_classified_mask.view(-1, 1, 1, 1)), args.image_lr)  # prior放大后相当于累积的更新量，可以用来更新
            image = proj_step(new_im)
            image = ch.clamp(image, 0, 1)

            ## Continue query count
            total_queries += 2 * args.gradient_iters * not_dones_mask  # gradient_iters是一个int值
            not_dones_mask = not_dones_mask * ((model_to_fool(image).argmax(1) == true_label).float())

            ## Logging stuff
            success_mask = (1 - not_dones_mask) * correct_classified_mask
            num_success = success_mask.sum()
            current_success_rate = (num_success / correct_classified_mask.detach().cpu().sum()).cpu().item()
            success_queries = ((success_mask * total_queries).sum() / num_success).cpu().item()
            max_curr_queries = total_queries.max().cpu().item()
            if args.log_progress:
                print("Queries: %d | Success rate: %f | Average queries: %f" % (
                max_curr_queries, current_success_rate, success_queries))

        normalized_q1 = np.ascontiguousarray(np.transpose(np.stack(normalized_q1), axes=(1,0,2,3,4)))
        normalized_q2 = np.ascontiguousarray(np.transpose(np.stack(normalized_q2), axes=(1,0,2,3,4)))
        images = np.ascontiguousarray(np.transpose(np.stack(images), axes=(1,0,2,3,4)))
        priors = np.ascontiguousarray(np.transpose(np.stack(priors), axes=(1,0,2,3,4)))  # B,T,C,H,W
        logits_q1_list = np.ascontiguousarray(np.transpose(np.stack(logits_q1_list),axes=(1,0,2)))  # B,T,#class
        logits_q2_list = np.ascontiguousarray(np.transpose(np.stack(logits_q2_list),axes=(1,0,2)))  # B,T,#class

        return {
            'average_queries': success_queries,
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
            'success_rate': current_success_rate,
            'images_orig': orig_images.cpu().numpy(),
            'images_adv': image.cpu().numpy(),
            'all_queries': total_queries.cpu().numpy(),
            'correctly_classified': correct_classified_mask.cpu().numpy(),
            'success': success_mask.cpu().numpy(),

            "q1":normalized_q1,
            "q2":normalized_q2,
            "images": images,
            "priors": priors,
            "logits_q1": logits_q1_list,
            "logits_q2": logits_q2_list
        }

    @classmethod
    def attack(cls, args, model_to_fool, dataset_loader, save_path_prefix):
        total_correct, total_adv, total_queries = 0, 0, 0
        normalized_q1_list = []
        normalized_q2_list = []
        images_list = []
        priors_list = []
        logits_q1_list = []
        logits_q2_list = []
        all_targets = []
        for i, (images, targets) in enumerate(dataset_loader):
            if i * args.batch_size >= args.total_images: # 导致文件过大，需要减小
                break
            res = BanditAttack.make_adversarial_examples(images.cuda(), targets.cuda(), args, model_to_fool)
            normalized_q1 = res["q1"]
            normalized_q2 = res["q2"]
            images = res["images"]
            priors = res["priors"]
            logits_q1 =  res["logits_q1"]
            logits_q2 =  res["logits_q2"]
            normalized_q1_list.append(normalized_q1)
            normalized_q2_list.append(normalized_q2)
            images_list.append(images)
            priors_list.append(priors)
            logits_q1_list.append(logits_q1)
            logits_q2_list.append(logits_q2)
            all_targets.extend(targets.detach().cpu().numpy().tolist())

            ncc = res['num_correctly_classified']  # Number of correctly classified images (originally)
            num_adv = ncc * res['success_rate']  # Success rate was calculated as (# adv)/(# correct classified)
            queries = num_adv * res[
                'average_queries']  # Average queries was calculated as (total queries for advs)/(# advs)
            total_correct += ncc
            total_adv += num_adv
            total_queries += queries

        print("-" * 80)
        if total_adv > 0 and total_correct > 0:
            print("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
                aq=total_queries / total_adv,
                succ=total_adv / total_correct))
        else:
            print("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
                aq=0,
                succ=0))
        print("-" * 80)
        normalized_q1_list = np.concatenate(normalized_q1_list,0)  # N,T,C,H,W
        normalized_q2_list = np.concatenate(normalized_q2_list,0)  # N,T,C,H,W
        images_list = np.concatenate(images_list,0)  # N,T,C,H,W
        priors_list = np.concatenate(priors_list,0)  # N,T,C,H,W
        logits_q1_list = np.concatenate(logits_q1_list,0)   # N,T,class
        logits_q2_list = np.concatenate(logits_q2_list,0)   # N,T,class
        store_shape = str(images_list.shape)
        gt_labels = np.array(all_targets)

        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        q1_path  = "{}@q1.npy".format(save_path_prefix)
        q2_path  = "{}@q2.npy".format(save_path_prefix)
        img_path = "{}@images.npy".format(save_path_prefix)
        prior_path = "{}@priors.npy".format(save_path_prefix)
        logits_q1_path = "{}@logits_q1.npy".format(save_path_prefix)
        logits_q2_path = "{}@logits_q2.npy".format(save_path_prefix)
        gt_labels_path = "{}@gt_labels.npy".format(save_path_prefix)
        count_path = "{}@shape.txt".format(save_path_prefix)
        fp = np.memmap(q1_path, dtype='float32', mode='w+', shape=normalized_q1_list.shape)
        fp[:, :, :, :, :] = normalized_q1_list[:, :, :, :, :]
        del fp
        del normalized_q1_list
        fp = np.memmap(q2_path, dtype='float32', mode='w+', shape=normalized_q2_list.shape)
        fp[:, :, :, :, :] = normalized_q2_list[:, :, :, :, :]
        del fp
        del normalized_q2_list
        fp = np.memmap(img_path, dtype='float32', mode='w+', shape=images_list.shape)
        fp[:, :, :, :, :] = images_list[:, :, :, :, :]
        del fp
        del images_list
        fp = np.memmap(prior_path, dtype='float32', mode='w+', shape=priors_list.shape)
        fp[:, :, :, :, :] = priors_list[:, :, :, :, :]
        del fp
        del priors_list
        fp = np.memmap(logits_q1_path, dtype='float32', mode='w+', shape=logits_q1_list.shape)
        fp[:, :, :] = logits_q1_list[:, :, :]
        del fp
        del logits_q1_list
        fp = np.memmap(logits_q2_path, dtype='float32', mode='w+', shape=logits_q2_list.shape)
        fp[:, :, :] = logits_q2_list[:, :, :]
        del fp
        del logits_q2_list
        fp = np.memmap(gt_labels_path, dtype="float32", mode="w+", shape=gt_labels.shape)
        fp[:] = gt_labels[:]
        del fp
        del gt_labels
        with open(count_path, "w") as file_count:
            file_count.write(store_shape)
            file_count.flush()
        print("write {} done".format(save_path_prefix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--json-config-file', type=str, help='a config file to be passed in instead of arguments')
    parser.add_argument("--dataset", type=str, choices=["CIFAR-10","CIFAR-100","MNIST","FashionMNIST","TinyImageNet","ImageNet"])
    parser.add_argument('--log-progress', action='store_true')
    parser.add_argument("--batch-size", type=int,default=500)
    parser.add_argument("--total_images",type=int,default=10000)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(args.gpu)
    print("using GPU {}".format(args.gpu))

    # If a json file is given, use the JSON file as the base, and then update it with args
    with open(args.json_config_file, "r") as file_obj:
        attack_json = json.load(file_obj)
    attack_type_params = []

    for attack_type, attack_conf in attack_json.items():
        if "nes" in attack_type:
            continue
        if attack_type!='linf':
            continue
        attack_conf.update(vars(args))
        params = SimpleNamespace(**attack_conf)
        attack_type_params.append((attack_type, params))
    trn_data_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, True)

    model_dir_path = "{}/train_pytorch_model/real_image_model/{}*.pth.tar".format(PY_ROOT, args.dataset)
    all_model_path_list = glob.glob(model_dir_path)
    model_names = dict()
    pattern = re.compile(".*{}@(.*?)@.*".format(args.dataset))
    for model_path in all_model_path_list:
        ma = pattern.match(os.path.basename(model_path))
        arch = ma.group(1)
        model_names[arch] = model_path


    models = []
    if args.dataset == "TinyImageNet":
        for arch, model_path in model_names.items():
            model = ModelConstructor.construct_tiny_imagenet_model(arch, args.dataset)
            model.load_state_dict(
                torch.load(model_path, map_location=lambda storage, location: storage)["state_dict"])
            model.eval()
            model.cuda()
            models.append({"arch_name":arch, "model":model})
    elif args.dataset == "ImageNet":
        for arch, model_path in model_names.items():
            model = ModelConstructor.construct_imagenet_model(arch)
            model.eval()
            models.append({"arch_name": arch, "model": model})
    else:
        for arch, model_path in model_names.items():
            model = ModelConstructor.construct_cifar_model(arch, args.dataset)
            model.load_state_dict(
                torch.load(model_path, map_location=lambda storage, location: storage)["state_dict"])
            model.eval()
            model.cuda()
            models.append({"arch_name":arch, "model":model})

    with ch.no_grad():
        # 双重循环，不同的arch,不同的attack_type :NES, linf等的组合，每个组合中一个label的数据出一个文件
        for attack_type, args_item in attack_type_params:
            for model_info in models:
                model_to_fool = model_info['model']
                arch_name = model_info["arch_name"]
                save_path_prefix = "{}/data_bandit_attack/{}/dataset_{}@attack_{}@arch_{}".format(PY_ROOT, args.dataset,
                                                                                                  args.dataset,
                                                                                                  attack_type,
                                                                                                  arch_name)
                if len(glob.glob(save_path_prefix + "*")) > 0:
                    print("skip {}".format(save_path_prefix))
                    continue
                model_to_fool.cuda()
                BanditAttack.attack(args_item, model_to_fool, trn_data_loader, save_path_prefix)
                model_to_fool.cpu()
