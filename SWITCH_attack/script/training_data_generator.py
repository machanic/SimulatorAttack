import os
import sys
import random
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
from types import SimpleNamespace
from collections import defaultdict
import glog as log
import json
import numpy as np
import torch
from torch.nn import functional as F
import os.path as osp
from config import PY_ROOT, CLASS_NUM, MODELS_TRAIN_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel

class PGD_attack(object):
    def __init__(self, dataset, targeted, target_type, epsilon, norm, lower_bound=0.0, upper_bound=1.0,
                 max_queries=10000):
        """
            :param targeted: whether is a targeted attack
            :param target_type: increment/random
            :param epsilon: perturbation limit according to lp-ball
            :param norm: norm for the lp-ball constraint
            :param lower_bound: minimum value data point can take in any coordinate
            :param upper_bound: maximum value data point can take in any coordinate
            :param max_queries: max number of calls to model per data point
        """
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        self.dataset = dataset
        self.epsilon = epsilon
        self.norm = norm
        self.max_queries = max_queries
        self.targeted = targeted
        self.target_type = target_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def normalize(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def l2_image_step(self, x, g, lr):
        return x + lr * g / self.normalize(g)

    def linf_image_step(self, x, g, lr):
        return x + lr * torch.sign(g)

    def linf_proj_step(self, image, epsilon, adv_image):
        return image + torch.clamp(adv_image - image, -epsilon, epsilon)

    def l2_proj_step(self, image, epsilon, adv_image):
        delta = adv_image - image
        out_of_bounds_mask = (self.normalize(delta) > epsilon).float()
        return out_of_bounds_mask * (image + epsilon * delta / self.normalize(delta)) + (1 - out_of_bounds_mask) * adv_image

    def xent_loss(self, logit, label, target=None):
        if target is not None:
            return -F.cross_entropy(logit, target, reduction='none')
        else:
            return F.cross_entropy(logit, label, reduction='none')

    def cw_loss(self, logit, label, target=None):
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), target]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    def get_grad(self, model, loss_fn, x, true_labels, target_labels):
        with torch.enable_grad():
            x.requires_grad_()
            logits = model(x)
            loss = loss_fn(logits, true_labels, target_labels)
            gradient = torch.autograd.grad(loss, x, torch.ones_like(loss), retain_graph=False)[0].detach()
        return gradient, logits, loss

    def attack_images(self, model, images, true_labels, target_labels, args):
        image_step = self.l2_image_step if args.norm == 'l2' else self.linf_image_step
        proj_step = self.l2_proj_step if args.norm == 'l2' else self.linf_proj_step
        criterion = self.cw_loss if args.loss == "cw" else self.xent_loss
        adv_images = images.clone()
        gradient_list = []
        logits_list = []
        loss_list = []
        image_list = []
        slice_len = 20
        slice_iteration_end = random.randint(slice_len, args.max_queries)
        for step_index in range(slice_iteration_end):
            gradient, logits, loss = self.get_grad(model, criterion, adv_images, true_labels, target_labels)
            if step_index >= slice_iteration_end - slice_len:
                image_list.append(adv_images.detach().cpu().numpy())
                gradient_list.append(gradient.detach().cpu().numpy())
                logits_list.append(logits.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
            adv_images = image_step(adv_images, gradient, args.image_lr)
            adv_images = proj_step(images, args.epsilon, adv_images)
            adv_images = torch.clamp(adv_images, 0, 1).detach()
        image_list = np.ascontiguousarray(np.transpose(np.stack(image_list), (1,0,2,3,4)))  # T,B,C,H,W -> B,T,C,H,W
        gradient_list = np.ascontiguousarray(np.transpose(np.stack(gradient_list), (1,0,2,3,4)))  # T,B,C,H,W -> B,T,C,H,W
        logits_list = np.ascontiguousarray(np.transpose(np.stack(logits_list), (1,0,2)))  # T,B,#class -> B,T,#class
        loss_list = np.ascontiguousarray(np.transpose(np.stack(loss_list), (1,0))) # T, B -> B,T
        return image_list, gradient_list, logits_list, loss_list


    def attack_all_images(self, args, model_data_dict, save_dir):

        for (arch_name, target_model), image_label_list in model_data_dict.items():
            all_image_list = []
            all_gradient_list = []
            all_logits_list = []
            all_target_label_list = []
            all_true_label_list = []
            all_loss_list = []
            target_model.cuda()
            targeted_str = "untargeted" if not args.targeted else "targeted"
            save_path_prefix = "{}/dataset_{}@arch_{}@norm_{}@loss_{}@{}".format(save_dir, args.dataset,
                                                                                 arch_name, args.norm, args.loss,
                                                                                 targeted_str)
            images_path = "{}@images.npy".format(save_path_prefix)
            gradients_path = "{}@gradients.npy".format(save_path_prefix)
            logits_loss_path = "{}@logits_loss".format(save_path_prefix)

            shape_path = "{}@shape.txt".format(save_path_prefix)
            log.info("Begin attack {}, the images will be saved to {}".format(arch_name, images_path))
            for batch_idx, (images, true_labels) in enumerate(image_label_list):
                images = images.cuda()
                true_labels = true_labels.cuda()
                if self.targeted:
                    if self.target_type == 'random':
                        target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                      size=true_labels.size()).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                        while invalid_target_index.sum().item() > 0:
                            target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                      size=target_labels[invalid_target_index].shape).long().cuda()
                            invalid_target_index = target_labels.eq(true_labels)
                    elif args.target_type == 'least_likely':
                        logits = target_model(images)
                        target_labels = logits.argmin(dim=1)
                    elif args.target_type == "increment":
                        target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                    else:
                        raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
                    all_target_label_list.extend(np.tile(target_labels.detach().cpu().numpy().reshape(-1,1), (1,20))) # each is B, T
                else:
                    target_labels = None
                all_true_label_list.extend(np.tile(true_labels.detach().cpu().numpy().reshape(-1,1), (1,20))) # each is B, T
                saved_images, saved_gradients, saved_logits, saved_losses = self.attack_images(target_model, images, true_labels, target_labels, args)

                all_image_list.extend(saved_images)  # B,T,C,H,W
                all_gradient_list.extend(saved_gradients)  # B,T,C,H,W
                all_logits_list.extend(saved_logits) # B,T, #class
                all_loss_list.extend(saved_losses)  # B, T

            all_image_list = np.stack(all_image_list)  # B,T,C,H,W
            all_gradient_list = np.stack(all_gradient_list)
            all_logits_list = np.stack(all_logits_list)
            all_loss_list = np.stack(all_loss_list)
            if all_target_label_list:
                all_target_label_list = np.stack(all_target_label_list)
            all_true_label_list = np.stack(all_true_label_list)
            store_shape = str(all_image_list.shape)
            with open(shape_path, "w") as file_shape:
                file_shape.write(store_shape)
                file_shape.flush()
            fp = np.memmap(images_path, dtype='float32', mode='w+', shape=all_image_list.shape)
            fp[:, :, :, :, :] = all_image_list[:, :, :, :, :]
            del fp
            fp= np.memmap(gradients_path, dtype='float32',mode='w+', shape=all_gradient_list.shape)
            fp[:, :, :, :, :] = all_gradient_list[:, :, :, :, :]
            del fp
            if args.targeted:
                np.savez(logits_loss_path, logits=all_logits_list, loss=all_loss_list, true_labels=all_true_label_list,
                         target_labels=all_target_label_list)
            else:
                np.savez(logits_loss_path, logits=all_logits_list, loss=all_loss_list, true_labels=all_true_label_list)
            log.info('{} is attacked finished, save to {}'.format(arch_name, images_path))
            target_model.cpu()


def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def get_log_path(dataset, loss, norm, targeted, target_type):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    dirname = 'generate_data-{}-{}loss-{}-{}.log'.format(dataset, loss, norm, target_str)
    return dirname


def main():
    parser = argparse.ArgumentParser(description='Square Attack Hyperparameters.')
    parser.add_argument('--norm', type=str, required=True, choices=['l2', 'linf'])
    parser.add_argument('--image_lr',type=float,default=None)
    parser.add_argument('--dataset',type=str, required=True)
    parser.add_argument('--gpu', type=str,required=True, help='GPU number. Multiple GPUs are possible for PT models.')
    parser.add_argument('--epsilon', type=float,  help='Radius of the Lp ball.')
    parser.add_argument('--max_queries',type=int,default=100)
    parser.add_argument('--json-config', type=str,
                        default='/home1/machen/meta_perturbations_black_box_attack/configures/surrogate_gradient_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--batch_size',type=int,default=100)
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='random', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--loss', choices=["cw","xent"], required=True, type=str)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)

    if args.targeted and args.dataset == "ImageNet":
        args.max_queries = 10000

    save_dir_path = "{}/data_surroage_gradient/{}/{}".format(PY_ROOT, args.dataset,
                                                         "targeted_attack" if args.targeted else "untargeted_attack")
    os.makedirs(save_dir_path, exist_ok=True)
    log_path = osp.join(save_dir_path,
                        get_log_path(args.dataset, args.loss, args.norm, args.targeted, args.target_type))

    set_log_file(log_path)

    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_path))
    log.info('Called with args:')
    print_args(args)
    trn_data_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, is_train=True)
    models = []
    for arch in MODELS_TRAIN_STANDARD[args.dataset]:
        if StandardModel.check_arch(arch, args.dataset):
            model = StandardModel(args.dataset, arch, False)
            model = model.eval()
            models.append({"arch_name": arch, "model": model})
    model_data_dict = defaultdict(list)
    for idx, (images, labels) in enumerate(trn_data_loader):
        model_info = random.choice(models)
        arch = model_info["arch_name"]
        model = model_info["model"]
        if images.size(-1) != model.input_size[-1]:
            images = F.interpolate(images, size=model.input_size[-1], mode='bilinear', align_corners=True)
        model_data_dict[(arch, model)].append((images, labels))
        if args.dataset == "ImageNet" and idx >= 200:
            break

    log.info("Assign data to multiple models over!")
    attacker = PGD_attack(args.dataset, args.targeted, args.target_type, args.epsilon, args.norm, max_queries=args.max_queries)
    attacker.attack_all_images(args, model_data_dict, save_dir_path)
    log.info("All done!")

if __name__ == "__main__":
    main()
