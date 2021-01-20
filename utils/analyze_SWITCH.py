import argparse
import json
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import glob
import numpy as np
import os

from config import PY_ROOT


def read_json(file_path):
    with open(file_path, "r") as file_obj:
        content = json.load(file_obj)
        if "rnd_save_stats" in file_path:
            improve_loss_after_switch_record = content["increase_loss_from_last_iter_after_switch_record"]
            improve_loss_record = content["increase_loss_from_last_iter_with_1st_model_grad_record"]
            loss_x_temp_record = content["loss_x_pos_temp_record"]
            loss_after_switch_grad_record = content["loss_after_switch_grad_record"]
        else:
            improve_loss_after_switch_record = content["increase_loss_from_last_iter_after_switch_record"]
            improve_loss_record = content["increase_loss_from_last_iter_with_pos_grad_record"]
            loss_x_temp_record = content["loss_x_pos_temp_record"]
            loss_after_switch_grad_record = content["loss_after_switch_grad_record"]
    return loss_x_temp_record, loss_after_switch_grad_record, improve_loss_record, improve_loss_after_switch_record, content

def merge_json(dict_content):
    all_list = []
    for k, value_list in sorted(dict_content.items(), key=lambda e:int(e[0])):
        all_list.extend(value_list)
    return np.array(all_list)

# 1，所有人里有多少比例切换了方向
def count_ratio_switch(improve_loss_record):
    switch_array = 1.0 - np.array(improve_loss_record).astype(np.float32)
    return np.mean(switch_array).item()

# 切换的人里有多少比例loss上升了
def count_ratio_improved_after_switch(improve_loss_record, improve_loss_after_switch_record):
    switch_array = 1.0 - np.array(improve_loss_record).astype(np.float32)
    switch_array = switch_array.astype(np.bool)
    return np.mean(improve_loss_after_switch_record[switch_array]).item()

#  switch之后的loss比switch前的loss大的比例
def count_ratio_swtiched_loss_bigger_than_orig(improve_loss_record, loss_x_temp_record, loss_after_switch_grad_record):
    switch_array = 1.0 - np.array(improve_loss_record).astype(np.float32)
    switch_array = switch_array.astype(np.bool)
    loss_x_temp_record = loss_x_temp_record[switch_array]
    loss_after_switch_grad_record = loss_after_switch_grad_record[switch_array]
    bigger = (loss_after_switch_grad_record > loss_x_temp_record).astype(np.float32)
    return np.mean(bigger).item()

# 切换的人里面有多少loss下降了，但下降的没原来多
def count_ratio_loss_decreased_no_more_than_orig_after_switch(improve_loss_record, improve_loss_after_switch_record, loss_x_temp_record, loss_after_switch_grad_record):
    switch_array = 1.0 - np.array(improve_loss_record).astype(np.float32)
    switch_array = switch_array.astype(np.bool)
    decrease_loss_after_switch = 1.0 - improve_loss_after_switch_record
    decrease_loss_after_switch = decrease_loss_after_switch.astype(np.bool)
    assert len(switch_array) == len(decrease_loss_after_switch) == len(loss_x_temp_record) == len(loss_after_switch_grad_record)
    loss_x_temp_record = loss_x_temp_record[switch_array & decrease_loss_after_switch]
    loss_after_switch_grad_record = loss_after_switch_grad_record[switch_array & decrease_loss_after_switch]
    desc = (loss_after_switch_grad_record > loss_x_temp_record).astype(np.float32)
    return np.mean(desc).item()


def get_SWITCH_rnd_dir_name(dataset,  norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    surrogate_models = "resnet-110,densenet-bc-100-12" if "CIFAR" in dataset else "resnet101,resnet152"
    loss = 'cw' if not targeted else "xent"

    if args.attack_defense:
        dirname = 'SWITCH_rnd_save_stats_attack_on_defensive_model_using_{}-{}-{}_loss-{}-{}'.format(surrogate_models, dataset, loss, norm, target_str)
    else:
        # SWITCH_rnd_save_stats_attack_using_resnet-110,densenet-bc-100-12-CIFAR-10-cw_loss-linf-untargeted
        dirname = 'SWITCH_rnd_save_stats_attack_using_{}-{}-{}_loss-{}-{}'.format(surrogate_models, dataset, loss, norm, target_str)
    return dirname

def get_SWITCH_neg_dir_name(dataset,  norm, targeted, target_type, args):
    # SWITCH_neg_save-CIFAR-10-xent_lr_0.003-loss-linf-targeted_increment
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    loss = 'cw' if not targeted else "xent"

    if args.attack_defense:
        dirname = 'SWITCH_neg_save_stats_attack_on_defensive_model-{}-{}_loss-{}-{}'.format(dataset, loss, norm, target_str)
    else:
        # SWITCH_neg_save_stats_attack-CIFAR-10-xent_loss-linf-targeted_increment
        dirname = 'SWITCH_neg_save_stats_attack-{}-{}_loss-{}-{}'.format(dataset, loss, norm, target_str)
    return dirname

def check_lr(dataset, norm, targeted, content):
    if "CIFAR" in dataset:
        if norm == "l2":
            lr = 0.1
        else:
            if targeted:
                lr = 0.003
            else:
                lr = 0.01
    elif dataset == "TinyImageNet":
        if norm == "l2":
            lr = 0.2
        else:  # linf
            lr = 0.003
    read_image_lr = content['args']['image_lr']
    assert lr == read_image_lr, "read json image-lr is {}".format(read_image_lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument("--type", type=str, required=True, choices=["neg","rnd"])
    args = parser.parse_args()
    if args.type == "neg":
        folder_path = get_SWITCH_neg_dir_name(args.dataset,  args.norm, args.targeted, args.target_type, args)
    elif args.type == "rnd":
        folder_path = get_SWITCH_rnd_dir_name(args.dataset,  args.norm, args.targeted, args.target_type, args)
    root = os.getcwd()
    folder_path = "{}/logs/{}".format(root, folder_path)
    print(folder_path)
    for json_file_path in glob.glob(folder_path + "/*.json"):
        if "stats" in os.path.basename(json_file_path):
            continue
        print("Reading {}".format(json_file_path))
        loss_x_temp_record, loss_after_switch_grad_record, improve_loss_record, improve_loss_after_switch_record, content = read_json(json_file_path)
        check_lr(args.dataset, args.norm, args.targeted, content)
        loss_x_temp_record = merge_json(loss_x_temp_record)
        loss_after_switch_grad_record = merge_json(loss_after_switch_grad_record)
        improve_loss_record = merge_json(improve_loss_record)
        improve_loss_after_switch_record = merge_json(improve_loss_after_switch_record)
        switch_ratio = count_ratio_switch(improve_loss_record)
        ratio_improved_after_switch = count_ratio_improved_after_switch(improve_loss_record, improve_loss_after_switch_record)
        ratio_loss_decreased_no_more_than_orig_after_switch = count_ratio_loss_decreased_no_more_than_orig_after_switch(improve_loss_record, improve_loss_after_switch_record, loss_x_temp_record, loss_after_switch_grad_record)
        ratio_swtiched_loss_bigger_than_orig = count_ratio_swtiched_loss_bigger_than_orig(improve_loss_record, loss_x_temp_record, loss_after_switch_grad_record)
        result_path = json_file_path[:json_file_path.rindex(".")] + "_stats.json"
        with open(result_path, "w") as file_obj:
            js = {"ratio_improved_after_switch":ratio_improved_after_switch,
                   "ratio_swtiched_loss_bigger_than_orig":ratio_swtiched_loss_bigger_than_orig,
                   "switch_ratio":switch_ratio}
                   # "ratio_loss_decreased_no_more_than_orig_after_switch":ratio_loss_decreased_no_more_than_orig_after_switch}
            json.dump(js, file_obj, sort_keys=True)
        print("output done of {}".format(result_path))