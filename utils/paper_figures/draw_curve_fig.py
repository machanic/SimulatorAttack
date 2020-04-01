import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json


def read_json_data(json_path, data_key):
    # data_key can be query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query
    x = []
    y = []
    print("begin read {}".format(json_path))
    with open(json_path, "r") as file_obj:
        data_txt = file_obj.read().replace('"not_done_loss": NaN, "not_done_prob": NaN,', "")
        data_json = json.loads(data_txt)
        data = data_json[data_key]
        for key, value in sorted(data.items(), key=lambda e:int(e[0])):
            x.append(int(key))
            y.append(value)
    if data_key.endswith("success_rate_dict"):
        return np.array(x), np.array(y) * 100
    return np.array(x), np.array(y)

def read_all_data(dataset_path_dict, arch, data_key):
    # dataset_path_dict {("CIFAR-10","l2","untargeted"): "/.../"， }
    data_info = {}
    for (dataset, norm, targeted, method), dir_path in dataset_path_dict.items():
        for file_path in os.listdir(dir_path):
            if arch in file_path and file_path.endswith(".json") and not file_path.startswith("tmp"):
                file_path = dir_path + "/" + file_path
                x, y = read_json_data(file_path, data_key)
                data_info[(dataset, norm, targeted, method)] = (x,y)
                break
    return data_info

method_name_to_paper = {"bandits_attack":"Bandits", "NES-attack":"NES", "P-RGF_biased_attack":"P-RGF","P-RGF_uniform_attack":"RGF",
                        # "ZOO_randomly_sample":"ZOO", "ZOO_importance_sample":"ZOO(I)",
                        "MetaGradAttack":"meta attack",
                        "simulate_bandits_shrink_attack":"MetaSimulator"}

def from_method_to_dir_path(dataset, method, norm, targeted):
    # methods = ["bandits_attack", "MetaGradAttack", "NES-attack","P-RGF_biased","P-RGF_uniform","ZOO","simulate_bandits_shrink_attack"]
    if method == "bandits_attack":
        path = "{method}-{dataset}-cw_loss-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                        norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NES-attack":
        path = "{method}-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "P-RGF_biased_attack":
        # P-RGF_biased_attack_CIFAR-100_surrogate_arch_resnet-110_l2_targeted_increment
        surrogate_arch = "resnet-110" if dataset.startswith("CIFAR") else "resnet101"
        path = "{method}_{dataset}_surrogate_arch_{surrogate_arch}_{norm}_{target_str}".format(method=method, dataset=dataset, surrogate_arch=surrogate_arch,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "P-RGF_uniform_attack":
        surrogate_arch = "resnet-110" if dataset.startswith("CIFAR") else "resnet101"
        path = "{method}_{dataset}_surrogate_arch_{surrogate_arch}_{norm}_{target_str}".format(method=method,
                                                                                               dataset=dataset,
                                                                                               surrogate_arch=surrogate_arch,
                                                                                               norm=norm,
                                                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "ZOO_randomly_sample":
        path = "ZOO_randomly_sample-{dataset}-no_tanh-no_log-{target_str}".format(dataset=dataset, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "ZOO_importance_sample":
        path = "ZOO_importance_sample-{dataset}-no_tanh-no_log-{target_str}".format(dataset=dataset, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NES-attack":
        path = "NES-attack-{dataset}-{norm}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "MetaGradAttack":
        use_tanh = "tanh" if norm == "l2" else "no_tanh"
        path = "MetaGradAttack_{dataset}_{norm}_{target_str}_{tanh}".format(dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment", tanh=use_tanh)
    elif method == "simulate_bandits_shrink_attack":
        path = "simulate_bandits_shrink_attack-{dataset}-cw_loss-{norm}-{target_str}-mse".format(dataset=dataset,norm=norm,target_str="untargeted" if not targeted else "targeted_increment")
    return path

def get_all_exists_folder(dataset, methods, norm, targeted):
    root_dir = "/home1/machen/meta_perturbations_black_box_attack/logs/"
    dataset_path_dict = {}  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): "/.../"， }
    for method in methods:
        file_name = from_method_to_dir_path(dataset, method, norm, targeted)
        file_path = root_dir + file_name
        if os.path.exists(file_path):
            dataset_path_dict[(dataset, norm, targeted, method_name_to_paper[method])] = file_path
        else:
            print("{} does not exist!!!".format(file_path))
    return dataset_path_dict

def draw_query_success_rate_figure(dataset, norm, targeted, arch, fig_type, dump_file_path, xlabel, ylabel):

    # fig_type can be [query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query]
    methods = list(method_name_to_paper.keys())
    dataset_path_dict= get_all_exists_folder(dataset, methods, norm, targeted)
    data_info = read_all_data(dataset_path_dict, arch, fig_type)  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): ([x],[y])， }

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 8))

    colors = ['b', 'g',  'c', 'm', 'y', 'k', 'w']
    markers = [".",",","o","^","s","p","x"]
    max_x = 0
    min_x = 0
    for idx, ((dataset, norm, targeted, method), (x,y)) in enumerate(data_info.items()):
        color = colors[idx%len(colors)]
        if method == "MetaSimulator":
            color = "r"
        marker = markers[idx%len(markers)]
        line, = plt.plot(x, y, label=method, color=color, linestyle="-")
        if np.max(x).item() > max_x:
            max_x = np.max(x).item()
        if np.min(x).item() < min_x:
            min_x = np.min(x).item()
    plt.xlim(0, 10000)
    plt.ylim(0, 100)
    plt.gcf().subplots_adjust(bottom=0.15)
    xtick = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    # xtick = [0, 5000, 10000]
    plt.xticks(xtick, fontsize=15)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90,100], fontsize=15)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(loc='lower right', prop={'size': 13})
    plt.savefig(dump_file_path, dpi=200)

def draw_success_rate_avg_query_fig(dataset, norm, targeted, arch, fig_type, dump_file_path):
    xlabel = "Attack Success Rate (%)"
    ylabel = "Avg. Queries"
    methods = list(method_name_to_paper.keys())
    dataset_path_dict = get_all_exists_folder(dataset, methods, norm, targeted)
    data_info = read_all_data(dataset_path_dict, arch, fig_type)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 8))

    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'w']
    markers = [".", ",", "o", "^", "s", "p", "x"]
    max_x = 0
    min_x = 0
    for idx, ((dataset, norm, targeted, method), (x, y)) in enumerate(data_info.items()):
        color = colors[idx % len(colors)]
        if method == "MetaSimulator":
            color = "r"
        line, = plt.plot(x, y, label=method, color=color, linestyle="-")
        if np.max(x).item() > max_x:
            max_x = np.max(x).item()
        if np.min(x).item() < min_x:
            min_x = np.min(x).item()
    plt.xlim(0, 100)
    plt.ylim(0, 1000)
    plt.gcf().subplots_adjust(bottom=0.15)
    xtick = [0, 25, 50, 75, 100]
    ytick = [0,200,400,600,800,1000]
    # xtick = [0, 5000, 10000]
    plt.xticks(xtick, fontsize=15)
    plt.yticks(ytick, fontsize=15)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(loc='lower right', prop={'size': 13})
    plt.savefig(dump_file_path, dpi=200)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta Model Training')
    parser.add_argument("--fig_type", type=str, choices=["query_success_rate_dict", "query_threshold_success_rate_dict",
                                                         "success_rate_to_avg_query"])
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to train")
    parser.add_argument("--model",type=str, required=True)
    parser.add_argument("--norm", type=str, choices=["l2", "linf"], required=True)
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dump_folder = "/home1/machen/meta_perturbations_black_box_attack/figures/{}/".format(args.fig_type)
    os.makedirs(dump_folder, exist_ok=True)
    file_path  = dump_folder + "{dataset}_{model}_{norm}_{target_str}_attack_{fig_type}.png".format(dataset=args.dataset,
                                                                            model=args.model, norm=args.norm, target_str="untargeted" if not args.targeted else "targeted",
                                                                            fig_type=args.fig_type)
    if args.fig_type == "query_threshold_success_rate_dict":
        x_label = "Maximum Query Number Threshold"
        y_label = "Attack Success Rate (%)"
    elif args.fig_type == "query_success_rate_dict":
        x_label = "Query Number"
        y_label = "Attack Success Rate (%)"
    else:
        x_label = "Attack Success Rate (%)"
        y_label = "Query Numbers"
    if args.fig_type == "query_threshold_success_rate_dict":
        draw_query_success_rate_figure(args.dataset, args.norm, args.targeted, args.model, args.fig_type, file_path, x_label, y_label)
    elif args.fig_type == "success_rate_to_avg_query":
        draw_success_rate_avg_query_fig(args.dataset, args.norm, args.targeted, args.model, args.fig_type, file_path)
    print("written to {}".format(file_path))