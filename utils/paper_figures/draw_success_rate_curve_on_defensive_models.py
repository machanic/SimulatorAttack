import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import make_interp_spline
import seaborn as sns

from utils.statistics_toolkit import success_rate_avg_query, success_rate_and_query_coorelation


def read_json_data(json_path, data_key):
    # data_key can be query_success_rate_dict, query_threshold_success_rate_dict, success_rate_to_avg_query
    x = []
    y = []
    print("begin read {}".format(json_path))
    with open(json_path, "r") as file_obj:
        data_txt = file_obj.read().replace('"not_done_loss": NaN, "not_done_prob": NaN,', "")
        data_json = json.loads(data_txt)
        query_all = np.array(data_json["query_all"])
        not_done_all = np.array(data_json["not_done_all"])
        correct_all = np.array(data_json["correct_all"])
        if data_key == "success_rate_to_avg_query":

            data = success_rate_avg_query(query_all, not_done_all, correct_all)
        else:
            data, _ = success_rate_and_query_coorelation(query_all, not_done_all, correct_all)
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
            resnet_50_name = "resnet-50" if dataset != "TinyImageNet" else "resnet50"
            if arch in file_path and file_path.endswith(".json") and (not file_path.startswith("tmp")) and resnet_50_name in file_path \
                    and "adv_train" not in file_path:  # FIXME
                file_path = dir_path + "/" + file_path
                x, y = read_json_data(file_path, data_key)
                data_info[(dataset, norm, targeted, method)] = (x,y)
                break
    return data_info

method_name_to_paper = {"bandits_attack":"Bandits", "NES-attack":"NES", "P-RGF_biased_attack":"P-RGF","P-RGF_uniform_attack":"RGF",
                        # "ZOO_randomly_sample":"ZOO", "ZOO_importance_sample":"ZOO(I)",
                        "MetaGradAttack":"Meta Attack",
                        "simulate_bandits_shrink_attack":"MetaSimulator"}

def from_method_to_dir_path(dataset, method, norm, targeted):
    # methods = ["bandits_attack", "MetaGradAttack", "NES-attack","P-RGF_biased","P-RGF_uniform","ZOO","simulate_bandits_shrink_attack"]
    if method == "bandits_attack":
        path = "{method}_on_defensive_model-{dataset}-cw_loss-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                        norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NES-attack":
        path = "{method}_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                norm=norm,
                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "P-RGF_biased_attack":
        # P-RGF_biased_attack_CIFAR-100_surrogate_arch_resnet-110_l2_targeted_increment
        surrogate_arch = "resnet-110" if dataset.startswith("CIFAR") else "resnet101"
        path = "{method}_on_defensive_model_{dataset}_surrogate_arch_{surrogate_arch}_{norm}_{target_str}".format(method=method, dataset=dataset, surrogate_arch=surrogate_arch,
                                                                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "P-RGF_uniform_attack":
        surrogate_arch = "resnet-110" if dataset.startswith("CIFAR") else "resnet101"
        path = "{method}_on_defensive_model_{dataset}_surrogate_arch_{surrogate_arch}_{norm}_{target_str}".format(method=method,
                                                                                               dataset=dataset,
                                                                                               surrogate_arch=surrogate_arch,
                                                                                               norm=norm,
                                                                                               target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "MetaGradAttack":
        use_tanh = "tanh" if norm == "l2" else "no_tanh"
        path = "MetaGradAttack_on_defensive_model_{dataset}_{norm}_{target_str}_{tanh}".format(dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment", tanh=use_tanh)
    elif method == "simulate_bandits_shrink_attack":
        path = "simulate_bandits_shrink_attack_on_defensive_model-{dataset}-cw_loss-{norm}-{target_str}-mse".format(dataset=dataset,norm=norm,target_str="untargeted" if not targeted else "targeted_increment")
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
    if targeted:
        methods = list(filter(lambda method_name:"RGF" not in method_name, methods))
    dataset_path_dict= get_all_exists_folder(dataset, methods, norm, targeted)
    data_info = read_all_data(dataset_path_dict, arch, fig_type)  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): ([x],[y])， }
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g',  'c', 'm', 'y', 'k', 'w']
    # markers = [".",",","o","^","s","p","x"]
    # max_x = 0
    # min_x = 0
    xtick = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    for idx, ((dataset, norm, targeted, method), (x,y)) in enumerate(data_info.items()):
        color = colors[idx%len(colors)]
        if method == "MetaSimulator":
            color = "r"
        # marker = markers[idx%len(markers)]
        # x_smooth = np.linspace(x.min(), x.max(), 300)
        # y_smooth = make_interp_spline(x, y)(x_smooth)
        max_x = np.max(x)
        if max_x < 10000:
            x = x.tolist()
            y = y.tolist()
            y_last = y[-1]
            for query_limit in range(0, 10001, 1000):
                if query_limit>0 and query_limit > max_x:
                    x.append(query_limit)
                    y.append(y_last)
        x = np.array(x)
        y = np.array(y)
        line, = plt.plot(x, y, label=method, color=color, linestyle="-")
        y_points = np.interp(xtick, x, y)
        plt.scatter(xtick, y_points,color=color,marker='.')
        # if np.max(x).item() > max_x:
        #     max_x = np.max(x).item()
        # if np.min(x).item() < min_x:
        #     min_x = np.min(x).item()
    plt.xlim(0, 10000)
    plt.ylim(0, 100)
    plt.gcf().subplots_adjust(bottom=0.15)

    # xtick = [0, 5000, 10000]
    plt.xticks(xtick, fontsize=15)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90,100], fontsize=15)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(loc='lower right', prop={'size': 18})
    plt.savefig(dump_file_path, dpi=200)

def draw_success_rate_avg_query_fig(dataset, norm, targeted, arch, fig_type, dump_file_path):
    xlabel = "Attack Success Rate (%)"
    ylabel = "Avg. Query"
    methods = list(method_name_to_paper.keys())
    if targeted:
        methods = list(filter(lambda method_name:"RGF" not in method_name, methods))
    # elif norm =='l2':
    #     methods = list(filter(lambda method_name: "MetaGradAttack" not in method_name, methods))

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
        line, = plt.plot(x, y, label=method, color=color, linestyle="-", marker='.')
        if np.max(x).item() > max_x:
            max_x = np.max(x).item()
        if np.min(x).item() < min_x:
            min_x = np.min(x).item()
    plt.xlim(0, 100)
    # if norm == "l2" or (dataset == "CIFAR-100" and norm == 'linf'):
    #     plt.ylim(0, 1000)
    # else:
    #     plt.ylim(0, 3000)
    # if dataset == "TinyImageNet":
    if targeted:
        plt.ylim(0,6000)
    else:
        plt.ylim(0, 2500)
    if dataset == "TinyImageNet":
        plt.ylim(0, 3000)

    plt.gcf().subplots_adjust(bottom=0.15)
    xtick = np.arange(0,101,5)
    # if norm == "l2" or (dataset == "CIFAR-100" and norm == 'linf'):
    #     ytick = [0,200,400,600,800,1000]
    # else:
    #     ytick = [0,500,1000,1500,2000,2500,3000]
    # if dataset == "TinyImageNet":
    #     ytick = np.arange(0,4001,200).tolist()
    # if targeted:
    #     ytick = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000,4500,5000,5500,6000] #,6500,7000,7500,8000,8500,9000,9500,10000]
    #
    # if dataset == "CIFAR-100" and (not targeted) and norm == "linf":
    #     ytick = np.arange(0, 1501, 100).tolist()
    # elif dataset == "TinyImageNet":
    #     ytick = np.arange(0, 5001, 200).tolist()
    # else:
    #     ytick = np.arange(0, 4001, 200).tolist()
    # if norm == "l2" and not targeted:
    #     ytick = np.arange(0, 251, 25).tolist()
    ytick = np.arange(0, 2501, 200).tolist()
    if dataset == "TinyImageNet":
        ytick = np.arange(0, 3001, 200).tolist()
    # xtick = [0, 5000, 10000]
    plt.xticks(xtick, fontsize=15)
    plt.yticks(ytick, fontsize=15)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(loc='upper left', prop={'size': 18})
    plt.savefig(dump_file_path, dpi=200)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Meta Model Training')
    parser.add_argument("--fig_type", type=str, choices=["query_threshold_success_rate_dict",
                                                         "success_rate_to_avg_query"])
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to train")
    parser.add_argument("--model",type=str, required=True, help="the defensive model names, e.g. pcl_loss, com_defend,feature_distillation,feature_scatter")
    parser.add_argument("--norm", type=str, choices=["l2", "linf"], required=True)
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dump_folder = "/home1/machen/meta_perturbations_black_box_attack/figures/{}_defensive_model/".format(args.fig_type)
    os.makedirs(dump_folder, exist_ok=True)
    file_path  = dump_folder + "{dataset}_{model}_{norm}_{target_str}_attack_{fig_type}.pdf".format(dataset=args.dataset,
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