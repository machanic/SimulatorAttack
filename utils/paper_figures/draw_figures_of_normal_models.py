import os
import sys
sys.path.append(os.getcwd())
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import make_interp_spline
import seaborn as sns
from utils.statistics_toolkit import success_rate_avg_query, success_rate_and_query_coorelation, query_to_bins

# from matplotlib import rcParams, rc
# rcParams['xtick.direction'] = 'out'
# rcParams['ytick.direction'] = 'out'
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rc('pdf', fonttype=42)
plt.switch_backend('agg')
plt.rcParams['pdf.use14corefonts'] = True
font = {'family': 'Helvetica'}
plt.rc('font', **font)


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
        success_rate = (1- data_json["avg_not_done"]) * 100
        if data_key == "success_rate_to_avg_query":
            data = success_rate_avg_query(query_all, not_done_all, correct_all, success_rate)
        else:
            data, _ = success_rate_and_query_coorelation(query_all, not_done_all, correct_all)
        for key, value in sorted(data.items(), key=lambda e:float(e[0])):
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


def get_success_queries(dataset_path_dict, arch):
    data_info = {}
    for (dataset, norm, targeted, method), dir_path in dataset_path_dict.items():
        for file_path in os.listdir(dir_path):
            if arch in file_path and file_path.endswith(".json") and not file_path.startswith("tmp"):
                file_path = dir_path + "/" + file_path
                with open(file_path, "r") as file_obj:
                    json_data = json.load(file_obj)
                    query_all = np.array(json_data["query_all"],dtype=np.int32)
                    not_done_all = np.array(json_data["not_done_all"], dtype=np.bool)
                    query_all = query_all[~not_done_all]
                    data_info[(dataset, norm, targeted, method)] = query_all
                break
    return data_info


method_name_to_paper = {"bandits_attack":"Bandits", "P-RGF_biased_attack":"P-RGF","P-RGF_uniform_attack":"RGF",
                        # "MetaGradAttack":"Meta Attack",
                        # "simulate_bandits_shrink_attack":"MetaSimulator",
                        "NO_SWITCH" : "NO SWITCH",
                        # "SWITCH_other":r'$\mathrm{SWITCH}_{other}$',
                        "SWITCH_neg":'SWITCH',
                        "SWITCH_RGF": 'SWITCH$_{{RGF}}$',
                        # "NO_SWITCH_rnd":  r'NO $\mathrm{SWITCH}_{rnd}$',
                        "PPBA_attack":"PPBA",
                        "parsimonious_attack":"Parsimonious",
                        "sign_hunter_attack":"SignHunter",
                        "square_attack":"Square Attack"}    #, "SimBA_DCT_attack":"SimBA"}

def from_method_to_dir_path(dataset, method, norm, targeted):
    if "SWITCH" in method:
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
    elif method == "MetaGradAttack":
        use_tanh = "tanh" if norm == "l2" else "no_tanh"
        path = "MetaGradAttack_{dataset}_{norm}_{target_str}_{tanh}".format(dataset=dataset,norm=norm, target_str="untargeted" if not targeted else "targeted_increment", tanh=use_tanh)
    elif method == "simulate_bandits_shrink_attack":
        path = "simulate_bandits_shrink_attack-{dataset}-cw_loss-{norm}-{target_str}-mse".format(dataset=dataset,norm=norm,target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "PPBA_attack":
        path = "PPBA_attack-{dataset}-{norm}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "parsimonious_attack":
        path = "parsimonious_attack-{norm}-{dataset}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SimBA_DCT_attack":
        path = "SimBA_DCT_attack-{dataset}-{norm}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "sign_hunter_attack":
        path = "sign_hunter_attack-{dataset}-{norm}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "square_attack":
        path = "square_attack-{dataset}-{norm}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")

    elif method == "SWITCH_neg":
        path = "SWITCH_neg_save-{dataset}-{loss}_lr_{lr}-loss-{norm}-{target_str}".format(dataset=dataset, lr=lr, loss="cw" if not targeted else "xent", norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SWITCH_other":
        path = "SWITCH_rnd_save-{dataset}-{loss}_lr_{lr}-loss-{norm}-{target_str}".format(dataset=dataset, lr=lr, loss="cw" if not targeted else "xent", norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NO_SWITCH":
        path = "NO_SWITCH-{dataset}-{loss}_loss-{norm}-{target_str}".format(dataset=dataset, loss="cw" if not targeted else "xent", norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NO_SWITCH_rnd":
        # NO_SWITCH_rnd_using_resnet-110,densenet-bc-100-12_on_defensive_model-CIFAR-100-lr_0.01_cw-loss-linf-untargeted
        path = "NO_SWITCH_rnd_using_{archs}-{dataset}-lr_{lr}_{loss}-loss-{norm}-{target_str}".format(
                                                                            archs="resnet-110,densenet-bc-100-12" if "CIFAR" in dataset else "resnet101,resnet152", dataset=dataset,
                                                                            lr=lr,
                                                                            loss="cw" if not targeted else "xent",
                                                                            norm=norm,
                                                                            target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SWITCH_RGF":
        if dataset.startswith("CIFAR"):
            path = "SWITCH_RGF-resnet-110-{dataset}-{loss}-loss-{norm}-{target_str}".format(dataset=dataset, loss="cw" if not targeted else "xent",
                                                                                            norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
        elif dataset == "TinyImageNet":
            path = "SWITCH_RGF-resnet101-{dataset}-{loss}-loss-{norm}-{target_str}".format(dataset=dataset,
                                                                                            loss="cw" if not targeted else "xent",
                                                                                            norm=norm,
                                                                                            target_str="untargeted" if not targeted else "targeted_increment")
    return path

def get_all_exists_folder(dataset, methods, norm, targeted):
    root_dir = "/home1/machen/query_based_black_box_attack/logs/"
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
    # if targeted:
    #     methods = list(filter(lambda method_name:"RGF" not in method_name, methods))
    dataset_path_dict= get_all_exists_folder(dataset, methods, norm, targeted)
    data_info = read_all_data(dataset_path_dict, arch, fig_type)  # dataset_path_dict {("CIFAR-10","l2","untargeted", "NES"): ([x],[y])， }
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g',  'c', 'm', 'y', 'k', 'orange', "pink","brown","slategrey","cornflowerblue","greenyellow"]
    # markers = [".",",","o","^","s","p","x"]
    # max_x = 0
    # min_x = 0
    our_method = 'SWITCH$_{{RGF}}$'

    xtick = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    for idx, ((dataset, norm, targeted, method), (x,y)) in enumerate(data_info.items()):
        color = colors[idx%len(colors)]
        if method == our_method:
            color = "r"
        # marker = markers[idx%len(markers)]
        # x_smooth = np.linspace(x.min(), x.max(), 300)
        # y_smooth = make_interp_spline(x, y)(x_smooth)

        min_x = x[0]
        if min_x > 0:
            x = x.tolist()
            y = y.tolist()
            x.insert(0, x[0])
            y.insert(0, 0)
            x.insert(0, 0)
            y.insert(0, 0)
        x = np.asarray(x)
        y = np.asarray(y)

        max_x = np.max(x).item()
        if max_x < 10000:  # prolong the curve with the horizontal line
            x = x.tolist()
            y = y.tolist()
            y_last = y[-1]
            for query_limit in range(0, 10001, 1000):
                if query_limit>0 and query_limit > max_x:
                    x.append(query_limit)
                    y.append(y_last)
        x = np.asarray(x)
        y = np.asarray(y)
        line, = plt.plot(x, y, label=method, color=color, linestyle="-")
        y_points = np.interp(xtick, x, y)
        plt.scatter(xtick, y_points,color=color,marker='.')

    plt.xlim(0, 10000)
    plt.ylim(0, 101)
    plt.gcf().subplots_adjust(bottom=0.15)

    # xtick = [0, 5000, 10000]
    plt.xticks(xtick,  ["0"] + ["{}K".format(int(xtick_each//1000)) for xtick_each in xtick[1:]], fontsize=22)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90,100], fontsize=22)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(loc='lower right', prop={'size': 22})
    print("save to {}".format(dump_file_path))
    plt.savefig(dump_file_path, dpi=200)
    plt.close()

def draw_success_rate_avg_query_fig(dataset, norm, targeted, arch, fig_type, dump_file_path):
    xlabel = "Attack Success Rate (%)"
    ylabel = "Avg. Query"
    methods = list(method_name_to_paper.keys())

    dataset_path_dict = get_all_exists_folder(dataset, methods, norm, targeted)
    data_info = read_all_data(dataset_path_dict, arch, fig_type)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g',  'c', 'm', 'y', 'orange', "pink","brown","slategrey","cornflowerblue","greenyellow"]
    # markers = [".", ",", "o", "^", "s", "p", "x"]
    # max_x = 0
    # min_x = 0
    max_y = 0
    our_method = 'SWITCH$_\\text{RGF}$'

    for idx, ((dataset, norm, targeted, method), (x, y)) in enumerate(data_info.items()):
        color = colors[idx % len(colors)]
        if method == our_method:
            color = "r"
        line, = plt.plot(x, y, label=method, color=color, linestyle="-", marker='.')
        if np.max(y).item() > max_y:
            max_y = np.max(y).item()
        # if np.min(x).item() < min_x:
        #     min_x = np.min(x).item()
    plt.xlim(0, 100)
    # if norm == "l2" or (dataset == "CIFAR-100" and norm == 'linf'):
    #     plt.ylim(0, 1000)
    # else:
    #     plt.ylim(0, 3000)
    # if dataset == "TinyImageNet":
    plt.ylim(max_y)
    # if dataset == "TinyImageNet":
    #     plt.ylim(0, 2750)
    #     if targeted:
    #         plt.ylim(0, 5010)
    # else:
    #     plt.ylim(0, 1750)
    #     if targeted:
    #         plt.ylim(0, 3000)

    plt.gcf().subplots_adjust(bottom=0.15)
    xtick = np.arange(0,101,5)
    # if norm == "l2" or (dataset == "CIFAR-100" and norm == 'linf'):
    #     ytick = [0,200,400,600,800,1000]
    # else:
    #     ytick = [0,500,1000,1500,2000,2500,3000]
    # if dataset == "TinyImageNet":
    #     ytick = np.arange(0,4001,200).tolist()
    space_len = int(max_y / 20.0)
    ytick = np.arange(0, max_y, space_len).tolist()

    # if dataset == "TinyImageNet":
    #     ytick = np.arange(0, 2751, 250).tolist() #,6500,7000,7500,8000,8500,9000,9500,10000]
    #     if targeted:
    #         ytick = np.arange(0, 5011, 500).tolist()
    # else:
    #     if targeted:
    #         ytick = np.arange(0, 3001, 250).tolist() #,6500,7000,7500,8000,8500,9000,9500,10000]
    #     else:
    #         ytick = np.arange(0, 1751, 200).tolist()


    # if dataset == "CIFAR-100" and (not targeted) and norm == "linf":
    #     ytick = np.arange(0, 1501, 100).tolist()
    # elif dataset == "CIFAR-10" and (not targeted) and norm == "linf":
    #     ytick = np.arange(0, 4001, 200).tolist()
    # elif dataset == "CIFAR-10" and targeted and norm == "l2":
    #     ytick = np.arange(0, 3001, 200).tolist()
    # elif norm == "l2" and targeted and dataset == "CIFAR-100":
    #     ytick = np.arange(0, 5001, 200).tolist()
    # elif dataset == "TinyImageNet" and norm == "linf":
    #     ytick = np.arange(0, 5001, 200).tolist()
    # elif dataset  == "TinyImageNet" and not targeted and norm == "l2":
    #     ytick = np.arange(0, 801, 50).tolist()
    # elif norm == "l2" and not targeted:
    #     ytick = np.arange(0, 301, 25).tolist()
    # elif norm == "l2" and not targeted and dataset == "CIFAR-100":
    #     ytick = np.arange(0, 201, 25).tolist()
    # else:
    #     ytick = np.arange(0, 4001, 200).tolist()
    # xtick = [0, 5000, 10000]
    plt.xticks(xtick, fontsize=15)
    plt.yticks(ytick, fontsize=15)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(loc='upper left', prop={'size': 18})
    plt.savefig(dump_file_path, dpi=200)
    plt.close('all')
    print("save to {}".format(dump_file_path))


def draw_histogram_fig(dataset, norm, targeted, arch, dump_folder):
    os.makedirs(dump_folder, exist_ok=True)
    x_label = "Query"
    y_label = "Number of Successfully Attacked Images"
    methods = list(method_name_to_paper.keys())
    # predefined_colors = ['b', 'g', 'c', 'm', 'y', 'k', 'w']


    predefined_colors = ['b', 'g',  'c', 'm', 'y', 'k', 'orange', "pink","brown","slategrey","cornflowerblue","greenyellow"]
    # if targeted:
    #     methods = list(filter(lambda method_name: "RGF" not in method_name, methods))
    dataset_path_dict = get_all_exists_folder(dataset, methods, norm, targeted)
    data_info = get_success_queries(dataset_path_dict, arch)
    data_dict = OrderedDict()
    colors = []
    our_method = 'SWITCH$_{{RGF}}$'
    for idx, ((dataset, norm, targeted, method), query_all) in enumerate(data_info.items()):
        color = predefined_colors[idx % len(predefined_colors)]
        if method == our_method:
            color = "r"
        # if method == "Meta Attack":
        #     color = "y"
        colors.append(color)
        data_dict[method] = query_all
    datasets = [query_all for query_all in data_dict.values()]
    labels = [method for method in data_dict.keys()]
    max_value = 1000
    if dataset == "TinyImageNet":
        max_value = 2000
    if targeted:
        max_value = 5000

    bins = 10
    # if max_value != 1000:
    #     bins = 10
    plt.hist(datasets, bins=bins, range=(0,max_value),histtype='bar', color=colors,label=labels)
    plt.xticks(np.arange(0,max_value+1,max_value//bins), fontsize=10)
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.xlim(0,max_value)
    plt.legend(loc='upper right', prop={'size': 15})
    plt.grid(True, linewidth=0.5) #,axis="y")
    dump_file_path = dump_folder + "/{dataset}_{norm}_{targeted}_attack_on_{arch}.pdf".format(dataset=dataset,
                                                                                              norm=norm,
                                                                                              targeted="untargeted" if not targeted else "targeted",
                                                                                              arch=arch)
    plt.savefig(dump_file_path)
    print("save to {}".format(dump_file_path))
    plt.close('all')

def parse_args():
    parser = argparse.ArgumentParser(description='Drawing Figures of Attacking Normal Models')
    parser.add_argument("--fig_type", type=str, choices=["query_threshold_success_rate_dict",
                                                         "success_rate_to_avg_query", "query_hist"])
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to train")
    parser.add_argument("--norm", type=str, choices=["l2", "linf"], required=True)
    parser.add_argument("--targeted", action="store_true", help="Does it train on the data of targeted attack?")
    parser.add_argument("--filter_attacks", nargs='+',help="whether to filter out some attacks, pass in the list of names")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dump_folder = "/home1/machen/query_based_black_box_attack/SWITCH_small_figures/{}/".format(args.fig_type)
    os.makedirs(dump_folder, exist_ok=True)

    if "CIFAR" in args.dataset:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        archs = ["densenet121", "resnext32_4", "resnext64_4"]

    for model in archs:

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
            # for dataset in ["CIFAR-10","CIFAR-100", "TinyImageNet"]:
                # norms = ["l2", "linf"] if not args.targeted else ["l2"]
                # for norm in norms:
                #     for model in MODELS_TEST_STANDARD[dataset]:
                #         file_path = dump_folder + "{dataset}_{model}_{norm}_{target_str}_attack_{fig_type}.pdf".format(
                #             dataset=dataset,
                #             model=model, norm=norm, target_str="untargeted" if not args.targeted else "targeted",
                #             fig_type=args.fig_type)
            for dataset in ["CIFAR-10","CIFAR-100", "TinyImageNet"]:
                if "CIFAR" in dataset:
                    archs = ['pyramidnet272', "gdas", "WRN-28-10-drop", "WRN-40-10-drop"]
                else:
                    archs = ["densenet121", "resnext32_4", "resnext64_4"]
                for norm in ["l2","linf"]:
                    for model in archs:
                        file_path = dump_folder + "{dataset}_{model}_{norm}_{target_str}_attack_{fig_type}.pdf".format(
                            dataset=dataset,
                            model=model, norm=norm, target_str="untargeted" if not args.targeted else "targeted",
                            fig_type=args.fig_type)
                        draw_query_success_rate_figure(dataset, norm, args.targeted, model, args.fig_type, file_path, x_label, y_label)
        elif args.fig_type == "success_rate_to_avg_query":
            draw_success_rate_avg_query_fig(args.dataset, args.norm, args.targeted, model, args.fig_type, file_path)
        elif args.fig_type == "query_hist":
            target_str = "/untargeted" if not args.targeted else "targeted"
            os.makedirs(dump_folder, exist_ok=True)
            for dataset in ["CIFAR-10","CIFAR-100", "TinyImageNet"]:
                if "CIFAR" in dataset:
                    archs = ['pyramidnet272', "gdas", "WRN-28-10-drop", "WRN-40-10-drop"]
                else:
                    archs = ["densenet121", "resnext32_4", "resnext64_4"]
                for norm in ["l2","linf"]:
                    for model in archs:
                        draw_histogram_fig(dataset, norm, args.targeted, model, dump_folder + target_str)
