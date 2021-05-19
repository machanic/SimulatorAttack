import numpy as np
import json
import os
def new_round(_float, _len):
    """
    Parameters
    ----------
    _float: float
    _len: int, 指定四舍五入需要保留的小数点后几位数为_len

    Returns
    -------
    type ==> float, 返回四舍五入后的值
    """
    if isinstance(_float, float):
        if str(_float)[::-1].find('.') <= _len:
            return (_float)
        if str(_float)[-1] == '5':
            return (round(float(str(_float)[:-1] + '6'), _len))
        else:
            return (round(_float, _len))
    else:
        return (round(_float, _len))


method_name_to_paper = {"bandits_attack":"Bandits",  "P-RGF_biased_attack":"PRGF",
                        "P-RGF_uniform_attack":"RGF",
                        # "MetaGradAttack":"Meta Attack",
                        # "simulate_bandits_shrink_attack":"MetaSimulator",
                        "NO_SWITCH": "NO_SWITCH",
                        #"NO_SWITCH_rnd": "NO_SWITCH_rnd",
                       # "SWITCH_rnd_save":'SWITCH_other',
                        "SWITCH_neg_save":'SWITCH_naive',
                        "SWITCH_RGF":"SWITCH_RGF",
                       # "SimBA_DCT_attack":"SimBA",
                        "PPBA_attack":"PPBA", "parsimonious_attack":"Parsimonious","sign_hunter_attack":"SignHunter",
                        "square_attack":"Square"}

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
    elif method == "NES":
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
    elif method == "SWITCH_neg_save":
        path = "SWITCH_neg_save-{dataset}-{loss}_lr_{lr}-loss-{norm}-{target_str}".format(dataset=dataset, lr=lr, loss="cw" if not targeted else "xent", norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SWITCH_rnd_save":
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


def read_json_and_extract(json_path):
    with open(json_path, "r") as file_obj:
        json_content = json.load(file_obj)
        failure_rate = json_content["avg_not_done"]
        success_rate = new_round((1-failure_rate) * 100, 1)
        if success_rate.is_integer():
            success_rate = int(success_rate)
        avg_query_over_successful_samples = int(new_round(json_content["mean_query"],0))
        median_query_over_successful_samples = int(new_round(json_content["median_query"],0))
        correct_all = np.array(json_content["correct_all"]).astype(np.bool)
        query_all = np.array(json_content["query_all"]).astype(np.float32)
        not_done_all = np.array(json_content["not_done_all"]).astype(np.bool)
        return success_rate, avg_query_over_successful_samples, median_query_over_successful_samples, \
               correct_all, query_all, not_done_all, json_content

def get_file_name_list(dataset, method_name_to_paper, norm, targeted):
    folder_path_dict = {}
    for method, paper_method_name in method_name_to_paper.items():
        if norm == "l2" and method == "parsimonious_attack":
            continue
        file_path = "/home1/machen/query_based_black_box_attack/logs/" + from_method_to_dir_path(dataset, method, norm, targeted)
        assert os.path.exists(file_path), "{} does not exist".format(file_path)
        folder_path_dict[paper_method_name] = file_path
    return folder_path_dict

def fetch_all_json_content_given_contraint(dataset, norm, targeted, arch):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result = {}
    for method, folder in folder_list.items():
        file_path = folder + "/{}_result.json".format(arch)
        assert os.path.exists(file_path), "{} does not exist!".format(file_path)
        success_rate, avg_query_over_successful_samples, median_query_over_successful_samples, \
        correct_all, query_all, not_done_all, json_content = read_json_and_extract(file_path)
        not_done_all[query_all>10000] = 1
        query_all[query_all > 10000] = 10000
        query_all[not_done_all==1] = 10000
        if norm == "l2":
            if "CIFAR" in dataset:
                epsilon = 1.0
            else:
                epsilon = 2.0
        else:
            epsilon = 0.031372
        assert epsilon == json_content["args"]["epsilon"], "eps is {} in {}".format(json_content["args"]["epsilon"], file_path)
        if "SWTICH" in method:
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
            assert lr == json_content["args"]["image_lr"]
        avg_query_over_all_samples = int(new_round(np.mean(query_all[correct_all.astype(np.bool)]).item(),0))
        median_query_over_all_samples = int(new_round(np.median(query_all[correct_all.astype(np.bool)]).item(),0))
        result[method] = {"success_rate":success_rate, "avg_query_over_successful_samples":avg_query_over_successful_samples,
                          "median_query_over_successful_samples": median_query_over_successful_samples,
                        "avg_query_over_all_samples": avg_query_over_all_samples, "median_query_over_all_samples":median_query_over_all_samples}
    return result

def draw_tables_for_TinyImageNet(norm, archs_result):
    result = archs_result
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    if norm == "linf":
        print("""
RGF \cite{{2017RGF}} & {D121_RGF_ASR}\% & {R32_RGF_ASR}\% & {R64_RGF_ASR}\% & {D121_RGF_AVGQ} & {R32_RGF_AVGQ} & {R64_RGF_AVGQ} & {D121_RGF_MEDQ} & {R32_RGF_MEDQ} & {R64_RGF_MEDQ} \\\\
P-RGF \cite{{cheng2019improving}} & {D121_PRGF_ASR}\% & {R32_PRGF_ASR}\% & {R64_PRGF_ASR}\% & {D121_PRGF_AVGQ} & {R32_PRGF_AVGQ} & {R64_PRGF_AVGQ} & {D121_PRGF_MEDQ} & {R32_PRGF_MEDQ} & {R64_PRGF_MEDQ} \\\\
Bandits \cite{{ilyas2018prior}} & {D121_Bandits_ASR}\% & {R32_Bandits_ASR}\% & {R64_Bandits_ASR}\% & {D121_Bandits_AVGQ} & {R32_Bandits_AVGQ} & {R64_Bandits_AVGQ} & {D121_Bandits_MEDQ} & {R32_Bandits_MEDQ} & {R64_Bandits_MEDQ} \\\\
PPBA \cite{{li2020projection}} & {D121_PPBA_ASR}\% & {R32_PPBA_ASR}\% & {R64_PPBA_ASR}\% & {D121_PPBA_AVGQ} & {R32_PPBA_AVGQ} & {R64_PPBA_AVGQ} & {D121_PPBA_MEDQ} & {R32_PPBA_MEDQ} & {R64_PPBA_MEDQ} \\\\
Parsimonious \cite{{moonICML19}} & {D121_Parsimonious_ASR}\% & {R32_Parsimonious_ASR}\% & {R64_Parsimonious_ASR}\% & {D121_Parsimonious_AVGQ} & {R32_Parsimonious_AVGQ} & {R64_Parsimonious_AVGQ} & {D121_Parsimonious_MEDQ} & {R32_Parsimonious_MEDQ} & {R64_Parsimonious_MEDQ} \\\\
SignHunter \cite{{al2019sign}} & {D121_SignHunter_ASR}\% & {R32_SignHunter_ASR}\% & {R64_SignHunter_ASR}\% & {D121_SignHunter_AVGQ} & {R32_SignHunter_AVGQ} & {R64_SignHunter_AVGQ} & {D121_SignHunter_MEDQ} & {R32_SignHunter_MEDQ} & {R64_SignHunter_MEDQ} \\\\
Square Attack \cite{{ACFH2020square}} & {D121_Square_ASR}\% & {R32_Square_ASR}\% & {R64_Square_ASR}\% & {D121_Square_AVGQ} & {R32_Square_AVGQ} & {R64_Square_AVGQ} & {D121_Square_MEDQ} & {R32_Square_MEDQ} & {R64_Square_MEDQ} \\\\
NO SWITCH & {D121_NO_SWITCH_ASR}\% & {R32_NO_SWITCH_ASR}\% & {R64_NO_SWITCH_ASR}\% & {D121_NO_SWITCH_AVGQ} & {R32_NO_SWITCH_AVGQ} & {R64_NO_SWITCH_AVGQ} & {D121_NO_SWITCH_MEDQ} & {R32_NO_SWITCH_MEDQ} & {R64_NO_SWITCH_MEDQ} \\\\
SWITCH_${{naive}}$ & {D121_SWITCH_naive_ASR}\% & {R32_SWITCH_naive_ASR}\% & {R64_SWITCH_naive_ASR}\% & {D121_SWITCH_naive_AVGQ} & {R32_SWITCH_naive_AVGQ} & {R64_SWITCH_naive_AVGQ} & {D121_SWITCH_naive_MEDQ} & {R32_SWITCH_naive_MEDQ} & {R64_SWITCH_naive_MEDQ} \\\\
SWITCH_${{RGF}}$ & {D121_SWITCH_RGF_ASR}\% & {R32_SWITCH_RGF_ASR}\% & {R64_SWITCH_RGF_ASR}\% & {D121_SWITCH_RGF_AVGQ} & {R32_SWITCH_RGF_AVGQ} & {R64_SWITCH_RGF_AVGQ} & {D121_SWITCH_RGF_MEDQ} & {R32_SWITCH_RGF_MEDQ} & {R64_SWITCH_RGF_MEDQ} \\\\
        """.format(
            D121_RGF_ASR=result["densenet121"]["RGF"]["success_rate"],R32_RGF_ASR=result["resnext32_4"]["RGF"]["success_rate"],R64_RGF_ASR=result["resnext64_4"]["RGF"]["success_rate"],
            D121_RGF_AVGQ=result["densenet121"]["RGF"][avg_q], R32_RGF_AVGQ=result["resnext32_4"]["RGF"][avg_q], R64_RGF_AVGQ=result["resnext64_4"]["RGF"][avg_q],
            D121_RGF_MEDQ=result["densenet121"]["RGF"][med_q], R32_RGF_MEDQ=result["resnext32_4"]["RGF"][med_q], R64_RGF_MEDQ=result["resnext64_4"]["RGF"][med_q],

            D121_PRGF_ASR=result["densenet121"]["PRGF"]["success_rate"],
            R32_PRGF_ASR=result["resnext32_4"]["PRGF"]["success_rate"],
            R64_PRGF_ASR=result["resnext64_4"]["PRGF"]["success_rate"],
            D121_PRGF_AVGQ=result["densenet121"]["PRGF"][avg_q], R32_PRGF_AVGQ=result["resnext32_4"]["PRGF"][avg_q],
            R64_PRGF_AVGQ=result["resnext64_4"]["PRGF"][avg_q],
            D121_PRGF_MEDQ=result["densenet121"]["PRGF"][med_q], R32_PRGF_MEDQ=result["resnext32_4"]["PRGF"][med_q],
            R64_PRGF_MEDQ=result["resnext64_4"]["PRGF"][med_q],

            D121_Bandits_ASR=result["densenet121"]["Bandits"]["success_rate"],
            R32_Bandits_ASR=result["resnext32_4"]["Bandits"]["success_rate"],
            R64_Bandits_ASR=result["resnext64_4"]["Bandits"]["success_rate"],
            D121_Bandits_AVGQ=result["densenet121"]["Bandits"][avg_q],
            R32_Bandits_AVGQ=result["resnext32_4"]["Bandits"][avg_q],
            R64_Bandits_AVGQ=result["resnext64_4"]["Bandits"][avg_q],
            D121_Bandits_MEDQ=result["densenet121"]["Bandits"][med_q],
            R32_Bandits_MEDQ=result["resnext32_4"]["Bandits"][med_q],
            R64_Bandits_MEDQ=result["resnext64_4"]["Bandits"][med_q],

            D121_PPBA_ASR=result["densenet121"]["PPBA"]["success_rate"],
            R32_PPBA_ASR=result["resnext32_4"]["PPBA"]["success_rate"],
            R64_PPBA_ASR=result["resnext64_4"]["PPBA"]["success_rate"],
            D121_PPBA_AVGQ=result["densenet121"]["PPBA"][avg_q], R32_PPBA_AVGQ=result["resnext32_4"]["PPBA"][avg_q],
            R64_PPBA_AVGQ=result["resnext64_4"]["PPBA"][avg_q],
            D121_PPBA_MEDQ=result["densenet121"]["PPBA"][med_q], R32_PPBA_MEDQ=result["resnext32_4"]["PPBA"][med_q],
            R64_PPBA_MEDQ=result["resnext64_4"]["PPBA"][med_q],

            D121_Parsimonious_ASR=result["densenet121"]["Parsimonious"]["success_rate"],
            R32_Parsimonious_ASR=result["resnext32_4"]["Parsimonious"]["success_rate"],
            R64_Parsimonious_ASR=result["resnext64_4"]["Parsimonious"]["success_rate"],
            D121_Parsimonious_AVGQ=result["densenet121"]["Parsimonious"][avg_q],
            R32_Parsimonious_AVGQ=result["resnext32_4"]["Parsimonious"][avg_q],
            R64_Parsimonious_AVGQ=result["resnext64_4"]["Parsimonious"][avg_q],
            D121_Parsimonious_MEDQ=result["densenet121"]["Parsimonious"][med_q],
            R32_Parsimonious_MEDQ=result["resnext32_4"]["Parsimonious"][med_q],
            R64_Parsimonious_MEDQ=result["resnext64_4"]["Parsimonious"][med_q],

            D121_SignHunter_ASR=result["densenet121"]["SignHunter"]["success_rate"],
            R32_SignHunter_ASR=result["resnext32_4"]["SignHunter"]["success_rate"],
            R64_SignHunter_ASR=result["resnext64_4"]["SignHunter"]["success_rate"],
            D121_SignHunter_AVGQ=result["densenet121"]["SignHunter"][avg_q],
            R32_SignHunter_AVGQ=result["resnext32_4"]["SignHunter"][avg_q],
            R64_SignHunter_AVGQ=result["resnext64_4"]["SignHunter"][avg_q],
            D121_SignHunter_MEDQ=result["densenet121"]["SignHunter"][med_q],
            R32_SignHunter_MEDQ=result["resnext32_4"]["SignHunter"][med_q],
            R64_SignHunter_MEDQ=result["resnext64_4"]["SignHunter"][med_q],

            D121_Square_ASR=result["densenet121"]["Square"]["success_rate"],
            R32_Square_ASR=result["resnext32_4"]["Square"]["success_rate"], R64_Square_ASR=result["resnext64_4"]
            ["Square"]["success_rate"],
            D121_Square_AVGQ=result["densenet121"]["Square"][avg_q],
            R32_Square_AVGQ=result["resnext32_4"]["Square"][avg_q],
            R64_Square_AVGQ=result["resnext64_4"]["Square"][avg_q],
            D121_Square_MEDQ=result["densenet121"]["Square"][med_q],
            R32_Square_MEDQ=result["resnext32_4"]["Square"][med_q],
            R64_Square_MEDQ=result["resnext64_4"]["Square"][med_q],

            D121_NO_SWITCH_ASR=result["densenet121"]["NO_SWITCH"]["success_rate"],
            R32_NO_SWITCH_ASR=result["resnext32_4"]["NO_SWITCH"]["success_rate"],
            R64_NO_SWITCH_ASR=result["resnext64_4"]
            ["NO_SWITCH"]["success_rate"],
            D121_NO_SWITCH_AVGQ=result["densenet121"]["NO_SWITCH"][avg_q],
            R32_NO_SWITCH_AVGQ=result["resnext32_4"]["NO_SWITCH"][avg_q],
            R64_NO_SWITCH_AVGQ=result["resnext64_4"]["NO_SWITCH"][avg_q],
            D121_NO_SWITCH_MEDQ=result["densenet121"]["NO_SWITCH"][med_q],
            R32_NO_SWITCH_MEDQ=result["resnext32_4"]["NO_SWITCH"][med_q],
            R64_NO_SWITCH_MEDQ=result["resnext64_4"]["NO_SWITCH"][med_q],

            D121_SWITCH_naive_ASR=result["densenet121"]["SWITCH_naive"]["success_rate"],
            R32_SWITCH_naive_ASR=result["resnext32_4"]["SWITCH_naive"]["success_rate"],
            R64_SWITCH_naive_ASR=result["resnext64_4"]["SWITCH_naive"]["success_rate"],
            D121_SWITCH_naive_AVGQ=result["densenet121"]["SWITCH_naive"][avg_q],
            R32_SWITCH_naive_AVGQ=result["resnext32_4"]["SWITCH_naive"][avg_q],
            R64_SWITCH_naive_AVGQ=result["resnext64_4"]["SWITCH_naive"][avg_q],
            D121_SWITCH_naive_MEDQ=result["densenet121"]["SWITCH_naive"][med_q],
            R32_SWITCH_naive_MEDQ=result["resnext32_4"]["SWITCH_naive"][med_q],
            R64_SWITCH_naive_MEDQ=result["resnext64_4"]["SWITCH_naive"][med_q],

            D121_SWITCH_RGF_ASR=result["densenet121"]["SWITCH_RGF"]["success_rate"],
            R32_SWITCH_RGF_ASR=result["resnext32_4"]["SWITCH_RGF"]["success_rate"],
            R64_SWITCH_RGF_ASR=result["resnext64_4"]["SWITCH_RGF"]["success_rate"],
            D121_SWITCH_RGF_AVGQ=result["densenet121"]["SWITCH_RGF"][avg_q],
            R32_SWITCH_RGF_AVGQ=result["resnext32_4"]["SWITCH_RGF"][avg_q],
            R64_SWITCH_RGF_AVGQ=result["resnext64_4"]["SWITCH_RGF"][avg_q],
            D121_SWITCH_RGF_MEDQ=result["densenet121"]["SWITCH_RGF"][med_q],
            R32_SWITCH_RGF_MEDQ=result["resnext32_4"]["SWITCH_RGF"][med_q],
            R64_SWITCH_RGF_MEDQ=result["resnext64_4"]["SWITCH_RGF"][med_q],

            # D121_NO_SWITCH_rnd_ASR=result["densenet121"]["NO_SWITCH_rnd"]["success_rate"],
            # R32_NO_SWITCH_rnd_ASR=result["resnext32_4"]["NO_SWITCH_rnd"]["success_rate"],
            # R64_NO_SWITCH_rnd_ASR=result["resnext64_4"]["NO_SWITCH_rnd"]["success_rate"],
            # D121_NO_SWITCH_rnd_AVGQ=result["densenet121"]["NO_SWITCH_rnd"][avg_q],
            # R32_NO_SWITCH_rnd_AVGQ=result["resnext32_4"]["NO_SWITCH_rnd"][avg_q],
            # R64_NO_SWITCH_rnd_AVGQ=result["resnext64_4"]["NO_SWITCH_rnd"][avg_q],
            # D121_NO_SWITCH_rnd_MEDQ=result["densenet121"]["NO_SWITCH_rnd"][med_q],
            # R32_NO_SWITCH_rnd_MEDQ=result["resnext32_4"]["NO_SWITCH_rnd"][med_q],
            # R64_NO_SWITCH_rnd_MEDQ=result["resnext64_4"]["NO_SWITCH_rnd"][med_q],
            #
            # D121_SWITCH_other_ASR=result["densenet121"]["SWITCH_other"]["success_rate"],
            # R32_SWITCH_other_ASR=result["resnext32_4"]["SWITCH_other"]["success_rate"],
            # R64_SWITCH_other_ASR=result["resnext64_4"]["SWITCH_other"]["success_rate"],
            # D121_SWITCH_other_AVGQ=result["densenet121"]["SWITCH_other"][avg_q],
            # R32_SWITCH_other_AVGQ=result["resnext32_4"]["SWITCH_other"][avg_q],
            # R64_SWITCH_other_AVGQ=result["resnext64_4"]["SWITCH_other"][avg_q],
            # D121_SWITCH_other_MEDQ=result["densenet121"]["SWITCH_other"][med_q],
            # R32_SWITCH_other_MEDQ=result["resnext32_4"]["SWITCH_other"][med_q],
            # R64_SWITCH_other_MEDQ=result["resnext64_4"]["SWITCH_other"][med_q]
        )
              )
    else:
        print("""
        RGF \cite{{2017RGF}} & {D121_RGF_ASR}\% & {R32_RGF_ASR}\% & {R64_RGF_ASR}\% & {D121_RGF_AVGQ} & {R32_RGF_AVGQ} & {R64_RGF_AVGQ} & {D121_RGF_MEDQ} & {R32_RGF_MEDQ} & {R64_RGF_MEDQ} \\\\
        P-RGF \cite{{cheng2019improving}} & {D121_PRGF_ASR}\% & {R32_PRGF_ASR}\% & {R64_PRGF_ASR}\% & {D121_PRGF_AVGQ} & {R32_PRGF_AVGQ} & {R64_PRGF_AVGQ} & {D121_PRGF_MEDQ} & {R32_PRGF_MEDQ} & {R64_PRGF_MEDQ} \\\\
        Bandits \cite{{ilyas2018prior}} & {D121_Bandits_ASR}\% & {R32_Bandits_ASR}\% & {R64_Bandits_ASR}\% & {D121_Bandits_AVGQ} & {R32_Bandits_AVGQ} & {R64_Bandits_AVGQ} & {D121_Bandits_MEDQ} & {R32_Bandits_MEDQ} & {R64_Bandits_MEDQ} \\\\
        PPBA \cite{{li2020projection}} & {D121_PPBA_ASR}\% & {R32_PPBA_ASR}\% & {R64_PPBA_ASR}\% & {D121_PPBA_AVGQ} & {R32_PPBA_AVGQ} & {R64_PPBA_AVGQ} & {D121_PPBA_MEDQ} & {R32_PPBA_MEDQ} & {R64_PPBA_MEDQ} \\\\
        SignHunter \cite{{al2019sign}} & {D121_SignHunter_ASR}\% & {R32_SignHunter_ASR}\% & {R64_SignHunter_ASR}\% & {D121_SignHunter_AVGQ} & {R32_SignHunter_AVGQ} & {R64_SignHunter_AVGQ} & {D121_SignHunter_MEDQ} & {R32_SignHunter_MEDQ} & {R64_SignHunter_MEDQ} \\\\
        Square Attack \cite{{ACFH2020square}} & {D121_Square_ASR}\% & {R32_Square_ASR}\% & {R64_Square_ASR}\% & {D121_Square_AVGQ} & {R32_Square_AVGQ} & {R64_Square_AVGQ} & {D121_Square_MEDQ} & {R32_Square_MEDQ} & {R64_Square_MEDQ} \\\\
        NO SWITCH & {D121_NO_SWITCH_ASR}\% & {R32_NO_SWITCH_ASR}\% & {R64_NO_SWITCH_ASR}\% & {D121_NO_SWITCH_AVGQ} & {R32_NO_SWITCH_AVGQ} & {R64_NO_SWITCH_AVGQ} & {D121_NO_SWITCH_MEDQ} & {R32_NO_SWITCH_MEDQ} & {R64_NO_SWITCH_MEDQ} \\\\
        SWITCH_${{naive}}$ & {D121_SWITCH_naive_ASR}\% & {R32_SWITCH_naive_ASR}\% & {R64_SWITCH_naive_ASR}\% & {D121_SWITCH_naive_AVGQ} & {R32_SWITCH_naive_AVGQ} & {R64_SWITCH_naive_AVGQ} & {D121_SWITCH_naive_MEDQ} & {R32_SWITCH_naive_MEDQ} & {R64_SWITCH_naive_MEDQ} \\\\
        SWITCH_${{RGF}}$ & {D121_SWITCH_RGF_ASR}\% & {R32_SWITCH_RGF_ASR}\% & {R64_SWITCH_RGF_ASR}\% & {D121_SWITCH_RGF_AVGQ} & {R32_SWITCH_RGF_AVGQ} & {R64_SWITCH_RGF_AVGQ} & {D121_SWITCH_RGF_MEDQ} & {R32_SWITCH_RGF_MEDQ} & {R64_SWITCH_RGF_MEDQ} \\\\        
                """.format(
            D121_RGF_ASR=result["densenet121"]["RGF"]["success_rate"],
            R32_RGF_ASR=result["resnext32_4"]["RGF"]["success_rate"],
            R64_RGF_ASR=result["resnext64_4"]["RGF"]["success_rate"],
            D121_RGF_AVGQ=result["densenet121"]["RGF"][avg_q], R32_RGF_AVGQ=result["resnext32_4"]["RGF"][avg_q],
            R64_RGF_AVGQ=result["resnext64_4"]["RGF"][avg_q],
            D121_RGF_MEDQ=result["densenet121"]["RGF"][med_q], R32_RGF_MEDQ=result["resnext32_4"]["RGF"][med_q],
            R64_RGF_MEDQ=result["resnext64_4"]["RGF"][med_q],

            D121_PRGF_ASR=result["densenet121"]["PRGF"]["success_rate"],
            R32_PRGF_ASR=result["resnext32_4"]["PRGF"]["success_rate"],
            R64_PRGF_ASR=result["resnext64_4"]["PRGF"]["success_rate"],
            D121_PRGF_AVGQ=result["densenet121"]["PRGF"][avg_q], R32_PRGF_AVGQ=result["resnext32_4"]["PRGF"][avg_q],
            R64_PRGF_AVGQ=result["resnext64_4"]["PRGF"][avg_q],
            D121_PRGF_MEDQ=result["densenet121"]["PRGF"][med_q], R32_PRGF_MEDQ=result["resnext32_4"]["PRGF"][med_q],
            R64_PRGF_MEDQ=result["resnext64_4"]["PRGF"][med_q],

            D121_Bandits_ASR=result["densenet121"]["Bandits"]["success_rate"],
            R32_Bandits_ASR=result["resnext32_4"]["Bandits"]["success_rate"],
            R64_Bandits_ASR=result["resnext64_4"]["Bandits"]["success_rate"],
            D121_Bandits_AVGQ=result["densenet121"]["Bandits"][avg_q],
            R32_Bandits_AVGQ=result["resnext32_4"]["Bandits"][avg_q],
            R64_Bandits_AVGQ=result["resnext64_4"]["Bandits"][avg_q],
            D121_Bandits_MEDQ=result["densenet121"]["Bandits"][med_q],
            R32_Bandits_MEDQ=result["resnext32_4"]["Bandits"][med_q],
            R64_Bandits_MEDQ=result["resnext64_4"]["Bandits"][med_q],

            D121_PPBA_ASR=result["densenet121"]["PPBA"]["success_rate"],
            R32_PPBA_ASR=result["resnext32_4"]["PPBA"]["success_rate"],
            R64_PPBA_ASR=result["resnext64_4"]["PPBA"]["success_rate"],
            D121_PPBA_AVGQ=result["densenet121"]["PPBA"][avg_q], R32_PPBA_AVGQ=result["resnext32_4"]["PPBA"][avg_q],
            R64_PPBA_AVGQ=result["resnext64_4"]["PPBA"][avg_q],
            D121_PPBA_MEDQ=result["densenet121"]["PPBA"][med_q], R32_PPBA_MEDQ=result["resnext32_4"]["PPBA"][med_q],
            R64_PPBA_MEDQ=result["resnext64_4"]["PPBA"][med_q],

            D121_SignHunter_ASR=result["densenet121"]["SignHunter"]["success_rate"],
            R32_SignHunter_ASR=result["resnext32_4"]["SignHunter"]["success_rate"],
            R64_SignHunter_ASR=result["resnext64_4"]["SignHunter"]["success_rate"],
            D121_SignHunter_AVGQ=result["densenet121"]["SignHunter"][avg_q],
            R32_SignHunter_AVGQ=result["resnext32_4"]["SignHunter"][avg_q],
            R64_SignHunter_AVGQ=result["resnext64_4"]["SignHunter"][avg_q],
            D121_SignHunter_MEDQ=result["densenet121"]["SignHunter"][med_q],
            R32_SignHunter_MEDQ=result["resnext32_4"]["SignHunter"][med_q],
            R64_SignHunter_MEDQ=result["resnext64_4"]["SignHunter"][med_q],

            D121_Square_ASR=result["densenet121"]["Square"]["success_rate"],
            R32_Square_ASR=result["resnext32_4"]["Square"]["success_rate"], R64_Square_ASR=result["resnext64_4"]
            ["Square"]["success_rate"],
            D121_Square_AVGQ=result["densenet121"]["Square"][avg_q],
            R32_Square_AVGQ=result["resnext32_4"]["Square"][avg_q],
            R64_Square_AVGQ=result["resnext64_4"]["Square"][avg_q],
            D121_Square_MEDQ=result["densenet121"]["Square"][med_q],
            R32_Square_MEDQ=result["resnext32_4"]["Square"][med_q],
            R64_Square_MEDQ=result["resnext64_4"]["Square"][med_q],

            D121_NO_SWITCH_ASR=result["densenet121"]["NO_SWITCH"]["success_rate"],
            R32_NO_SWITCH_ASR=result["resnext32_4"]["NO_SWITCH"]["success_rate"],
            R64_NO_SWITCH_ASR=result["resnext64_4"]
            ["NO_SWITCH"]["success_rate"],
            D121_NO_SWITCH_AVGQ=result["densenet121"]["NO_SWITCH"][avg_q],
            R32_NO_SWITCH_AVGQ=result["resnext32_4"]["NO_SWITCH"][avg_q],
            R64_NO_SWITCH_AVGQ=result["resnext64_4"]["NO_SWITCH"][avg_q],
            D121_NO_SWITCH_MEDQ=result["densenet121"]["NO_SWITCH"][med_q],
            R32_NO_SWITCH_MEDQ=result["resnext32_4"]["NO_SWITCH"][med_q],
            R64_NO_SWITCH_MEDQ=result["resnext64_4"]["NO_SWITCH"][med_q],

            D121_SWITCH_naive_ASR=result["densenet121"]["SWITCH_naive"]["success_rate"],
            R32_SWITCH_naive_ASR=result["resnext32_4"]["SWITCH_naive"]["success_rate"],
            R64_SWITCH_naive_ASR=result["resnext64_4"]["SWITCH_naive"]["success_rate"],
            D121_SWITCH_naive_AVGQ=result["densenet121"]["SWITCH_naive"][avg_q],
            R32_SWITCH_naive_AVGQ=result["resnext32_4"]["SWITCH_naive"][avg_q],
            R64_SWITCH_naive_AVGQ=result["resnext64_4"]["SWITCH_naive"][avg_q],
            D121_SWITCH_naive_MEDQ=result["densenet121"]["SWITCH_naive"][med_q],
            R32_SWITCH_naive_MEDQ=result["resnext32_4"]["SWITCH_naive"][med_q],
            R64_SWITCH_naive_MEDQ=result["resnext64_4"]["SWITCH_naive"][med_q],

            D121_SWITCH_RGF_ASR=result["densenet121"]["SWITCH_RGF"]["success_rate"],
            R32_SWITCH_RGF_ASR=result["resnext32_4"]["SWITCH_RGF"]["success_rate"],
            R64_SWITCH_RGF_ASR=result["resnext64_4"]["SWITCH_RGF"]["success_rate"],
            D121_SWITCH_RGF_AVGQ=result["densenet121"]["SWITCH_RGF"][avg_q],
            R32_SWITCH_RGF_AVGQ=result["resnext32_4"]["SWITCH_RGF"][avg_q],
            R64_SWITCH_RGF_AVGQ=result["resnext64_4"]["SWITCH_RGF"][avg_q],
            D121_SWITCH_RGF_MEDQ=result["densenet121"]["SWITCH_RGF"][med_q],
            R32_SWITCH_RGF_MEDQ=result["resnext32_4"]["SWITCH_RGF"][med_q],
            R64_SWITCH_RGF_MEDQ=result["resnext64_4"]["SWITCH_RGF"][med_q],
            # 
            # D121_NO_SWITCH_rnd_ASR=result["densenet121"]["NO_SWITCH_rnd"]["success_rate"],
            # R32_NO_SWITCH_rnd_ASR=result["resnext32_4"]["NO_SWITCH_rnd"]["success_rate"],
            # R64_NO_SWITCH_rnd_ASR=result["resnext64_4"]["NO_SWITCH_rnd"]["success_rate"],
            # D121_NO_SWITCH_rnd_AVGQ=result["densenet121"]["NO_SWITCH_rnd"][avg_q],
            # R32_NO_SWITCH_rnd_AVGQ=result["resnext32_4"]["NO_SWITCH_rnd"][avg_q],
            # R64_NO_SWITCH_rnd_AVGQ=result["resnext64_4"]["NO_SWITCH_rnd"][avg_q],
            # D121_NO_SWITCH_rnd_MEDQ=result["densenet121"]["NO_SWITCH_rnd"][med_q],
            # R32_NO_SWITCH_rnd_MEDQ=result["resnext32_4"]["NO_SWITCH_rnd"][med_q],
            # R64_NO_SWITCH_rnd_MEDQ=result["resnext64_4"]["NO_SWITCH_rnd"][med_q],
            # 
            # D121_SWITCH_other_ASR=result["densenet121"]["SWITCH_other"]["success_rate"],
            # R32_SWITCH_other_ASR=result["resnext32_4"]["SWITCH_other"]["success_rate"],
            # R64_SWITCH_other_ASR=result["resnext64_4"]["SWITCH_other"]["success_rate"],
            # D121_SWITCH_other_AVGQ=result["densenet121"]["SWITCH_other"][avg_q],
            # R32_SWITCH_other_AVGQ=result["resnext32_4"]["SWITCH_other"][avg_q],
            # R64_SWITCH_other_AVGQ=result["resnext64_4"]["SWITCH_other"][avg_q],
            # D121_SWITCH_other_MEDQ=result["densenet121"]["SWITCH_other"][med_q],
            # R32_SWITCH_other_MEDQ=result["resnext32_4"]["SWITCH_other"][med_q],
            # R64_SWITCH_other_MEDQ=result["resnext64_4"]["SWITCH_other"][med_q]
        )
        )


def draw_tables_for_CIFAR(norm, archs_result):
    result = archs_result
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    # archs_result = {"pyramidnet272" : {"NES": {}, "P-RGF": {}, }, "gdas": { 各种方法}}
    if norm == "linf":
        print("""
                & {norm_str} & RGF \cite{{2017RGF}} & {pyramidnet272_RGF_ASR}\% & {gdas_RGF_ASR}\% & {WRN28_RGF_ASR}\% & {WRN40_RGF_ASR}\% & {pyramidnet272_RGF_AVGQ} & {gdas_RGF_AVGQ} & {WRN28_RGF_AVGQ} & {WRN40_RGF_AVGQ} & {pyramidnet272_RGF_MEDQ} & {gdas_RGF_MEDQ} & {WRN28_RGF_MEDQ} & {WRN40_RGF_MEDQ} \\\\
                & & P-RGF \cite{{cheng2019improving}} & {pyramidnet272_PRGF_ASR}\% & {gdas_PRGF_ASR}\% & {WRN28_PRGF_ASR}\% & {WRN40_PRGF_ASR}\% & {pyramidnet272_PRGF_AVGQ} & {gdas_PRGF_AVGQ} & {WRN28_PRGF_AVGQ} & {WRN40_PRGF_AVGQ} & {pyramidnet272_PRGF_MEDQ} & {gdas_PRGF_MEDQ} & {WRN28_PRGF_MEDQ} & {WRN40_PRGF_MEDQ} \\\\
                & & Bandits \cite{{ilyas2018prior}} & {pyramidnet272_Bandits_ASR}\% & {gdas_Bandits_ASR}\% & {WRN28_Bandits_ASR}\% & {WRN40_Bandits_ASR}\% & {pyramidnet272_Bandits_AVGQ} & {gdas_Bandits_AVGQ} & {WRN28_Bandits_AVGQ} & {WRN40_Bandits_AVGQ} & {pyramidnet272_Bandits_MEDQ} & {gdas_Bandits_MEDQ} & {WRN28_Bandits_MEDQ} & {WRN40_Bandits_MEDQ} \\\\
                & & PPBA \cite{{li2020projection}} & {pyramidnet272_PPBA_ASR}\% & {gdas_PPBA_ASR}\% & {WRN28_PPBA_ASR}\% & {WRN40_PPBA_ASR}\% & {pyramidnet272_PPBA_AVGQ} & {gdas_PPBA_AVGQ} & {WRN28_PPBA_AVGQ} & {WRN40_PPBA_AVGQ} & {pyramidnet272_PPBA_MEDQ} & {gdas_PPBA_MEDQ} & {WRN28_PPBA_MEDQ} & {WRN40_PPBA_MEDQ} \\\\
                & & Parsimonious \cite{{moonICML19}} & {pyramidnet272_Parsimonious_ASR}\% & {gdas_Parsimonious_ASR}\% & {WRN28_Parsimonious_ASR}\% & {WRN40_Parsimonious_ASR}\% & {pyramidnet272_Parsimonious_AVGQ} & {gdas_Parsimonious_AVGQ} & {WRN28_Parsimonious_AVGQ} & {WRN40_Parsimonious_AVGQ} & {pyramidnet272_Parsimonious_MEDQ} & {gdas_Parsimonious_MEDQ} & {WRN28_Parsimonious_MEDQ} & {WRN40_Parsimonious_MEDQ} \\\\
                & & SignHunter \cite{{al2019sign}} & {pyramidnet272_SignHunter_ASR}\% & {gdas_SignHunter_ASR}\% & {WRN28_SignHunter_ASR}\% & {WRN40_SignHunter_ASR}\% & {pyramidnet272_SignHunter_AVGQ} & {gdas_SignHunter_AVGQ} & {WRN28_SignHunter_AVGQ} & {WRN40_SignHunter_AVGQ} & {pyramidnet272_SignHunter_MEDQ} & {gdas_SignHunter_MEDQ} & {WRN28_SignHunter_MEDQ} & {WRN40_SignHunter_MEDQ} \\\\
                & & Square Attack \cite{{ACFH2020square}} & {pyramidnet272_Square_ASR}\% & {gdas_Square_ASR}\% & {WRN28_Square_ASR}\% & {WRN40_Square_ASR}\% & {pyramidnet272_Square_AVGQ} & {gdas_Square_AVGQ} & {WRN28_Square_AVGQ} & {WRN40_Square_AVGQ} & {pyramidnet272_Square_MEDQ} & {gdas_Square_MEDQ} & {WRN28_Square_MEDQ} & {WRN40_Square_MEDQ} \\\\
                & & NO SWITCH & {pyramidnet272_NO_SWITCH_ASR}\% & {gdas_NO_SWITCH_ASR}\% & {WRN28_NO_SWITCH_ASR}\% & {WRN40_NO_SWITCH_ASR}\% & {pyramidnet272_NO_SWITCH_AVGQ} & {gdas_NO_SWITCH_AVGQ} & {WRN28_NO_SWITCH_AVGQ} & {WRN40_NO_SWITCH_AVGQ} & {pyramidnet272_NO_SWITCH_MEDQ} & {gdas_NO_SWITCH_MEDQ} & {WRN28_NO_SWITCH_MEDQ} & {WRN40_NO_SWITCH_MEDQ} \\\\
                & & SWITCH$_{{naive}}$ & {pyramidnet272_SWITCH_naive_ASR}\% & {gdas_SWITCH_naive_ASR}\% & {WRN28_SWITCH_naive_ASR}\% & {WRN40_SWITCH_naive_ASR}\% & {pyramidnet272_SWITCH_naive_AVGQ} & {gdas_SWITCH_naive_AVGQ} & {WRN28_SWITCH_naive_AVGQ} & {WRN40_SWITCH_naive_AVGQ} & {pyramidnet272_SWITCH_naive_MEDQ} & {gdas_SWITCH_naive_MEDQ} & {WRN28_SWITCH_naive_MEDQ} & {WRN40_SWITCH_naive_MEDQ} \\\\
                & & SWITCH$_{{RGF}}$ & {pyramidnet272_SWITCH_RGF_ASR}\% & {gdas_SWITCH_RGF_ASR}\% & {WRN28_SWITCH_RGF_ASR}\% & {WRN40_SWITCH_RGF_ASR}\% & {pyramidnet272_SWITCH_RGF_AVGQ} & {gdas_SWITCH_RGF_AVGQ} & {WRN28_SWITCH_RGF_AVGQ} & {WRN40_SWITCH_RGF_AVGQ} & {pyramidnet272_SWITCH_RGF_MEDQ} & {gdas_SWITCH_RGF_MEDQ} & {WRN28_SWITCH_RGF_MEDQ} & {WRN40_SWITCH_RGF_MEDQ} \\\\
        """.format(norm_str="$\ell_\infty$" if norm == "linf" else "$\ell_2$",
                   # pyramidnet272_NES_ASR=result["pyramidnet272"]["NES"]["success_rate"], gdas_NES_ASR=result["gdas"]["NES"]["success_rate"],
                   # WRN28_NES_ASR=result["WRN-28-10-drop"]["NES"]["success_rate"], WRN40_NES_ASR=result["WRN-40-10-drop"]["NES"]["success_rate"],
                   # pyramidnet272_NES_AVGQ=result["pyramidnet272"]["NES"][avg_q],
                   # gdas_NES_AVGQ=result["gdas"]["NES"][avg_q], WRN28_NES_AVGQ=result["WRN-28-10-drop"]["NES"][avg_q],  WRN40_NES_AVGQ=result["WRN-40-10-drop"]["NES"][avg_q],
                   # pyramidnet272_NES_MEDQ=result["pyramidnet272"]["NES"][med_q],
                   # gdas_NES_MEDQ=result["gdas"]["NES"][med_q], WRN28_NES_MEDQ=result["WRN-28-10-drop"]["NES"][med_q],
                   # WRN40_NES_MEDQ=result["WRN-40-10-drop"]["NES"][med_q],

                   pyramidnet272_RGF_ASR=result["pyramidnet272"]["RGF"]["success_rate"],
                   gdas_RGF_ASR=result["gdas"]["RGF"]["success_rate"],
                   WRN28_RGF_ASR=result["WRN-28-10-drop"]["RGF"]["success_rate"],
                   WRN40_RGF_ASR=result["WRN-40-10-drop"]["RGF"]["success_rate"],
                   pyramidnet272_RGF_AVGQ=result["pyramidnet272"]["RGF"][avg_q],
                   gdas_RGF_AVGQ=result["gdas"]["RGF"][avg_q], WRN28_RGF_AVGQ=result["WRN-28-10-drop"]["RGF"][avg_q],
                   WRN40_RGF_AVGQ=result["WRN-40-10-drop"]["RGF"][avg_q],
                   pyramidnet272_RGF_MEDQ=result["pyramidnet272"]["RGF"][med_q],
                   gdas_RGF_MEDQ=result["gdas"]["RGF"][med_q], WRN28_RGF_MEDQ=result["WRN-28-10-drop"]["RGF"][med_q],
                   WRN40_RGF_MEDQ=result["WRN-40-10-drop"]["RGF"][med_q],

                   pyramidnet272_PRGF_ASR=result["pyramidnet272"]["PRGF"]["success_rate"],
                   gdas_PRGF_ASR=result["gdas"]["PRGF"]["success_rate"],
                   WRN28_PRGF_ASR=result["WRN-28-10-drop"]["PRGF"]["success_rate"],
                   WRN40_PRGF_ASR=result["WRN-40-10-drop"]["PRGF"]["success_rate"],
                   pyramidnet272_PRGF_AVGQ=result["pyramidnet272"]["PRGF"][avg_q],
                   gdas_PRGF_AVGQ=result["gdas"]["PRGF"][avg_q], WRN28_PRGF_AVGQ=result["WRN-28-10-drop"]["PRGF"][avg_q],
                   WRN40_PRGF_AVGQ=result["WRN-40-10-drop"]["PRGF"][avg_q],
                   pyramidnet272_PRGF_MEDQ=result["pyramidnet272"]["PRGF"][med_q],
                   gdas_PRGF_MEDQ=result["gdas"]["PRGF"][med_q], WRN28_PRGF_MEDQ=result["WRN-28-10-drop"]["PRGF"][med_q],
                   WRN40_PRGF_MEDQ=result["WRN-40-10-drop"]["PRGF"][med_q],

                   pyramidnet272_Bandits_ASR=result["pyramidnet272"]["Bandits"]["success_rate"],
                   gdas_Bandits_ASR=result["gdas"]["Bandits"]["success_rate"],
                   WRN28_Bandits_ASR=result["WRN-28-10-drop"]["Bandits"]["success_rate"],
                   WRN40_Bandits_ASR=result["WRN-40-10-drop"]["Bandits"]["success_rate"],
                   pyramidnet272_Bandits_AVGQ=result["pyramidnet272"]["Bandits"][avg_q],
                   gdas_Bandits_AVGQ=result["gdas"]["Bandits"][avg_q],
                   WRN28_Bandits_AVGQ=result["WRN-28-10-drop"]["Bandits"][avg_q],
                   WRN40_Bandits_AVGQ=result["WRN-40-10-drop"]["Bandits"][avg_q],
                   pyramidnet272_Bandits_MEDQ=result["pyramidnet272"]["Bandits"][med_q],
                   gdas_Bandits_MEDQ=result["gdas"]["Bandits"][med_q],
                   WRN28_Bandits_MEDQ=result["WRN-28-10-drop"]["Bandits"][med_q],
                   WRN40_Bandits_MEDQ=result["WRN-40-10-drop"]["Bandits"][med_q],

                   pyramidnet272_PPBA_ASR=result["pyramidnet272"]["PPBA"]["success_rate"],
                   gdas_PPBA_ASR=result["gdas"]["PPBA"]["success_rate"],
                   WRN28_PPBA_ASR=result["WRN-28-10-drop"]["PPBA"]["success_rate"],
                   WRN40_PPBA_ASR=result["WRN-40-10-drop"]["PPBA"]["success_rate"],
                   pyramidnet272_PPBA_AVGQ=result["pyramidnet272"]["PPBA"][avg_q],
                   gdas_PPBA_AVGQ=result["gdas"]["PPBA"][avg_q], WRN28_PPBA_AVGQ=result["WRN-28-10-drop"]["PPBA"][avg_q],
                   WRN40_PPBA_AVGQ=result["WRN-40-10-drop"]["PPBA"][avg_q],
                   pyramidnet272_PPBA_MEDQ=result["pyramidnet272"]["PPBA"][med_q],
                   gdas_PPBA_MEDQ=result["gdas"]["PPBA"][med_q], WRN28_PPBA_MEDQ=result["WRN-28-10-drop"]["PPBA"][med_q],
                   WRN40_PPBA_MEDQ=result["WRN-40-10-drop"]["PPBA"][med_q],

                   pyramidnet272_Parsimonious_ASR=result["pyramidnet272"]["Parsimonious"]["success_rate"],
                   gdas_Parsimonious_ASR=result["gdas"]["Parsimonious"]["success_rate"],
                   WRN28_Parsimonious_ASR=result["WRN-28-10-drop"]["Parsimonious"]["success_rate"],
                   WRN40_Parsimonious_ASR=result["WRN-40-10-drop"]["Parsimonious"]["success_rate"],
                   pyramidnet272_Parsimonious_AVGQ=result["pyramidnet272"]["Parsimonious"][avg_q],
                   gdas_Parsimonious_AVGQ=result["gdas"]["Parsimonious"][avg_q],
                   WRN28_Parsimonious_AVGQ=result["WRN-28-10-drop"]["Parsimonious"][avg_q],
                   WRN40_Parsimonious_AVGQ=result["WRN-40-10-drop"]["Parsimonious"][avg_q],
                   pyramidnet272_Parsimonious_MEDQ=result["pyramidnet272"]["Parsimonious"][med_q],
                   gdas_Parsimonious_MEDQ=result["gdas"]["Parsimonious"][med_q],
                   WRN28_Parsimonious_MEDQ=result["WRN-28-10-drop"]["Parsimonious"][med_q],
                   WRN40_Parsimonious_MEDQ=result["WRN-40-10-drop"]["Parsimonious"][med_q],

                   # pyramidnet272_SimBA_ASR=result["pyramidnet272"]["SimBA"]["success_rate"],
                   # gdas_SimBA_ASR=result["gdas"]["SimBA"]["success_rate"],
                   # WRN28_SimBA_ASR=result["WRN-28-10-drop"]["SimBA"]["success_rate"],
                   # WRN40_SimBA_ASR=result["WRN-40-10-drop"]["SimBA"]["success_rate"],
                   # pyramidnet272_SimBA_AVGQ=result["pyramidnet272"]["SimBA"][avg_q],
                   # gdas_SimBA_AVGQ=result["gdas"]["SimBA"][avg_q],
                   # WRN28_SimBA_AVGQ=result["WRN-28-10-drop"]["SimBA"][avg_q],
                   # WRN40_SimBA_AVGQ=result["WRN-40-10-drop"]["SimBA"][avg_q],
                   # pyramidnet272_SimBA_MEDQ=result["pyramidnet272"]["SimBA"][med_q],
                   # gdas_SimBA_MEDQ=result["gdas"]["SimBA"][med_q],
                   # WRN28_SimBA_MEDQ=result["WRN-28-10-drop"]["SimBA"][med_q],
                   # WRN40_SimBA_MEDQ=result["WRN-40-10-drop"]["SimBA"][med_q],

                   pyramidnet272_SignHunter_ASR=result["pyramidnet272"]["SignHunter"]["success_rate"],
                   gdas_SignHunter_ASR=result["gdas"]["SignHunter"]["success_rate"],
                   WRN28_SignHunter_ASR=result["WRN-28-10-drop"]["SignHunter"]["success_rate"],
                   WRN40_SignHunter_ASR=result["WRN-40-10-drop"]["SignHunter"]["success_rate"],
                   pyramidnet272_SignHunter_AVGQ=result["pyramidnet272"]["SignHunter"][avg_q],
                   gdas_SignHunter_AVGQ=result["gdas"]["SignHunter"][avg_q],
                   WRN28_SignHunter_AVGQ=result["WRN-28-10-drop"]["SignHunter"][avg_q],
                   WRN40_SignHunter_AVGQ=result["WRN-40-10-drop"]["SignHunter"][avg_q],
                   pyramidnet272_SignHunter_MEDQ=result["pyramidnet272"]["SignHunter"][med_q],
                   gdas_SignHunter_MEDQ=result["gdas"]["SignHunter"][med_q],
                   WRN28_SignHunter_MEDQ=result["WRN-28-10-drop"]["SignHunter"][med_q],
                   WRN40_SignHunter_MEDQ=result["WRN-40-10-drop"]["SignHunter"][med_q],

                   pyramidnet272_Square_ASR=result["pyramidnet272"]["Square"]["success_rate"],
                   gdas_Square_ASR=result["gdas"]["Square"]["success_rate"],
                   WRN28_Square_ASR=result["WRN-28-10-drop"]["Square"]["success_rate"],
                   WRN40_Square_ASR=result["WRN-40-10-drop"]["Square"]["success_rate"],
                   pyramidnet272_Square_AVGQ=result["pyramidnet272"]["Square"][avg_q],
                   gdas_Square_AVGQ=result["gdas"]["Square"][avg_q],
                   WRN28_Square_AVGQ=result["WRN-28-10-drop"]["Square"][avg_q],
                   WRN40_Square_AVGQ=result["WRN-40-10-drop"]["Square"][avg_q],
                   pyramidnet272_Square_MEDQ=result["pyramidnet272"]["Square"][med_q],
                   gdas_Square_MEDQ=result["gdas"]["Square"][med_q],
                   WRN28_Square_MEDQ=result["WRN-28-10-drop"]["Square"][med_q],
                   WRN40_Square_MEDQ=result["WRN-40-10-drop"]["Square"][med_q],

                   pyramidnet272_NO_SWITCH_ASR=result["pyramidnet272"]["NO_SWITCH"]["success_rate"],
                   gdas_NO_SWITCH_ASR=result["gdas"]["NO_SWITCH"]["success_rate"],
                   WRN28_NO_SWITCH_ASR=result["WRN-28-10-drop"]["NO_SWITCH"]["success_rate"],
                   WRN40_NO_SWITCH_ASR=result["WRN-40-10-drop"]["NO_SWITCH"]["success_rate"],
                   pyramidnet272_NO_SWITCH_AVGQ=result["pyramidnet272"]["NO_SWITCH"][avg_q],
                   gdas_NO_SWITCH_AVGQ=result["gdas"]["NO_SWITCH"][avg_q],
                   WRN28_NO_SWITCH_AVGQ=result["WRN-28-10-drop"]["NO_SWITCH"][avg_q],
                   WRN40_NO_SWITCH_AVGQ=result["WRN-40-10-drop"]["NO_SWITCH"][avg_q],
                   pyramidnet272_NO_SWITCH_MEDQ=result["pyramidnet272"]["NO_SWITCH"][med_q],
                   gdas_NO_SWITCH_MEDQ=result["gdas"]["NO_SWITCH"][med_q],
                   WRN28_NO_SWITCH_MEDQ=result["WRN-28-10-drop"]["NO_SWITCH"][med_q],
                   WRN40_NO_SWITCH_MEDQ=result["WRN-40-10-drop"]["NO_SWITCH"][med_q],

                   pyramidnet272_SWITCH_naive_ASR=result["pyramidnet272"]["SWITCH_naive"]["success_rate"],
                   gdas_SWITCH_naive_ASR=result["gdas"]["SWITCH_naive"]["success_rate"],
                   WRN28_SWITCH_naive_ASR=result["WRN-28-10-drop"]["SWITCH_naive"]["success_rate"],
                   WRN40_SWITCH_naive_ASR=result["WRN-40-10-drop"]["SWITCH_naive"]["success_rate"],
                   pyramidnet272_SWITCH_naive_AVGQ=result["pyramidnet272"]["SWITCH_naive"][avg_q],
                   gdas_SWITCH_naive_AVGQ=result["gdas"]["SWITCH_naive"][avg_q],
                   WRN28_SWITCH_naive_AVGQ=result["WRN-28-10-drop"]["SWITCH_naive"][avg_q],
                   WRN40_SWITCH_naive_AVGQ=result["WRN-40-10-drop"]["SWITCH_naive"][avg_q],
                   pyramidnet272_SWITCH_naive_MEDQ=result["pyramidnet272"]["SWITCH_naive"][med_q],
                   gdas_SWITCH_naive_MEDQ=result["gdas"]["SWITCH_naive"][med_q],
                   WRN28_SWITCH_naive_MEDQ=result["WRN-28-10-drop"]["SWITCH_naive"][med_q],
                   WRN40_SWITCH_naive_MEDQ=result["WRN-40-10-drop"]["SWITCH_naive"][med_q],

                   pyramidnet272_SWITCH_RGF_ASR=result["pyramidnet272"]["SWITCH_RGF"]["success_rate"],
                   gdas_SWITCH_RGF_ASR=result["gdas"]["SWITCH_RGF"]["success_rate"],
                   WRN28_SWITCH_RGF_ASR=result["WRN-28-10-drop"]["SWITCH_RGF"]["success_rate"],
                   WRN40_SWITCH_RGF_ASR=result["WRN-40-10-drop"]["SWITCH_RGF"]["success_rate"],
                   pyramidnet272_SWITCH_RGF_AVGQ=result["pyramidnet272"]["SWITCH_RGF"][avg_q],
                   gdas_SWITCH_RGF_AVGQ=result["gdas"]["SWITCH_RGF"][avg_q],
                   WRN28_SWITCH_RGF_AVGQ=result["WRN-28-10-drop"]["SWITCH_RGF"][avg_q],
                   WRN40_SWITCH_RGF_AVGQ=result["WRN-40-10-drop"]["SWITCH_RGF"][avg_q],
                   pyramidnet272_SWITCH_RGF_MEDQ=result["pyramidnet272"]["SWITCH_RGF"][med_q],
                   gdas_SWITCH_RGF_MEDQ=result["gdas"]["SWITCH_RGF"][med_q],
                   WRN28_SWITCH_RGF_MEDQ=result["WRN-28-10-drop"]["SWITCH_RGF"][med_q],
                   WRN40_SWITCH_RGF_MEDQ=result["WRN-40-10-drop"]["SWITCH_RGF"][med_q],

                   # pyramidnet272_SWITCH_other_ASR=result["pyramidnet272"]["SWITCH_other"]["success_rate"],
                   # gdas_SWITCH_other_ASR=result["gdas"]["SWITCH_other"]["success_rate"],
                   # WRN28_SWITCH_other_ASR=result["WRN-28-10-drop"]["SWITCH_other"]["success_rate"],
                   # WRN40_SWITCH_other_ASR=result["WRN-40-10-drop"]["SWITCH_other"]["success_rate"],
                   # pyramidnet272_SWITCH_other_AVGQ=result["pyramidnet272"]["SWITCH_other"][avg_q],
                   # gdas_SWITCH_other_AVGQ=result["gdas"]["SWITCH_other"][avg_q],
                   # WRN28_SWITCH_other_AVGQ=result["WRN-28-10-drop"]["SWITCH_other"][avg_q],
                   # WRN40_SWITCH_other_AVGQ=result["WRN-40-10-drop"]["SWITCH_other"][avg_q],
                   # pyramidnet272_SWITCH_other_MEDQ=result["pyramidnet272"]["SWITCH_other"][med_q],
                   # gdas_SWITCH_other_MEDQ=result["gdas"]["SWITCH_other"][med_q],
                   # WRN28_SWITCH_other_MEDQ=result["WRN-28-10-drop"]["SWITCH_other"][med_q],
                   # WRN40_SWITCH_other_MEDQ=result["WRN-40-10-drop"]["SWITCH_other"][med_q],
                   # 
                   # pyramidnet272_NO_SWITCH_rnd_ASR=result["pyramidnet272"]["NO_SWITCH_rnd"]["success_rate"],
                   # gdas_NO_SWITCH_rnd_ASR=result["gdas"]["NO_SWITCH_rnd"]["success_rate"],
                   # WRN28_NO_SWITCH_rnd_ASR=result["WRN-28-10-drop"]["NO_SWITCH_rnd"]["success_rate"],
                   # WRN40_NO_SWITCH_rnd_ASR=result["WRN-40-10-drop"]["NO_SWITCH_rnd"]["success_rate"],
                   # pyramidnet272_NO_SWITCH_rnd_AVGQ=result["pyramidnet272"]["NO_SWITCH_rnd"][avg_q],
                   # gdas_NO_SWITCH_rnd_AVGQ=result["gdas"]["NO_SWITCH_rnd"][avg_q],
                   # WRN28_NO_SWITCH_rnd_AVGQ=result["WRN-28-10-drop"]["NO_SWITCH_rnd"][avg_q],
                   # WRN40_NO_SWITCH_rnd_AVGQ=result["WRN-40-10-drop"]["NO_SWITCH_rnd"][avg_q],
                   # pyramidnet272_NO_SWITCH_rnd_MEDQ=result["pyramidnet272"]["NO_SWITCH_rnd"][med_q],
                   # gdas_NO_SWITCH_rnd_MEDQ=result["gdas"]["NO_SWITCH_rnd"][med_q],
                   # WRN28_NO_SWITCH_rnd_MEDQ=result["WRN-28-10-drop"]["NO_SWITCH_rnd"][med_q],
                   # WRN40_NO_SWITCH_rnd_MEDQ=result["WRN-40-10-drop"]["NO_SWITCH_rnd"][med_q],
                   )
              )
    else:
        print("""
                        & {norm_str} & RGF \cite{{2017RGF}} & {pyramidnet272_RGF_ASR}\% & {gdas_RGF_ASR}\% & {WRN28_RGF_ASR}\% & {WRN40_RGF_ASR}\% & {pyramidnet272_RGF_AVGQ} & {gdas_RGF_AVGQ} & {WRN28_RGF_AVGQ} & {WRN40_RGF_AVGQ} & {pyramidnet272_RGF_MEDQ} & {gdas_RGF_MEDQ} & {WRN28_RGF_MEDQ} & {WRN40_RGF_MEDQ} \\\\
                        & & P-RGF \cite{{cheng2019improving}} & {pyramidnet272_PRGF_ASR}\% & {gdas_PRGF_ASR}\% & {WRN28_PRGF_ASR}\% & {WRN40_PRGF_ASR}\% & {pyramidnet272_PRGF_AVGQ} & {gdas_PRGF_AVGQ} & {WRN28_PRGF_AVGQ} & {WRN40_PRGF_AVGQ} & {pyramidnet272_PRGF_MEDQ} & {gdas_PRGF_MEDQ} & {WRN28_PRGF_MEDQ} & {WRN40_PRGF_MEDQ} \\\\
                        & & Bandits  \cite{{ilyas2018prior}} & {pyramidnet272_Bandits_ASR}\% & {gdas_Bandits_ASR}\% & {WRN28_Bandits_ASR}\% & {WRN40_Bandits_ASR}\% & {pyramidnet272_Bandits_AVGQ} & {gdas_Bandits_AVGQ} & {WRN28_Bandits_AVGQ} & {WRN40_Bandits_AVGQ} & {pyramidnet272_Bandits_MEDQ} & {gdas_Bandits_MEDQ} & {WRN28_Bandits_MEDQ} & {WRN40_Bandits_MEDQ} \\\\
                        & & PPBA \cite{{li2020projection}} & {pyramidnet272_PPBA_ASR}\% & {gdas_PPBA_ASR}\% & {WRN28_PPBA_ASR}\% & {WRN40_PPBA_ASR}\% & {pyramidnet272_PPBA_AVGQ} & {gdas_PPBA_AVGQ} & {WRN28_PPBA_AVGQ} & {WRN40_PPBA_AVGQ} & {pyramidnet272_PPBA_MEDQ} & {gdas_PPBA_MEDQ} & {WRN28_PPBA_MEDQ} & {WRN40_PPBA_MEDQ} \\\\
                        & & SignHunter \cite{{al2019sign}} & {pyramidnet272_SignHunter_ASR}\% & {gdas_SignHunter_ASR}\% & {WRN28_SignHunter_ASR}\% & {WRN40_SignHunter_ASR}\% & {pyramidnet272_SignHunter_AVGQ} & {gdas_SignHunter_AVGQ} & {WRN28_SignHunter_AVGQ} & {WRN40_SignHunter_AVGQ} & {pyramidnet272_SignHunter_MEDQ} & {gdas_SignHunter_MEDQ} & {WRN28_SignHunter_MEDQ} & {WRN40_SignHunter_MEDQ} \\\\
                        & & Square Attack \cite{{ACFH2020square}} & {pyramidnet272_Square_ASR}\% & {gdas_Square_ASR}\% & {WRN28_Square_ASR}\% & {WRN40_Square_ASR}\% & {pyramidnet272_Square_AVGQ} & {gdas_Square_AVGQ} & {WRN28_Square_AVGQ} & {WRN40_Square_AVGQ} & {pyramidnet272_Square_MEDQ} & {gdas_Square_MEDQ} & {WRN28_Square_MEDQ} & {WRN40_Square_MEDQ} \\\\
                        & & NO SWITCH & {pyramidnet272_NO_SWITCH_ASR}\% & {gdas_NO_SWITCH_ASR}\% & {WRN28_NO_SWITCH_ASR}\% & {WRN40_NO_SWITCH_ASR}\% & {pyramidnet272_NO_SWITCH_AVGQ} & {gdas_NO_SWITCH_AVGQ} & {WRN28_NO_SWITCH_AVGQ} & {WRN40_NO_SWITCH_AVGQ} & {pyramidnet272_NO_SWITCH_MEDQ} & {gdas_NO_SWITCH_MEDQ} & {WRN28_NO_SWITCH_MEDQ} & {WRN40_NO_SWITCH_MEDQ} \\\\
                        & & SWITCH$_{{naive}}$ & {pyramidnet272_SWITCH_naive_ASR}\% & {gdas_SWITCH_naive_ASR}\% & {WRN28_SWITCH_naive_ASR}\% & {WRN40_SWITCH_naive_ASR}\% & {pyramidnet272_SWITCH_naive_AVGQ} & {gdas_SWITCH_naive_AVGQ} & {WRN28_SWITCH_naive_AVGQ} & {WRN40_SWITCH_naive_AVGQ} & {pyramidnet272_SWITCH_naive_MEDQ} & {gdas_SWITCH_naive_MEDQ} & {WRN28_SWITCH_naive_MEDQ} & {WRN40_SWITCH_naive_MEDQ} \\\\
                        & & SWITCH$_{{RGF}}$ & {pyramidnet272_SWITCH_RGF_ASR}\% & {gdas_SWITCH_RGF_ASR}\% & {WRN28_SWITCH_RGF_ASR}\% & {WRN40_SWITCH_RGF_ASR}\% & {pyramidnet272_SWITCH_RGF_AVGQ} & {gdas_SWITCH_RGF_AVGQ} & {WRN28_SWITCH_RGF_AVGQ} & {WRN40_SWITCH_RGF_AVGQ} & {pyramidnet272_SWITCH_RGF_MEDQ} & {gdas_SWITCH_RGF_MEDQ} & {WRN28_SWITCH_RGF_MEDQ} & {WRN40_SWITCH_RGF_MEDQ} \\\\
                """.format(norm_str="$\ell_\infty$" if norm == "linf" else "$\ell_2$",
                          pyramidnet272_RGF_ASR=result["pyramidnet272"]["RGF"]["success_rate"],
                           gdas_RGF_ASR=result["gdas"]["RGF"]["success_rate"],
                           WRN28_RGF_ASR=result["WRN-28-10-drop"]["RGF"]["success_rate"],
                           WRN40_RGF_ASR=result["WRN-40-10-drop"]["RGF"]["success_rate"],
                           pyramidnet272_RGF_AVGQ=result["pyramidnet272"]["RGF"][avg_q],
                           gdas_RGF_AVGQ=result["gdas"]["RGF"][avg_q],
                           WRN28_RGF_AVGQ=result["WRN-28-10-drop"]["RGF"][avg_q],
                           WRN40_RGF_AVGQ=result["WRN-40-10-drop"]["RGF"][avg_q],
                           pyramidnet272_RGF_MEDQ=result["pyramidnet272"]["RGF"][med_q],
                           gdas_RGF_MEDQ=result["gdas"]["RGF"][med_q],
                           WRN28_RGF_MEDQ=result["WRN-28-10-drop"]["RGF"][med_q],
                           WRN40_RGF_MEDQ=result["WRN-40-10-drop"]["RGF"][med_q],

                           pyramidnet272_PRGF_ASR=result["pyramidnet272"]["PRGF"]["success_rate"],
                           gdas_PRGF_ASR=result["gdas"]["PRGF"]["success_rate"],
                           WRN28_PRGF_ASR=result["WRN-28-10-drop"]["PRGF"]["success_rate"],
                           WRN40_PRGF_ASR=result["WRN-40-10-drop"]["PRGF"]["success_rate"],
                           pyramidnet272_PRGF_AVGQ=result["pyramidnet272"]["PRGF"][avg_q],
                           gdas_PRGF_AVGQ=result["gdas"]["PRGF"][avg_q],
                           WRN28_PRGF_AVGQ=result["WRN-28-10-drop"]["PRGF"][avg_q],
                           WRN40_PRGF_AVGQ=result["WRN-40-10-drop"]["PRGF"][avg_q],
                           pyramidnet272_PRGF_MEDQ=result["pyramidnet272"]["PRGF"][med_q],
                           gdas_PRGF_MEDQ=result["gdas"]["PRGF"][med_q],
                           WRN28_PRGF_MEDQ=result["WRN-28-10-drop"]["PRGF"][med_q],
                           WRN40_PRGF_MEDQ=result["WRN-40-10-drop"]["PRGF"][med_q],

                           pyramidnet272_Bandits_ASR=result["pyramidnet272"]["Bandits"]["success_rate"],
                           gdas_Bandits_ASR=result["gdas"]["Bandits"]["success_rate"],
                           WRN28_Bandits_ASR=result["WRN-28-10-drop"]["Bandits"]["success_rate"],
                           WRN40_Bandits_ASR=result["WRN-40-10-drop"]["Bandits"]["success_rate"],
                           pyramidnet272_Bandits_AVGQ=result["pyramidnet272"]["Bandits"][avg_q],
                           gdas_Bandits_AVGQ=result["gdas"]["Bandits"][avg_q],
                           WRN28_Bandits_AVGQ=result["WRN-28-10-drop"]["Bandits"][avg_q],
                           WRN40_Bandits_AVGQ=result["WRN-40-10-drop"]["Bandits"][avg_q],
                           pyramidnet272_Bandits_MEDQ=result["pyramidnet272"]["Bandits"][med_q],
                           gdas_Bandits_MEDQ=result["gdas"]["Bandits"][med_q],
                           WRN28_Bandits_MEDQ=result["WRN-28-10-drop"]["Bandits"][med_q],
                           WRN40_Bandits_MEDQ=result["WRN-40-10-drop"]["Bandits"][med_q],

                           pyramidnet272_PPBA_ASR=result["pyramidnet272"]["PPBA"]["success_rate"],
                           gdas_PPBA_ASR=result["gdas"]["PPBA"]["success_rate"],
                           WRN28_PPBA_ASR=result["WRN-28-10-drop"]["PPBA"]["success_rate"],
                           WRN40_PPBA_ASR=result["WRN-40-10-drop"]["PPBA"]["success_rate"],
                           pyramidnet272_PPBA_AVGQ=result["pyramidnet272"]["PPBA"][avg_q],
                           gdas_PPBA_AVGQ=result["gdas"]["PPBA"][avg_q],
                           WRN28_PPBA_AVGQ=result["WRN-28-10-drop"]["PPBA"][avg_q],
                           WRN40_PPBA_AVGQ=result["WRN-40-10-drop"]["PPBA"][avg_q],
                           pyramidnet272_PPBA_MEDQ=result["pyramidnet272"]["PPBA"][med_q],
                           gdas_PPBA_MEDQ=result["gdas"]["PPBA"][med_q],
                           WRN28_PPBA_MEDQ=result["WRN-28-10-drop"]["PPBA"][med_q],
                           WRN40_PPBA_MEDQ=result["WRN-40-10-drop"]["PPBA"][med_q],

                           # pyramidnet272_SimBA_ASR=result["pyramidnet272"]["SimBA"]["success_rate"],
                           # gdas_SimBA_ASR=result["gdas"]["SimBA"]["success_rate"],
                           # WRN28_SimBA_ASR=result["WRN-28-10-drop"]["SimBA"]["success_rate"],
                           # WRN40_SimBA_ASR=result["WRN-40-10-drop"]["SimBA"]["success_rate"],
                           # pyramidnet272_SimBA_AVGQ=result["pyramidnet272"]["SimBA"][avg_q],
                           # gdas_SimBA_AVGQ=result["gdas"]["SimBA"][avg_q],
                           # WRN28_SimBA_AVGQ=result["WRN-28-10-drop"]["SimBA"][avg_q],
                           # WRN40_SimBA_AVGQ=result["WRN-40-10-drop"]["SimBA"][avg_q],
                           # pyramidnet272_SimBA_MEDQ=result["pyramidnet272"]["SimBA"][med_q],
                           # gdas_SimBA_MEDQ=result["gdas"]["SimBA"][med_q],
                           # WRN28_SimBA_MEDQ=result["WRN-28-10-drop"]["SimBA"][med_q],
                           # WRN40_SimBA_MEDQ=result["WRN-40-10-drop"]["SimBA"][med_q],

                           pyramidnet272_SignHunter_ASR=result["pyramidnet272"]["SignHunter"]["success_rate"],
                           gdas_SignHunter_ASR=result["gdas"]["SignHunter"]["success_rate"],
                           WRN28_SignHunter_ASR=result["WRN-28-10-drop"]["SignHunter"]["success_rate"],
                           WRN40_SignHunter_ASR=result["WRN-40-10-drop"]["SignHunter"]["success_rate"],
                           pyramidnet272_SignHunter_AVGQ=result["pyramidnet272"]["SignHunter"][avg_q],
                           gdas_SignHunter_AVGQ=result["gdas"]["SignHunter"][avg_q],
                           WRN28_SignHunter_AVGQ=result["WRN-28-10-drop"]["SignHunter"][avg_q],
                           WRN40_SignHunter_AVGQ=result["WRN-40-10-drop"]["SignHunter"][avg_q],
                           pyramidnet272_SignHunter_MEDQ=result["pyramidnet272"]["SignHunter"][med_q],
                           gdas_SignHunter_MEDQ=result["gdas"]["SignHunter"][med_q],
                           WRN28_SignHunter_MEDQ=result["WRN-28-10-drop"]["SignHunter"][med_q],
                           WRN40_SignHunter_MEDQ=result["WRN-40-10-drop"]["SignHunter"][med_q],

                           pyramidnet272_Square_ASR=result["pyramidnet272"]["Square"]["success_rate"],
                           gdas_Square_ASR=result["gdas"]["Square"]["success_rate"],
                           WRN28_Square_ASR=result["WRN-28-10-drop"]["Square"]["success_rate"],
                           WRN40_Square_ASR=result["WRN-40-10-drop"]["Square"]["success_rate"],
                           pyramidnet272_Square_AVGQ=result["pyramidnet272"]["Square"][avg_q],
                           gdas_Square_AVGQ=result["gdas"]["Square"][avg_q],
                           WRN28_Square_AVGQ=result["WRN-28-10-drop"]["Square"][avg_q],
                           WRN40_Square_AVGQ=result["WRN-40-10-drop"]["Square"][avg_q],
                           pyramidnet272_Square_MEDQ=result["pyramidnet272"]["Square"][med_q],
                           gdas_Square_MEDQ=result["gdas"]["Square"][med_q],
                           WRN28_Square_MEDQ=result["WRN-28-10-drop"]["Square"][med_q],
                           WRN40_Square_MEDQ=result["WRN-40-10-drop"]["Square"][med_q],

                           pyramidnet272_NO_SWITCH_ASR=result["pyramidnet272"]["NO_SWITCH"]["success_rate"],
                           gdas_NO_SWITCH_ASR=result["gdas"]["NO_SWITCH"]["success_rate"],
                           WRN28_NO_SWITCH_ASR=result["WRN-28-10-drop"]["NO_SWITCH"]["success_rate"],
                           WRN40_NO_SWITCH_ASR=result["WRN-40-10-drop"]["NO_SWITCH"]["success_rate"],
                           pyramidnet272_NO_SWITCH_AVGQ=result["pyramidnet272"]["NO_SWITCH"][avg_q],
                           gdas_NO_SWITCH_AVGQ=result["gdas"]["NO_SWITCH"][avg_q],
                           WRN28_NO_SWITCH_AVGQ=result["WRN-28-10-drop"]["NO_SWITCH"][avg_q],
                           WRN40_NO_SWITCH_AVGQ=result["WRN-40-10-drop"]["NO_SWITCH"][avg_q],
                           pyramidnet272_NO_SWITCH_MEDQ=result["pyramidnet272"]["NO_SWITCH"][med_q],
                           gdas_NO_SWITCH_MEDQ=result["gdas"]["NO_SWITCH"][med_q],
                           WRN28_NO_SWITCH_MEDQ=result["WRN-28-10-drop"]["NO_SWITCH"][med_q],
                           WRN40_NO_SWITCH_MEDQ=result["WRN-40-10-drop"]["NO_SWITCH"][med_q],

                           pyramidnet272_SWITCH_naive_ASR=result["pyramidnet272"]["SWITCH_naive"]["success_rate"],
                           gdas_SWITCH_naive_ASR=result["gdas"]["SWITCH_naive"]["success_rate"],
                           WRN28_SWITCH_naive_ASR=result["WRN-28-10-drop"]["SWITCH_naive"]["success_rate"],
                           WRN40_SWITCH_naive_ASR=result["WRN-40-10-drop"]["SWITCH_naive"]["success_rate"],
                           pyramidnet272_SWITCH_naive_AVGQ=result["pyramidnet272"]["SWITCH_naive"][avg_q],
                           gdas_SWITCH_naive_AVGQ=result["gdas"]["SWITCH_naive"][avg_q],
                           WRN28_SWITCH_naive_AVGQ=result["WRN-28-10-drop"]["SWITCH_naive"][avg_q],
                           WRN40_SWITCH_naive_AVGQ=result["WRN-40-10-drop"]["SWITCH_naive"][avg_q],
                           pyramidnet272_SWITCH_naive_MEDQ=result["pyramidnet272"]["SWITCH_naive"][med_q],
                           gdas_SWITCH_naive_MEDQ=result["gdas"]["SWITCH_naive"][med_q],
                           WRN28_SWITCH_naive_MEDQ=result["WRN-28-10-drop"]["SWITCH_naive"][med_q],
                           WRN40_SWITCH_naive_MEDQ=result["WRN-40-10-drop"]["SWITCH_naive"][med_q],

                           pyramidnet272_SWITCH_RGF_ASR=result["pyramidnet272"]["SWITCH_RGF"]["success_rate"],
                           gdas_SWITCH_RGF_ASR=result["gdas"]["SWITCH_RGF"]["success_rate"],
                           WRN28_SWITCH_RGF_ASR=result["WRN-28-10-drop"]["SWITCH_RGF"]["success_rate"],
                           WRN40_SWITCH_RGF_ASR=result["WRN-40-10-drop"]["SWITCH_RGF"]["success_rate"],
                           pyramidnet272_SWITCH_RGF_AVGQ=result["pyramidnet272"]["SWITCH_RGF"][avg_q],
                           gdas_SWITCH_RGF_AVGQ=result["gdas"]["SWITCH_RGF"][avg_q],
                           WRN28_SWITCH_RGF_AVGQ=result["WRN-28-10-drop"]["SWITCH_RGF"][avg_q],
                           WRN40_SWITCH_RGF_AVGQ=result["WRN-40-10-drop"]["SWITCH_RGF"][avg_q],
                           pyramidnet272_SWITCH_RGF_MEDQ=result["pyramidnet272"]["SWITCH_RGF"][med_q],
                           gdas_SWITCH_RGF_MEDQ=result["gdas"]["SWITCH_RGF"][med_q],
                           WRN28_SWITCH_RGF_MEDQ=result["WRN-28-10-drop"]["SWITCH_RGF"][med_q],
                           WRN40_SWITCH_RGF_MEDQ=result["WRN-40-10-drop"]["SWITCH_RGF"][med_q],
                           # pyramidnet272_SWITCH_other_ASR=result["pyramidnet272"]["SWITCH_other"]["success_rate"],
                           # gdas_SWITCH_other_ASR=result["gdas"]["SWITCH_other"]["success_rate"],
                           # WRN28_SWITCH_other_ASR=result["WRN-28-10-drop"]["SWITCH_other"]["success_rate"],
                           # WRN40_SWITCH_other_ASR=result["WRN-40-10-drop"]["SWITCH_other"]["success_rate"],
                           # pyramidnet272_SWITCH_other_AVGQ=result["pyramidnet272"]["SWITCH_other"][avg_q],
                           # gdas_SWITCH_other_AVGQ=result["gdas"]["SWITCH_other"][avg_q],
                           # WRN28_SWITCH_other_AVGQ=result["WRN-28-10-drop"]["SWITCH_other"][avg_q],
                           # WRN40_SWITCH_other_AVGQ=result["WRN-40-10-drop"]["SWITCH_other"][avg_q],
                           # pyramidnet272_SWITCH_other_MEDQ=result["pyramidnet272"]["SWITCH_other"][med_q],
                           # gdas_SWITCH_other_MEDQ=result["gdas"]["SWITCH_other"][med_q],
                           # WRN28_SWITCH_other_MEDQ=result["WRN-28-10-drop"]["SWITCH_other"][med_q],
                           # WRN40_SWITCH_other_MEDQ=result["WRN-40-10-drop"]["SWITCH_other"][med_q],
                           #
                           # pyramidnet272_NO_SWITCH_rnd_ASR=result["pyramidnet272"]["NO_SWITCH_rnd"]["success_rate"],
                           # gdas_NO_SWITCH_rnd_ASR=result["gdas"]["NO_SWITCH_rnd"]["success_rate"],
                           # WRN28_NO_SWITCH_rnd_ASR=result["WRN-28-10-drop"]["NO_SWITCH_rnd"]["success_rate"],
                           # WRN40_NO_SWITCH_rnd_ASR=result["WRN-40-10-drop"]["NO_SWITCH_rnd"]["success_rate"],
                           # WRN40_NO_SWITCH_rnd_ASR=result["WRN-40-10-drop"]["NO_SWITCH_rnd"]["success_rate"],
                           # pyramidnet272_NO_SWITCH_rnd_AVGQ=result["pyramidnet272"]["NO_SWITCH_rnd"][avg_q],
                           # gdas_NO_SWITCH_rnd_AVGQ=result["gdas"]["NO_SWITCH_rnd"][avg_q],
                           # WRN28_NO_SWITCH_rnd_AVGQ=result["WRN-28-10-drop"]["NO_SWITCH_rnd"][avg_q],
                           # WRN40_NO_SWITCH_rnd_AVGQ=result["WRN-40-10-drop"]["NO_SWITCH_rnd"][avg_q],
                           # pyramidnet272_NO_SWITCH_rnd_MEDQ=result["pyramidnet272"]["NO_SWITCH_rnd"][med_q],
                           # gdas_NO_SWITCH_rnd_MEDQ=result["gdas"]["NO_SWITCH_rnd"][med_q],
                           # WRN28_NO_SWITCH_rnd_MEDQ=result["WRN-28-10-drop"]["NO_SWITCH_rnd"][med_q],
                           # WRN40_NO_SWITCH_rnd_MEDQ=result["WRN-40-10-drop"]["NO_SWITCH_rnd"][med_q],
                           )
              )
if __name__ == "__main__":
    dataset = "TinyImageNet"
    norm = "linf"
    targeted = False
    if "CIFAR" in dataset:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        archs = ["densenet121", "resnext32_4", "resnext64_4"]
    result_archs = {}
    for arch in archs:
        result = fetch_all_json_content_given_contraint(dataset, norm, targeted, arch)
        result_archs[arch] = result
    if "CIFAR" in dataset:
        draw_tables_for_CIFAR(norm, result_archs)
    elif "TinyImageNet" in dataset:
        draw_tables_for_TinyImageNet(norm, result_archs)
    print("THIS IS {} {} {}".format(dataset, norm, "untargeted" if not targeted else "targeted"))
