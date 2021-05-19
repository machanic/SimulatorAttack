from collections import defaultdict

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
                        #"SWITCH_rnd_save":'SWITCH_other',
                        "SWITCH_neg_save":'SWITCH_naive',
                        "SWITCH_RGF":'SWITCH_RGF',
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
        path = "{method}_on_defensive_model-{dataset}-cw_loss-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                        norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NES":
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
    elif method == "PPBA_attack":
        path = "PPBA_attack_on_defensive_model-{dataset}-{norm}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "parsimonious_attack":
        path = "parsimonious_attack-{norm}_on_defensive_model-{dataset}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SimBA_DCT_attack":
        path = "SimBA_DCT_attack_on_defensive_model-{dataset}-{norm}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "sign_hunter_attack":
        path = "sign_hunter_attack_on_defensive_model-{dataset}-{norm}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "square_attack":
        path = "square_attack_on_defensive_model-{dataset}-{norm}-{target_str}".format(dataset=dataset, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SWITCH_neg_save":
        path = "SWITCH_neg_save_on_defensive_model-{dataset}-{loss}_lr_{lr}-loss-{norm}-{target_str}".format(dataset=dataset, lr=lr, loss="cw" if not targeted else "xent", norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SWITCH_rnd_save":
        path = "SWITCH_rnd_save_on_defensive_model-{dataset}-{loss}_lr_{lr}-loss-{norm}-{target_str}".format(dataset=dataset, lr=lr, loss="cw" if not targeted else "xent", norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NO_SWITCH":
        # NO_SWITCH_on_defensive_model-CIFAR-10-cw_lr_0.01-loss-linf-untargeted
        path = "NO_SWITCH_on_defensive_model-{dataset}-{loss}_lr_{lr}-loss-{norm}-{target_str}".format(dataset=dataset, loss="cw" if not targeted else "xent", lr=lr, norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NO_SWITCH_rnd":
        # NO_SWITCH_rnd_using_resnet-110,densenet-bc-100-12_on_defensive_model-CIFAR-100-lr_0.01_cw-loss-linf-untargeted
        path = "NO_SWITCH_rnd_on_defensive_model_using_{archs}-{dataset}-lr_{lr}_{loss}-loss-{norm}-{target_str}".format(
                                                                            archs="resnet-110,densenet-bc-100-12" if "CIFAR" in dataset else "resnet101,resnet152", dataset=dataset,
                                                                            lr=lr,
                                                                            loss="cw" if not targeted else "xent",
                                                                            norm=norm,
                                                                            target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "SWITCH_RGF":
        if dataset.startswith("CIFAR"):
            path = "SWITCH_RGF_on_defensive_model-resnet-110-{dataset}-{loss}-loss-{norm}-{target_str}".format(dataset=dataset,
                                                                                                               loss="cw" if not targeted else "xent",
                                                                                                               norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
        elif dataset == "TinyImageNet":
            path = "SWITCH_RGF_on_defensive_model-resnet101-{dataset}-{loss}-loss-{norm}-{target_str}".format(
                dataset=dataset,
                loss="cw" if not targeted else "xent",
                norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
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

def fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, defense_models):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result = defaultdict(dict)
    for method, folder in folder_list.items():
        for defense_model in defense_models:
            file_path = folder + "/{}_{}_result.json".format(arch, defense_model)
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
            # assert epsilon == json_content["args"]["epsilon"], "eps is {} in {}".format(json_content["args"]["epsilon"], file_path)
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
            result[defense_model][method] = {"success_rate":success_rate, "avg_query_over_successful_samples":avg_query_over_successful_samples,
                              "median_query_over_successful_samples": median_query_over_successful_samples,
                            "avg_query_over_all_samples": avg_query_over_all_samples, "median_query_over_all_samples":median_query_over_all_samples}
    return result

def draw_tables_for_TinyImageNet(norm, archs_result):
    result = archs_result
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    if norm == "linf":
        print("""
                                RGF & {ComDefend_RGF_ASR}\% & {FeatureDistillation_RGF_ASR}\% & {jpeg_RGF_ASR}\% & {ComDefend_RGF_AVGQ} & {FeatureDistillation_RGF_AVGQ} & {jpeg_RGF_AVGQ}   \\\\
                                P-RGF & {ComDefend_PRGF_ASR}\% & {FeatureDistillation_PRGF_ASR}\%  & {jpeg_PRGF_ASR}\%  & {ComDefend_PRGF_AVGQ} & {FeatureDistillation_PRGF_AVGQ} & {jpeg_PRGF_AVGQ}  \\\\
                                Bandits & {ComDefend_Bandits_ASR}\% & {FeatureDistillation_Bandits_ASR}\% & {jpeg_Bandits_ASR}\% & {ComDefend_Bandits_AVGQ} & {FeatureDistillation_Bandits_AVGQ} &  {jpeg_Bandits_AVGQ}  \\\\
                                PPBA & {ComDefend_PPBA_ASR}\% & {FeatureDistillation_PPBA_ASR}\% &  {jpeg_PPBA_ASR}\%  & {ComDefend_PPBA_AVGQ} & {FeatureDistillation_PPBA_AVGQ} &  {jpeg_PPBA_AVGQ}  \\\\
                                Parsimonious & {ComDefend_Parsimonious_ASR}\% & {FeatureDistillation_Parsimonious_ASR}\% & {jpeg_Parsimonious_ASR}\% &{ComDefend_Parsimonious_AVGQ} & {FeatureDistillation_Parsimonious_AVGQ} & {jpeg_Parsimonious_AVGQ}  \\\\
                                SignHunter & {ComDefend_SignHunter_ASR}\% & {FeatureDistillation_SignHunter_ASR}\% & {jpeg_SignHunter_ASR}\% & {ComDefend_SignHunter_AVGQ} & {FeatureDistillation_SignHunter_AVGQ} & {jpeg_SignHunter_AVGQ}  \\\\
                                Square Attack & {ComDefend_Square_ASR}\% & {FeatureDistillation_Square_ASR}\%& {jpeg_Square_ASR}\% & {ComDefend_Square_AVGQ} & {FeatureDistillation_Square_AVGQ} & {jpeg_Square_AVGQ} \\\\
                                NO SWITCH & {ComDefend_NO_SWITCH_ASR}\% & {FeatureDistillation_NO_SWITCH_ASR}\% &  {jpeg_NO_SWITCH_ASR}\% &{ComDefend_NO_SWITCH_AVGQ} & {FeatureDistillation_NO_SWITCH_AVGQ} &  {jpeg_NO_SWITCH_AVGQ}  \\\\
                                SWITCH$_\\text{{naive}}$ & {ComDefend_SWITCH_naive_ASR}\% & {FeatureDistillation_SWITCH_naive_ASR}\% & {jpeg_SWITCH_naive_ASR}\% & {ComDefend_SWITCH_naive_AVGQ} & {FeatureDistillation_SWITCH_naive_AVGQ} & {jpeg_SWITCH_naive_AVGQ} \\\\
                                SWITCH$_\\text{{RGF}}$ & {ComDefend_SWITCH_RGF_ASR}\% & {FeatureDistillation_SWITCH_RGF_ASR}\% & {jpeg_SWITCH_RGF_ASR}\% & {ComDefend_SWITCH_RGF_AVGQ} & {FeatureDistillation_SWITCH_RGF_AVGQ} & {jpeg_SWITCH_RGF_AVGQ} \\\\
                                        """.format(
            ComDefend_RGF_ASR=result["com_defend"]["RGF"]["success_rate"],
            FeatureDistillation_RGF_ASR=result["feature_distillation"]["RGF"]["success_rate"],
            ComDefend_RGF_AVGQ=result["com_defend"]["RGF"][avg_q],
            FeatureDistillation_RGF_AVGQ=result["feature_distillation"]["RGF"][avg_q],
            jpeg_RGF_ASR=result["jpeg"]["RGF"]["success_rate"],
            jpeg_RGF_AVGQ=result["jpeg"]["RGF"][avg_q],
            # ComDefend_RGF_MEDQ=result["com_defend"]["RGF"][med_q],
            # FeatureDistillation_RGF_MEDQ=result["feature_distillation"]["RGF"][med_q],

            ComDefend_PRGF_ASR=result["com_defend"]["PRGF"]["success_rate"],
            FeatureDistillation_PRGF_ASR=result["feature_distillation"]["PRGF"]["success_rate"],

            ComDefend_PRGF_AVGQ=result["com_defend"]["PRGF"][avg_q],
            FeatureDistillation_PRGF_AVGQ=result["feature_distillation"]["PRGF"][avg_q],
            jpeg_PRGF_ASR=result["jpeg"]["PRGF"]["success_rate"],
            jpeg_PRGF_AVGQ=result["jpeg"]["PRGF"][avg_q],

            # ComDefend_PRGF_MEDQ=result["com_defend"]["PRGF"][med_q],
            # FeatureDistillation_PRGF_MEDQ=result["feature_distillation"]["PRGF"][med_q],

            ComDefend_Bandits_ASR=result["com_defend"]["Bandits"]["success_rate"],
            FeatureDistillation_Bandits_ASR=result["feature_distillation"]["Bandits"]["success_rate"],

            ComDefend_Bandits_AVGQ=result["com_defend"]["Bandits"][avg_q],
            FeatureDistillation_Bandits_AVGQ=result["feature_distillation"]["Bandits"][avg_q],
            jpeg_Bandits_ASR=result["jpeg"]["Bandits"]["success_rate"],
            jpeg_Bandits_AVGQ=result["jpeg"]["Bandits"][avg_q],

            # ComDefend_Bandits_MEDQ=result["com_defend"]["Bandits"][med_q],
            # FeatureDistillation_Bandits_MEDQ=result["feature_distillation"]["Bandits"][med_q],

            ComDefend_PPBA_ASR=result["com_defend"]["PPBA"]["success_rate"],
            FeatureDistillation_PPBA_ASR=result["feature_distillation"]["PPBA"]["success_rate"],

            ComDefend_PPBA_AVGQ=result["com_defend"]["PPBA"][avg_q],
            FeatureDistillation_PPBA_AVGQ=result["feature_distillation"]["PPBA"][avg_q],
            jpeg_PPBA_ASR=result["jpeg"]["PPBA"]["success_rate"],
            jpeg_PPBA_AVGQ=result["jpeg"]["PPBA"][avg_q],
            # ComDefend_PPBA_MEDQ=result["com_defend"]["PPBA"][med_q],
            # FeatureDistillation_PPBA_MEDQ=result["feature_distillation"]["PPBA"][med_q],

            ComDefend_Parsimonious_ASR=result["com_defend"]["Parsimonious"]["success_rate"],
            FeatureDistillation_Parsimonious_ASR=result["feature_distillation"]["Parsimonious"]["success_rate"],

            ComDefend_Parsimonious_AVGQ=result["com_defend"]["Parsimonious"][avg_q],
            FeatureDistillation_Parsimonious_AVGQ=result["feature_distillation"]["Parsimonious"][avg_q],
            jpeg_Parsimonious_ASR=result["jpeg"]["Parsimonious"]["success_rate"],
            jpeg_Parsimonious_AVGQ=result["jpeg"]["Parsimonious"][avg_q],
            # ComDefend_Parsimonious_MEDQ=result["com_defend"]["Parsimonious"][med_q],
            # FeatureDistillation_Parsimonious_MEDQ=result["feature_distillation"]["Parsimonious"][med_q],

            ComDefend_SignHunter_ASR=result["com_defend"]["SignHunter"]["success_rate"],
            FeatureDistillation_SignHunter_ASR=result["feature_distillation"]["SignHunter"]["success_rate"],

            ComDefend_SignHunter_AVGQ=result["com_defend"]["SignHunter"][avg_q],
            FeatureDistillation_SignHunter_AVGQ=result["feature_distillation"]["SignHunter"][avg_q],
            jpeg_SignHunter_ASR=result["jpeg"]["SignHunter"]["success_rate"],
            jpeg_SignHunter_AVGQ=result["jpeg"]["SignHunter"][avg_q],

            # ComDefend_SignHunter_MEDQ=result["com_defend"]["SignHunter"][med_q],
            # FeatureDistillation_SignHunter_MEDQ=result["feature_distillation"]["SignHunter"][med_q],

            ComDefend_Square_ASR=result["com_defend"]["Square"]["success_rate"],
            FeatureDistillation_Square_ASR=result["feature_distillation"]["Square"]["success_rate"],
            ComDefend_Square_AVGQ=result["com_defend"]["Square"][avg_q],
            FeatureDistillation_Square_AVGQ=result["feature_distillation"]["Square"][avg_q],
            jpeg_Square_ASR=result["jpeg"]["Square"]["success_rate"],
            jpeg_Square_AVGQ=result["jpeg"]["Square"][avg_q],

            # ComDefend_Square_MEDQ=result["com_defend"]["Square"][med_q],
            # FeatureDistillation_Square_MEDQ=result["feature_distillation"]["Square"][med_q],

            ComDefend_NO_SWITCH_ASR=result["com_defend"]["NO_SWITCH"]["success_rate"],
            FeatureDistillation_NO_SWITCH_ASR=result["feature_distillation"]["NO_SWITCH"]["success_rate"],

            ComDefend_NO_SWITCH_AVGQ=result["com_defend"]["NO_SWITCH"][avg_q],
            FeatureDistillation_NO_SWITCH_AVGQ=result["feature_distillation"]["NO_SWITCH"][avg_q],
            jpeg_NO_SWITCH_ASR=result["jpeg"]["NO_SWITCH"]["success_rate"],
            jpeg_NO_SWITCH_AVGQ=result["jpeg"]["NO_SWITCH"][avg_q],
            # ComDefend_NO_SWITCH_MEDQ=result["com_defend"]["NO_SWITCH"][med_q],
            # FeatureDistillation_NO_SWITCH_MEDQ=result["feature_distillation"]["NO_SWITCH"][med_q],

            ComDefend_SWITCH_naive_ASR=result["com_defend"]["SWITCH_naive"]["success_rate"],
            FeatureDistillation_SWITCH_naive_ASR=result["feature_distillation"]["SWITCH_naive"]["success_rate"],

            ComDefend_SWITCH_naive_AVGQ=result["com_defend"]["SWITCH_naive"][avg_q],
            FeatureDistillation_SWITCH_naive_AVGQ=result["feature_distillation"]["SWITCH_naive"][avg_q],
            jpeg_SWITCH_naive_ASR=result["jpeg"]["SWITCH_naive"]["success_rate"],
            jpeg_SWITCH_naive_AVGQ=result["jpeg"]["SWITCH_naive"][avg_q],

            # ComDefend_SWITCH_naive_MEDQ=result["com_defend"]["SWITCH_naive"][med_q],
            # FeatureDistillation_SWITCH_naive_MEDQ=result["feature_distillation"]["SWITCH_naive"][med_q],

            ComDefend_SWITCH_RGF_ASR=result["com_defend"]["SWITCH_RGF"]["success_rate"],
            FeatureDistillation_SWITCH_RGF_ASR=result["feature_distillation"]["SWITCH_RGF"]["success_rate"],

            ComDefend_SWITCH_RGF_AVGQ=result["com_defend"]["SWITCH_RGF"][avg_q],
            FeatureDistillation_SWITCH_RGF_AVGQ=result["feature_distillation"]["SWITCH_RGF"][avg_q],
            jpeg_SWITCH_RGF_ASR=result["jpeg"]["SWITCH_RGF"]["success_rate"],
            jpeg_SWITCH_RGF_AVGQ=result["jpeg"]["SWITCH_RGF"][avg_q],

            # ComDefend_SWITCH_RGF_MEDQ=result["com_defend"]["SWITCH_RGF"][med_q],
            # FeatureDistillation_SWITCH_RGF_MEDQ=result["feature_distillation"]["SWITCH_RGF"][med_q],

            # ComDefend_NO_SWITCH_rnd_ASR=result["com_defend"]["NO_SWITCH_rnd"]["success_rate"],
            # FeatureDistillation_NO_SWITCH_rnd_ASR=result["feature_distillation"]["NO_SWITCH_rnd"]["success_rate"],
            #
            # ComDefend_NO_SWITCH_rnd_AVGQ=result["com_defend"]["NO_SWITCH_rnd"][avg_q],
            # FeatureDistillation_NO_SWITCH_rnd_AVGQ=result["feature_distillation"]["NO_SWITCH_rnd"][avg_q],
            #
            # ComDefend_NO_SWITCH_rnd_MEDQ=result["com_defend"]["NO_SWITCH_rnd"][med_q],
            # FeatureDistillation_NO_SWITCH_rnd_MEDQ=result["feature_distillation"]["NO_SWITCH_rnd"][med_q],

            # ComDefend_SWITCH_other_ASR=result["com_defend"]["SWITCH_other"]["success_rate"],
            # FeatureDistillation_SWITCH_other_ASR=result["feature_distillation"]["SWITCH_other"]["success_rate"],

            # ComDefend_SWITCH_other_AVGQ=result["com_defend"]["SWITCH_other"][avg_q],
            # FeatureDistillation_SWITCH_other_AVGQ=result["feature_distillation"]["SWITCH_other"][avg_q],
            #
            # ComDefend_SWITCH_other_MEDQ=result["com_defend"]["SWITCH_other"][med_q],
            # FeatureDistillation_SWITCH_other_MEDQ=result["feature_distillation"]["SWITCH_other"][med_q],
        )
        )
    else:
        print("""
        RGF & {D121_RGF_ASR}\% & {R32_RGF_ASR}\% & {R64_RGF_ASR}\% & {D121_RGF_AVGQ} & {R32_RGF_AVGQ} & {R64_RGF_AVGQ} & {D121_RGF_MEDQ} & {R32_RGF_MEDQ} & {R64_RGF_MEDQ} \\\\
        P-RGF & {D121_PRGF_ASR}\% & {R32_PRGF_ASR}\% & {R64_PRGF_ASR}\% & {D121_PRGF_AVGQ} & {R32_PRGF_AVGQ} & {R64_PRGF_AVGQ} & {D121_PRGF_MEDQ} & {R32_PRGF_MEDQ} & {R64_PRGF_MEDQ} \\\\
        Bandits & {D121_Bandits_ASR}\% & {R32_Bandits_ASR}\% & {R64_Bandits_ASR}\% & {D121_Bandits_AVGQ} & {R32_Bandits_AVGQ} & {R64_Bandits_AVGQ} & {D121_Bandits_MEDQ} & {R32_Bandits_MEDQ} & {R64_Bandits_MEDQ} \\\\
        PPBA & {D121_PPBA_ASR}\% & {R32_PPBA_ASR}\% & {R64_PPBA_ASR}\% & {D121_PPBA_AVGQ} & {R32_PPBA_AVGQ} & {R64_PPBA_AVGQ} & {D121_PPBA_MEDQ} & {R32_PPBA_MEDQ} & {R64_PPBA_MEDQ} \\\\
        SignHunter & {D121_SignHunter_ASR}\% & {R32_SignHunter_ASR}\% & {R64_SignHunter_ASR}\% & {D121_SignHunter_AVGQ} & {R32_SignHunter_AVGQ} & {R64_SignHunter_AVGQ} & {D121_SignHunter_MEDQ} & {R32_SignHunter_MEDQ} & {R64_SignHunter_MEDQ} \\\\
        Square Attack & {D121_Square_ASR}\% & {R32_Square_ASR}\% & {R64_Square_ASR}\% & {D121_Square_AVGQ} & {R32_Square_AVGQ} & {R64_Square_AVGQ} & {D121_Square_MEDQ} & {R32_Square_MEDQ} & {R64_Square_MEDQ} \\\\
        NO SWITCH & {D121_NO_SWITCH_ASR}\% & {R32_NO_SWITCH_ASR}\% & {R64_NO_SWITCH_ASR}\% & {D121_NO_SWITCH_AVGQ} & {R32_NO_SWITCH_AVGQ} & {R64_NO_SWITCH_AVGQ} & {D121_NO_SWITCH_MEDQ} & {R32_NO_SWITCH_MEDQ} & {R64_NO_SWITCH_MEDQ} \\\\
        SWITCH_neg & {D121_SWITCH_neg_ASR}\% & {R32_SWITCH_neg_ASR}\% & {R64_SWITCH_neg_ASR}\% & {D121_SWITCH_neg_AVGQ} & {R32_SWITCH_neg_AVGQ} & {R64_SWITCH_neg_AVGQ} & {D121_SWITCH_neg_MEDQ} & {R32_SWITCH_neg_MEDQ} & {R64_SWITCH_neg_MEDQ} \\\\
        NO SWITCH_rnd & {D121_NO_SWITCH_rnd_ASR}\% & {R32_NO_SWITCH_rnd_ASR}\% & {R64_NO_SWITCH_rnd_ASR}\% & {D121_NO_SWITCH_rnd_AVGQ} & {R32_NO_SWITCH_rnd_AVGQ} & {R64_NO_SWITCH_rnd_AVGQ} & {D121_NO_SWITCH_rnd_MEDQ} & {R32_NO_SWITCH_rnd_MEDQ} & {R64_NO_SWITCH_rnd_MEDQ} \\\\
        SWITCH_other & {D121_SWITCH_other_ASR}\% & {R32_SWITCH_other_ASR}\% & {R64_SWITCH_other_ASR}\% & {D121_SWITCH_other_AVGQ} & {R32_SWITCH_other_AVGQ} & {R64_SWITCH_other_AVGQ} & {D121_SWITCH_other_MEDQ} & {R32_SWITCH_other_MEDQ} & {R64_SWITCH_other_MEDQ} \\\\
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

            D121_SWITCH_neg_ASR=result["densenet121"]["SWITCH_neg"]["success_rate"],
            R32_SWITCH_neg_ASR=result["resnext32_4"]["SWITCH_neg"]["success_rate"],
            R64_SWITCH_neg_ASR=result["resnext64_4"]["SWITCH_neg"]["success_rate"],
            D121_SWITCH_neg_AVGQ=result["densenet121"]["SWITCH_neg"][avg_q],
            R32_SWITCH_neg_AVGQ=result["resnext32_4"]["SWITCH_neg"][avg_q],
            R64_SWITCH_neg_AVGQ=result["resnext64_4"]["SWITCH_neg"][avg_q],
            D121_SWITCH_neg_MEDQ=result["densenet121"]["SWITCH_neg"][med_q],
            R32_SWITCH_neg_MEDQ=result["resnext32_4"]["SWITCH_neg"][med_q],
            R64_SWITCH_neg_MEDQ=result["resnext64_4"]["SWITCH_neg"][med_q],

            D121_NO_SWITCH_rnd_ASR=result["densenet121"]["NO_SWITCH_rnd"]["success_rate"],
            R32_NO_SWITCH_rnd_ASR=result["resnext32_4"]["NO_SWITCH_rnd"]["success_rate"],
            R64_NO_SWITCH_rnd_ASR=result["resnext64_4"]["NO_SWITCH_rnd"]["success_rate"],
            D121_NO_SWITCH_rnd_AVGQ=result["densenet121"]["NO_SWITCH_rnd"][avg_q],
            R32_NO_SWITCH_rnd_AVGQ=result["resnext32_4"]["NO_SWITCH_rnd"][avg_q],
            R64_NO_SWITCH_rnd_AVGQ=result["resnext64_4"]["NO_SWITCH_rnd"][avg_q],
            D121_NO_SWITCH_rnd_MEDQ=result["densenet121"]["NO_SWITCH_rnd"][med_q],
            R32_NO_SWITCH_rnd_MEDQ=result["resnext32_4"]["NO_SWITCH_rnd"][med_q],
            R64_NO_SWITCH_rnd_MEDQ=result["resnext64_4"]["NO_SWITCH_rnd"][med_q],

            D121_SWITCH_other_ASR=result["densenet121"]["SWITCH_other"]["success_rate"],
            R32_SWITCH_other_ASR=result["resnext32_4"]["SWITCH_other"]["success_rate"],
            R64_SWITCH_other_ASR=result["resnext64_4"]["SWITCH_other"]["success_rate"],
            D121_SWITCH_other_AVGQ=result["densenet121"]["SWITCH_other"][avg_q],
            R32_SWITCH_other_AVGQ=result["resnext32_4"]["SWITCH_other"][avg_q],
            R64_SWITCH_other_AVGQ=result["resnext64_4"]["SWITCH_other"][avg_q],
            D121_SWITCH_other_MEDQ=result["densenet121"]["SWITCH_other"][med_q],
            R32_SWITCH_other_MEDQ=result["resnext32_4"]["SWITCH_other"][med_q],
            R64_SWITCH_other_MEDQ=result["resnext64_4"]["SWITCH_other"][med_q]
        )
        )


def draw_tables_for_CIFAR(norm, archs_result):
    result = archs_result
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    if norm == "linf":
        print("""
                        RGF & {ComDefend_RGF_ASR}\% & {FeatureDistillation_RGF_ASR}\% & {jpeg_RGF_ASR}\% & {ComDefend_RGF_AVGQ} & {FeatureDistillation_RGF_AVGQ} & {jpeg_RGF_AVGQ}   \\\\
                        P-RGF & {ComDefend_PRGF_ASR}\% & {FeatureDistillation_PRGF_ASR}\%  & {jpeg_PRGF_ASR}\%  & {ComDefend_PRGF_AVGQ} & {FeatureDistillation_PRGF_AVGQ} & {jpeg_PRGF_AVGQ}  \\\\
                        Bandits & {ComDefend_Bandits_ASR}\% & {FeatureDistillation_Bandits_ASR}\% & {jpeg_Bandits_ASR}\% & {ComDefend_Bandits_AVGQ} & {FeatureDistillation_Bandits_AVGQ} &  {jpeg_Bandits_AVGQ}  \\\\
                        PPBA & {ComDefend_PPBA_ASR}\% & {FeatureDistillation_PPBA_ASR}\% &  {jpeg_PPBA_ASR}\%  & {ComDefend_PPBA_AVGQ} & {FeatureDistillation_PPBA_AVGQ} &  {jpeg_PPBA_AVGQ}  \\\\
                        Parsimonious & {ComDefend_Parsimonious_ASR}\% & {FeatureDistillation_Parsimonious_ASR}\% & {jpeg_Parsimonious_ASR}\% &{ComDefend_Parsimonious_AVGQ} & {FeatureDistillation_Parsimonious_AVGQ} & {jpeg_Parsimonious_AVGQ}  \\\\
                        SignHunter & {ComDefend_SignHunter_ASR}\% & {FeatureDistillation_SignHunter_ASR}\% & {jpeg_SignHunter_ASR}\% & {ComDefend_SignHunter_AVGQ} & {FeatureDistillation_SignHunter_AVGQ} & {jpeg_SignHunter_AVGQ}  \\\\
                        Square Attack & {ComDefend_Square_ASR}\% & {FeatureDistillation_Square_ASR}\%& {jpeg_Square_ASR}\% & {ComDefend_Square_AVGQ} & {FeatureDistillation_Square_AVGQ} & {jpeg_Square_AVGQ} \\\\
                        NO SWITCH & {ComDefend_NO_SWITCH_ASR}\% & {FeatureDistillation_NO_SWITCH_ASR}\% &  {jpeg_NO_SWITCH_ASR}\% &{ComDefend_NO_SWITCH_AVGQ} & {FeatureDistillation_NO_SWITCH_AVGQ} &  {jpeg_NO_SWITCH_AVGQ}  \\\\
                        SWITCH$_\\text{{naive}}$ & {ComDefend_SWITCH_naive_ASR}\% & {FeatureDistillation_SWITCH_naive_ASR}\% & {jpeg_SWITCH_naive_ASR}\% & {ComDefend_SWITCH_naive_AVGQ} & {FeatureDistillation_SWITCH_naive_AVGQ} & {jpeg_SWITCH_naive_AVGQ} \\\\
                        SWITCH$_\\text{{RGF}}$ & {ComDefend_SWITCH_RGF_ASR}\% & {FeatureDistillation_SWITCH_RGF_ASR}\% & {jpeg_SWITCH_RGF_ASR}\% & {ComDefend_SWITCH_RGF_AVGQ} & {FeatureDistillation_SWITCH_RGF_AVGQ} & {jpeg_SWITCH_RGF_AVGQ} \\\\
                                                                """.format(
            ComDefend_RGF_ASR=result["com_defend"]["RGF"]["success_rate"],
            FeatureDistillation_RGF_ASR=result["feature_distillation"]["RGF"]["success_rate"],
            ComDefend_RGF_AVGQ=result["com_defend"]["RGF"][avg_q],
            FeatureDistillation_RGF_AVGQ=result["feature_distillation"]["RGF"][avg_q],
            jpeg_RGF_ASR=result["jpeg"]["RGF"]["success_rate"],
            jpeg_RGF_AVGQ=result["jpeg"]["RGF"][avg_q],

            ComDefend_PRGF_ASR=result["com_defend"]["PRGF"]["success_rate"],
            FeatureDistillation_PRGF_ASR=result["feature_distillation"]["PRGF"]["success_rate"],

            ComDefend_PRGF_AVGQ=result["com_defend"]["PRGF"][avg_q],
            FeatureDistillation_PRGF_AVGQ=result["feature_distillation"]["PRGF"][avg_q],
            jpeg_PRGF_ASR=result["jpeg"]["PRGF"]["success_rate"],
            jpeg_PRGF_AVGQ=result["jpeg"]["PRGF"][avg_q],

            ComDefend_Bandits_ASR=result["com_defend"]["Bandits"]["success_rate"],
            FeatureDistillation_Bandits_ASR=result["feature_distillation"]["Bandits"]["success_rate"],

            ComDefend_Bandits_AVGQ=result["com_defend"]["Bandits"][avg_q],
            FeatureDistillation_Bandits_AVGQ=result["feature_distillation"]["Bandits"][avg_q],
            jpeg_Bandits_ASR=result["jpeg"]["Bandits"]["success_rate"],
            jpeg_Bandits_AVGQ=result["jpeg"]["Bandits"][avg_q],

            ComDefend_PPBA_ASR=result["com_defend"]["PPBA"]["success_rate"],
            FeatureDistillation_PPBA_ASR=result["feature_distillation"]["PPBA"]["success_rate"],

            ComDefend_PPBA_AVGQ=result["com_defend"]["PPBA"][avg_q],
            FeatureDistillation_PPBA_AVGQ=result["feature_distillation"]["PPBA"][avg_q],
            jpeg_PPBA_ASR=result["jpeg"]["PPBA"]["success_rate"],
            jpeg_PPBA_AVGQ=result["jpeg"]["PPBA"][avg_q],

            ComDefend_Parsimonious_ASR=result["com_defend"]["Parsimonious"]["success_rate"],
            FeatureDistillation_Parsimonious_ASR=result["feature_distillation"]["Parsimonious"]["success_rate"],

            ComDefend_Parsimonious_AVGQ=result["com_defend"]["Parsimonious"][avg_q],
            FeatureDistillation_Parsimonious_AVGQ=result["feature_distillation"]["Parsimonious"][avg_q],
            jpeg_Parsimonious_ASR=result["jpeg"]["Parsimonious"]["success_rate"],
            jpeg_Parsimonious_AVGQ=result["jpeg"]["Parsimonious"][avg_q],

            ComDefend_SignHunter_ASR=result["com_defend"]["SignHunter"]["success_rate"],
            FeatureDistillation_SignHunter_ASR=result["feature_distillation"]["SignHunter"]["success_rate"],

            ComDefend_SignHunter_AVGQ=result["com_defend"]["SignHunter"][avg_q],
            FeatureDistillation_SignHunter_AVGQ=result["feature_distillation"]["SignHunter"][avg_q],
            jpeg_SignHunter_ASR=result["jpeg"]["SignHunter"]["success_rate"],
            jpeg_SignHunter_AVGQ=result["jpeg"]["SignHunter"][avg_q],

            ComDefend_Square_ASR=result["com_defend"]["Square"]["success_rate"],
            FeatureDistillation_Square_ASR=result["feature_distillation"]["Square"]["success_rate"],
            ComDefend_Square_AVGQ=result["com_defend"]["Square"][avg_q],
            FeatureDistillation_Square_AVGQ=result["feature_distillation"]["Square"][avg_q],
            jpeg_Square_ASR=result["jpeg"]["Square"]["success_rate"],
            jpeg_Square_AVGQ=result["jpeg"]["Square"][avg_q],

            ComDefend_NO_SWITCH_ASR=result["com_defend"]["NO_SWITCH"]["success_rate"],
            FeatureDistillation_NO_SWITCH_ASR=result["feature_distillation"]["NO_SWITCH"]["success_rate"],

            ComDefend_NO_SWITCH_AVGQ=result["com_defend"]["NO_SWITCH"][avg_q],
            FeatureDistillation_NO_SWITCH_AVGQ=result["feature_distillation"]["NO_SWITCH"][avg_q],
            jpeg_NO_SWITCH_ASR=result["jpeg"]["NO_SWITCH"]["success_rate"],
            jpeg_NO_SWITCH_AVGQ=result["jpeg"]["NO_SWITCH"][avg_q],

            ComDefend_SWITCH_naive_ASR=result["com_defend"]["SWITCH_naive"]["success_rate"],
            FeatureDistillation_SWITCH_naive_ASR=result["feature_distillation"]["SWITCH_naive"]["success_rate"],

            ComDefend_SWITCH_naive_AVGQ=result["com_defend"]["SWITCH_naive"][avg_q],
            FeatureDistillation_SWITCH_naive_AVGQ=result["feature_distillation"]["SWITCH_naive"][avg_q],
            jpeg_SWITCH_naive_ASR=result["jpeg"]["SWITCH_naive"]["success_rate"],
            jpeg_SWITCH_naive_AVGQ=result["jpeg"]["SWITCH_naive"][avg_q],

            ComDefend_SWITCH_RGF_ASR=result["com_defend"]["SWITCH_RGF"]["success_rate"],
            FeatureDistillation_SWITCH_RGF_ASR=result["feature_distillation"]["SWITCH_RGF"]["success_rate"],

            ComDefend_SWITCH_RGF_AVGQ=result["com_defend"]["SWITCH_RGF"][avg_q],
            FeatureDistillation_SWITCH_RGF_AVGQ=result["feature_distillation"]["SWITCH_RGF"][avg_q],
            jpeg_SWITCH_RGF_ASR=result["jpeg"]["SWITCH_RGF"]["success_rate"],
            jpeg_SWITCH_RGF_AVGQ=result["jpeg"]["SWITCH_RGF"][avg_q],

        )
        )
    else:
        print("""
                        & {norm_str} & RGF & {pyramidnet272_RGF_ASR}\% & {gdas_RGF_ASR}\% & {WRN28_RGF_ASR}\% & {WRN40_RGF_ASR}\% & {pyramidnet272_RGF_AVGQ} & {gdas_RGF_AVGQ} & {WRN28_RGF_AVGQ} & {WRN40_RGF_AVGQ} & {pyramidnet272_RGF_MEDQ} & {gdas_RGF_MEDQ} & {WRN28_RGF_MEDQ} & {WRN40_RGF_MEDQ} \\\\
                        & & P-RGF & {pyramidnet272_PRGF_ASR}\% & {gdas_PRGF_ASR}\% & {WRN28_PRGF_ASR}\% & {WRN40_PRGF_ASR}\% & {pyramidnet272_PRGF_AVGQ} & {gdas_PRGF_AVGQ} & {WRN28_PRGF_AVGQ} & {WRN40_PRGF_AVGQ} & {pyramidnet272_PRGF_MEDQ} & {gdas_PRGF_MEDQ} & {WRN28_PRGF_MEDQ} & {WRN40_PRGF_MEDQ} \\\\
                        & & Bandits & {pyramidnet272_Bandits_ASR}\% & {gdas_Bandits_ASR}\% & {WRN28_Bandits_ASR}\% & {WRN40_Bandits_ASR}\% & {pyramidnet272_Bandits_AVGQ} & {gdas_Bandits_AVGQ} & {WRN28_Bandits_AVGQ} & {WRN40_Bandits_AVGQ} & {pyramidnet272_Bandits_MEDQ} & {gdas_Bandits_MEDQ} & {WRN28_Bandits_MEDQ} & {WRN40_Bandits_MEDQ} \\\\
                        & & PPBA & {pyramidnet272_PPBA_ASR}\% & {gdas_PPBA_ASR}\% & {WRN28_PPBA_ASR}\% & {WRN40_PPBA_ASR}\% & {pyramidnet272_PPBA_AVGQ} & {gdas_PPBA_AVGQ} & {WRN28_PPBA_AVGQ} & {WRN40_PPBA_AVGQ} & {pyramidnet272_PPBA_MEDQ} & {gdas_PPBA_MEDQ} & {WRN28_PPBA_MEDQ} & {WRN40_PPBA_MEDQ} \\\\
                        & & SignHunter & {pyramidnet272_SignHunter_ASR}\% & {gdas_SignHunter_ASR}\% & {WRN28_SignHunter_ASR}\% & {WRN40_SignHunter_ASR}\% & {pyramidnet272_SignHunter_AVGQ} & {gdas_SignHunter_AVGQ} & {WRN28_SignHunter_AVGQ} & {WRN40_SignHunter_AVGQ} & {pyramidnet272_SignHunter_MEDQ} & {gdas_SignHunter_MEDQ} & {WRN28_SignHunter_MEDQ} & {WRN40_SignHunter_MEDQ} \\\\
                        & & Square Attack & {pyramidnet272_Square_ASR}\% & {gdas_Square_ASR}\% & {WRN28_Square_ASR}\% & {WRN40_Square_ASR}\% & {pyramidnet272_Square_AVGQ} & {gdas_Square_AVGQ} & {WRN28_Square_AVGQ} & {WRN40_Square_AVGQ} & {pyramidnet272_Square_MEDQ} & {gdas_Square_MEDQ} & {WRN28_Square_MEDQ} & {WRN40_Square_MEDQ} \\\\
                        & & NO SWITCH & {pyramidnet272_NO_SWITCH_ASR}\% & {gdas_NO_SWITCH_ASR}\% & {WRN28_NO_SWITCH_ASR}\% & {WRN40_NO_SWITCH_ASR}\% & {pyramidnet272_NO_SWITCH_AVGQ} & {gdas_NO_SWITCH_AVGQ} & {WRN28_NO_SWITCH_AVGQ} & {WRN40_NO_SWITCH_AVGQ} & {pyramidnet272_NO_SWITCH_MEDQ} & {gdas_NO_SWITCH_MEDQ} & {WRN28_NO_SWITCH_MEDQ} & {WRN40_NO_SWITCH_MEDQ} \\\\
                        & & SWITCH$_neg$ & {pyramidnet272_SWITCH_neg_ASR}\% & {gdas_SWITCH_neg_ASR}\% & {WRN28_SWITCH_neg_ASR}\% & {WRN40_SWITCH_neg_ASR}\% & {pyramidnet272_SWITCH_neg_AVGQ} & {gdas_SWITCH_neg_AVGQ} & {WRN28_SWITCH_neg_AVGQ} & {WRN40_SWITCH_neg_AVGQ} & {pyramidnet272_SWITCH_neg_MEDQ} & {gdas_SWITCH_neg_MEDQ} & {WRN28_SWITCH_neg_MEDQ} & {WRN40_SWITCH_neg_MEDQ} \\\\
                        & & NO SWITCH$_rnd$ & {pyramidnet272_NO_SWITCH_rnd_ASR}\% & {gdas_NO_SWITCH_rnd_ASR}\% & {WRN28_NO_SWITCH_rnd_ASR}\% & {WRN40_NO_SWITCH_rnd_ASR}\% & {pyramidnet272_NO_SWITCH_rnd_AVGQ} & {gdas_NO_SWITCH_rnd_AVGQ} & {WRN28_NO_SWITCH_rnd_AVGQ} & {WRN40_NO_SWITCH_rnd_AVGQ} & {pyramidnet272_NO_SWITCH_rnd_MEDQ} & {gdas_NO_SWITCH_rnd_MEDQ} & {WRN28_NO_SWITCH_rnd_MEDQ} & {WRN40_NO_SWITCH_rnd_MEDQ} \\\\
                        & & SWITCH$_other$ & {pyramidnet272_SWITCH_other_ASR}\% & {gdas_SWITCH_other_ASR}\% & {WRN28_SWITCH_other_ASR}\% & {WRN40_SWITCH_other_ASR}\% & {pyramidnet272_SWITCH_other_AVGQ} & {gdas_SWITCH_other_AVGQ} & {WRN28_SWITCH_other_AVGQ} & {WRN40_SWITCH_other_AVGQ} & {pyramidnet272_SWITCH_other_MEDQ} & {gdas_SWITCH_other_MEDQ} & {WRN28_SWITCH_other_MEDQ} & {WRN40_SWITCH_other_MEDQ} \\\\
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

                           pyramidnet272_SWITCH_neg_ASR=result["pyramidnet272"]["SWITCH_neg"]["success_rate"],
                           gdas_SWITCH_neg_ASR=result["gdas"]["SWITCH_neg"]["success_rate"],
                           WRN28_SWITCH_neg_ASR=result["WRN-28-10-drop"]["SWITCH_neg"]["success_rate"],
                           WRN40_SWITCH_neg_ASR=result["WRN-40-10-drop"]["SWITCH_neg"]["success_rate"],
                           pyramidnet272_SWITCH_neg_AVGQ=result["pyramidnet272"]["SWITCH_neg"][avg_q],
                           gdas_SWITCH_neg_AVGQ=result["gdas"]["SWITCH_neg"][avg_q],
                           WRN28_SWITCH_neg_AVGQ=result["WRN-28-10-drop"]["SWITCH_neg"][avg_q],
                           WRN40_SWITCH_neg_AVGQ=result["WRN-40-10-drop"]["SWITCH_neg"][avg_q],
                           pyramidnet272_SWITCH_neg_MEDQ=result["pyramidnet272"]["SWITCH_neg"][med_q],
                           gdas_SWITCH_neg_MEDQ=result["gdas"]["SWITCH_neg"][med_q],
                           WRN28_SWITCH_neg_MEDQ=result["WRN-28-10-drop"]["SWITCH_neg"][med_q],
                           WRN40_SWITCH_neg_MEDQ=result["WRN-40-10-drop"]["SWITCH_neg"][med_q],

                           pyramidnet272_SWITCH_other_ASR=result["pyramidnet272"]["SWITCH_other"]["success_rate"],
                           gdas_SWITCH_other_ASR=result["gdas"]["SWITCH_other"]["success_rate"],
                           WRN28_SWITCH_other_ASR=result["WRN-28-10-drop"]["SWITCH_other"]["success_rate"],
                           WRN40_SWITCH_other_ASR=result["WRN-40-10-drop"]["SWITCH_other"]["success_rate"],
                           pyramidnet272_SWITCH_other_AVGQ=result["pyramidnet272"]["SWITCH_other"][avg_q],
                           gdas_SWITCH_other_AVGQ=result["gdas"]["SWITCH_other"][avg_q],
                           WRN28_SWITCH_other_AVGQ=result["WRN-28-10-drop"]["SWITCH_other"][avg_q],
                           WRN40_SWITCH_other_AVGQ=result["WRN-40-10-drop"]["SWITCH_other"][avg_q],
                           pyramidnet272_SWITCH_other_MEDQ=result["pyramidnet272"]["SWITCH_other"][med_q],
                           gdas_SWITCH_other_MEDQ=result["gdas"]["SWITCH_other"][med_q],
                           WRN28_SWITCH_other_MEDQ=result["WRN-28-10-drop"]["SWITCH_other"][med_q],
                           WRN40_SWITCH_other_MEDQ=result["WRN-40-10-drop"]["SWITCH_other"][med_q],

                           pyramidnet272_NO_SWITCH_rnd_ASR=result["pyramidnet272"]["NO_SWITCH_rnd"]["success_rate"],
                           gdas_NO_SWITCH_rnd_ASR=result["gdas"]["NO_SWITCH_rnd"]["success_rate"],
                           WRN28_NO_SWITCH_rnd_ASR=result["WRN-28-10-drop"]["NO_SWITCH_rnd"]["success_rate"],
                           WRN40_NO_SWITCH_rnd_ASR=result["WRN-40-10-drop"]["NO_SWITCH_rnd"]["success_rate"],
                           pyramidnet272_NO_SWITCH_rnd_AVGQ=result["pyramidnet272"]["NO_SWITCH_rnd"][avg_q],
                           gdas_NO_SWITCH_rnd_AVGQ=result["gdas"]["NO_SWITCH_rnd"][avg_q],
                           WRN28_NO_SWITCH_rnd_AVGQ=result["WRN-28-10-drop"]["NO_SWITCH_rnd"][avg_q],
                           WRN40_NO_SWITCH_rnd_AVGQ=result["WRN-40-10-drop"]["NO_SWITCH_rnd"][avg_q],
                           pyramidnet272_NO_SWITCH_rnd_MEDQ=result["pyramidnet272"]["NO_SWITCH_rnd"][med_q],
                           gdas_NO_SWITCH_rnd_MEDQ=result["gdas"]["NO_SWITCH_rnd"][med_q],
                           WRN28_NO_SWITCH_rnd_MEDQ=result["WRN-28-10-drop"]["NO_SWITCH_rnd"][med_q],
                           WRN40_NO_SWITCH_rnd_MEDQ=result["WRN-40-10-drop"]["NO_SWITCH_rnd"][med_q],
                           )
              )
if __name__ == "__main__":
    dataset = "TinyImageNet"
    norm = "linf"
    targeted = False
    if "CIFAR" in dataset:
        arch = 'resnet-50'
    else:
        arch = "resnet50"
    defense_models = ["com_defend", "feature_distillation","jpeg"]
    result = fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, defense_models)
    if "CIFAR" in dataset:
        draw_tables_for_CIFAR(norm, result)
    elif "TinyImageNet" in dataset:
        draw_tables_for_TinyImageNet(norm, result)
    print("THIS IS {} {} {}".format(dataset, norm, "untargeted" if not targeted else "targeted"))
