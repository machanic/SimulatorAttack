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
                        "NO_SWITCH_rnd": "NO_SWITCH_rnd",
                        # "SWITCH_rnd_save":'SWITCH_other', FIXME
                        "SWITCH_neg_save":'SWITCH_neg',
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

def fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, defense_method):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result = {}
    for method, folder in folder_list.items():
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

        file_path = folder + "/{}_{}_result.json".format(arch, defense_method)
        assert os.path.exists(file_path), "{} does not exist!".format(file_path)
        success_rate, avg_query_over_successful_samples, median_query_over_successful_samples, \
        correct_all, query_all, not_done_all, json_content = read_json_and_extract(file_path)
        not_done_all[query_all>10000] = 1
        query_all[query_all > 10000] = 10000
        query_all[not_done_all==1] = 10000

        if "SWITCH" in method:
            assert lr == json_content["args"]["image_lr"]
        avg_query_over_all_samples = int(new_round(np.mean(query_all[correct_all.astype(np.bool)]).item(),0))
        median_query_over_all_samples = int(new_round(np.median(query_all[correct_all.astype(np.bool)]).item(),0))
        result[method] = {"success_rate":success_rate, "avg_query_over_successful_samples":avg_query_over_successful_samples,
                          "median_query_over_successful_samples": median_query_over_successful_samples,
                        "avg_query_over_all_samples": avg_query_over_all_samples, "median_query_over_all_samples":median_query_over_all_samples}
    return result

def draw_tables(norm, result):
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    if norm == "linf":
        print("""
& & RGF & {RGF_ASR_CIFAR10}\% & {RGF_AVGQ_CIFAR10} & {RGF_MEDQ_CIFAR10} &  & {RGF_ASR_CIFAR100}\% & {RGF_AVGQ_CIFAR100} & {RGF_MEDQ_CIFAR100} & &  {RGF_ASR_TinyImageNet}\% & {RGF_AVGQ_TinyImageNet} & {RGF_MEDQ_TinyImageNet} \\\\
& & P-RGF & {PRGF_ASR_CIFAR10}\% & {PRGF_AVGQ_CIFAR10} & {PRGF_MEDQ_CIFAR10} &  & {PRGF_ASR_CIFAR100}\% & {PRGF_AVGQ_CIFAR100} & {PRGF_MEDQ_CIFAR100} & &  {PRGF_ASR_TinyImageNet}\% & {PRGF_AVGQ_TinyImageNet} & {PRGF_MEDQ_TinyImageNet} \\\\
& & Bandits & {Bandits_ASR_CIFAR10}\% & {Bandits_AVGQ_CIFAR10} & {Bandits_MEDQ_CIFAR10} &  & {Bandits_ASR_CIFAR100}\% & {Bandits_AVGQ_CIFAR100} & {Bandits_MEDQ_CIFAR100} & &  {Bandits_ASR_TinyImageNet}\% & {Bandits_AVGQ_TinyImageNet} & {Bandits_MEDQ_TinyImageNet} \\\\
& & PPBA & {PPBA_ASR_CIFAR10}\% & {PPBA_AVGQ_CIFAR10} & {PPBA_MEDQ_CIFAR10} &  & {PPBA_ASR_CIFAR100}\% & {PPBA_AVGQ_CIFAR100} & {PPBA_MEDQ_CIFAR100} & &  {PPBA_ASR_TinyImageNet}\% & {PPBA_AVGQ_TinyImageNet} & {PPBA_MEDQ_TinyImageNet} \\\\
& & Parsimonious & {Parsimonious_ASR_CIFAR10}\% & {Parsimonious_AVGQ_CIFAR10} & {Parsimonious_MEDQ_CIFAR10} &  & {Parsimonious_ASR_CIFAR100}\% & {Parsimonious_AVGQ_CIFAR100} & {Parsimonious_MEDQ_CIFAR100} & &  {Parsimonious_ASR_TinyImageNet}\% & {Parsimonious_AVGQ_TinyImageNet} & {Parsimonious_MEDQ_TinyImageNet} \\\\ 
& & SignHunter & {SignHunter_ASR_CIFAR10}\% & {SignHunter_AVGQ_CIFAR10} & {SignHunter_MEDQ_CIFAR10} &  & {SignHunter_ASR_CIFAR100}\% & {SignHunter_AVGQ_CIFAR100} & {SignHunter_MEDQ_CIFAR100} & &  {SignHunter_ASR_TinyImageNet}\% & {SignHunter_AVGQ_TinyImageNet} & {SignHunter_MEDQ_TinyImageNet} \\\\
& & Square Attack & {Square_ASR_CIFAR10}\% & {Square_AVGQ_CIFAR10} & {Square_MEDQ_CIFAR10} &  & {Square_ASR_CIFAR100}\% & {Square_AVGQ_CIFAR100} & {Square_MEDQ_CIFAR100} & &  {Square_ASR_TinyImageNet}\% & {Square_AVGQ_TinyImageNet} & {Square_MEDQ_TinyImageNet} \\\\
& & NO SWITCH & {NO_SWITCH_ASR_CIFAR10}\% & {NO_SWITCH_AVGQ_CIFAR10} & {NO_SWITCH_MEDQ_CIFAR10} &  & {NO_SWITCH_ASR_CIFAR100}\% & {NO_SWITCH_AVGQ_CIFAR100} & {NO_SWITCH_MEDQ_CIFAR100} & &  {NO_SWITCH_ASR_TinyImageNet}\% & {NO_SWITCH_AVGQ_TinyImageNet} & {NO_SWITCH_MEDQ_TinyImageNet} \\\\
& & SWITCH_neg & {SWITCH_neg_ASR_CIFAR10}\% & {SWITCH_neg_AVGQ_CIFAR10} & {SWITCH_neg_MEDQ_CIFAR10} &  & {SWITCH_neg_ASR_CIFAR100}\% & {SWITCH_neg_AVGQ_CIFAR100} & {SWITCH_neg_MEDQ_CIFAR100} & &  {SWITCH_neg_ASR_TinyImageNet}\% & {SWITCH_neg_AVGQ_TinyImageNet} & {SWITCH_neg_MEDQ_TinyImageNet} \\\\
& & NO SWITCH$_rnd$ & {NO_SWITCH_rnd_ASR_CIFAR10}\% & {NO_SWITCH_rnd_AVGQ_CIFAR10} & {NO_SWITCH_rnd_MEDQ_CIFAR10} &  & {NO_SWITCH_rnd_ASR_CIFAR100}\% & {NO_SWITCH_rnd_AVGQ_CIFAR100} & {NO_SWITCH_rnd_MEDQ_CIFAR100} & &  {NO_SWITCH_rnd_ASR_TinyImageNet}\% & {NO_SWITCH_rnd_AVGQ_TinyImageNet} & {NO_SWITCH_rnd_MEDQ_TinyImageNet} \\\\
& & SWITCH$_other$ & {SWITCH_other_ASR_CIFAR10}\% & {SWITCH_other_AVGQ_CIFAR10} & {SWITCH_other_MEDQ_CIFAR10} &  & {SWITCH_other_ASR_CIFAR100}\% & {SWITCH_other_AVGQ_CIFAR100} & {SWITCH_other_MEDQ_CIFAR100} & &  {SWITCH_other_ASR_TinyImageNet}\% & {SWITCH_other_AVGQ_TinyImageNet} & {SWITCH_other_MEDQ_TinyImageNet} \\\\
        """.format(
            RGF_ASR_CIFAR10=result["CIFAR-10"]["RGF"]["success_rate"],RGF_AVGQ_CIFAR10=result["CIFAR-10"]["RGF"][avg_q],RGF_MEDQ_CIFAR10=result["CIFAR-10"]["RGF"][med_q],
            RGF_ASR_CIFAR100=result["CIFAR-100"]["RGF"]["success_rate"], RGF_AVGQ_CIFAR100=result["CIFAR-100"]["RGF"][avg_q], RGF_MEDQ_CIFAR100=result["CIFAR-100"]["RGF"][med_q],
            RGF_ASR_TinyImageNet=result["TinyImageNet"]["RGF"]["success_rate"], RGF_AVGQ_TinyImageNet=result["TinyImageNet"]["RGF"][avg_q], RGF_MEDQ_TinyImageNet=result["TinyImageNet"]["RGF"][med_q],

            PRGF_ASR_CIFAR10=result["CIFAR-10"]["PRGF"]["success_rate"],
            PRGF_AVGQ_CIFAR10=result["CIFAR-10"]["PRGF"][avg_q], PRGF_MEDQ_CIFAR10=result["CIFAR-10"]["PRGF"][med_q],
            PRGF_ASR_CIFAR100=result["CIFAR-100"]["PRGF"]["success_rate"],
            PRGF_AVGQ_CIFAR100=result["CIFAR-100"]["PRGF"][avg_q],
            PRGF_MEDQ_CIFAR100=result["CIFAR-100"]["PRGF"][med_q],
            PRGF_ASR_TinyImageNet=result["TinyImageNet"]["PRGF"]["success_rate"],
            PRGF_AVGQ_TinyImageNet=result["TinyImageNet"]["PRGF"][avg_q],
            PRGF_MEDQ_TinyImageNet=result["TinyImageNet"]["PRGF"][med_q],

            Bandits_ASR_CIFAR10=result["CIFAR-10"]["Bandits"]["success_rate"],
            Bandits_AVGQ_CIFAR10=result["CIFAR-10"]["Bandits"][avg_q],
            Bandits_MEDQ_CIFAR10=result["CIFAR-10"]["Bandits"][med_q],
            Bandits_ASR_CIFAR100=result["CIFAR-100"]["Bandits"]["success_rate"],
            Bandits_AVGQ_CIFAR100=result["CIFAR-100"]["Bandits"][avg_q],
            Bandits_MEDQ_CIFAR100=result["CIFAR-100"]["Bandits"][med_q],
            Bandits_ASR_TinyImageNet=result["TinyImageNet"]["Bandits"]["success_rate"],
            Bandits_AVGQ_TinyImageNet=result["TinyImageNet"]["Bandits"][avg_q],
            Bandits_MEDQ_TinyImageNet=result["TinyImageNet"]["Bandits"][med_q],

            PPBA_ASR_CIFAR10=result["CIFAR-10"]["PPBA"]["success_rate"],
            PPBA_AVGQ_CIFAR10=result["CIFAR-10"]["PPBA"][avg_q], PPBA_MEDQ_CIFAR10=result["CIFAR-10"]["PPBA"][med_q],
            PPBA_ASR_CIFAR100=result["CIFAR-100"]["PPBA"]["success_rate"],
            PPBA_AVGQ_CIFAR100=result["CIFAR-100"]["PPBA"][avg_q],
            PPBA_MEDQ_CIFAR100=result["CIFAR-100"]["PPBA"][med_q],
            PPBA_ASR_TinyImageNet=result["TinyImageNet"]["PPBA"]["success_rate"],
            PPBA_AVGQ_TinyImageNet=result["TinyImageNet"]["PPBA"][avg_q],
            PPBA_MEDQ_TinyImageNet=result["TinyImageNet"]["PPBA"][med_q],

            Parsimonious_ASR_CIFAR10=result["CIFAR-10"]["Parsimonious"]["success_rate"],
            Parsimonious_AVGQ_CIFAR10=result["CIFAR-10"]["Parsimonious"][avg_q],
            Parsimonious_MEDQ_CIFAR10=result["CIFAR-10"]["Parsimonious"][med_q],
            Parsimonious_ASR_CIFAR100=result["CIFAR-100"]["Parsimonious"]["success_rate"],
            Parsimonious_AVGQ_CIFAR100=result["CIFAR-100"]["Parsimonious"][avg_q],
            Parsimonious_MEDQ_CIFAR100=result["CIFAR-100"]["Parsimonious"][med_q],
            Parsimonious_ASR_TinyImageNet=result["TinyImageNet"]["Parsimonious"]["success_rate"],
            Parsimonious_AVGQ_TinyImageNet=result["TinyImageNet"]["Parsimonious"][avg_q],
            Parsimonious_MEDQ_TinyImageNet=result["TinyImageNet"]["Parsimonious"][med_q],

            SignHunter_ASR_CIFAR10=result["CIFAR-10"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_CIFAR10=result["CIFAR-10"]["SignHunter"][avg_q],
            SignHunter_MEDQ_CIFAR10=result["CIFAR-10"]["SignHunter"][med_q],
            SignHunter_ASR_CIFAR100=result["CIFAR-100"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_CIFAR100=result["CIFAR-100"]["SignHunter"][avg_q],
            SignHunter_MEDQ_CIFAR100=result["CIFAR-100"]["SignHunter"][med_q],
            SignHunter_ASR_TinyImageNet=result["TinyImageNet"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_TinyImageNet=result["TinyImageNet"]["SignHunter"][avg_q],
            SignHunter_MEDQ_TinyImageNet=result["TinyImageNet"]["SignHunter"][med_q],

            Square_ASR_CIFAR10=result["CIFAR-10"]["Square"]["success_rate"],
            Square_AVGQ_CIFAR10=result["CIFAR-10"]["Square"][avg_q],
            Square_MEDQ_CIFAR10=result["CIFAR-10"]["Square"][med_q],
            Square_ASR_CIFAR100=result["CIFAR-100"]["Square"]["success_rate"],
            Square_AVGQ_CIFAR100=result["CIFAR-100"]["Square"][avg_q],
            Square_MEDQ_CIFAR100=result["CIFAR-100"]["Square"][med_q],
            Square_ASR_TinyImageNet=result["TinyImageNet"]["Square"]["success_rate"],
            Square_AVGQ_TinyImageNet=result["TinyImageNet"]["Square"][avg_q],
            Square_MEDQ_TinyImageNet=result["TinyImageNet"]["Square"][med_q],

            NO_SWITCH_ASR_CIFAR10=result["CIFAR-10"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH"][med_q],
            NO_SWITCH_ASR_CIFAR100=result["CIFAR-100"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH"][med_q],
            NO_SWITCH_ASR_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"][med_q],

            SWITCH_neg_ASR_CIFAR10=result["CIFAR-10"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_CIFAR10=result["CIFAR-10"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_CIFAR10=result["CIFAR-10"]["SWITCH_neg"][med_q],
            SWITCH_neg_ASR_CIFAR100=result["CIFAR-100"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_CIFAR100=result["CIFAR-100"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_CIFAR100=result["CIFAR-100"]["SWITCH_neg"][med_q],
            SWITCH_neg_ASR_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"][med_q],

            NO_SWITCH_rnd_ASR_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"][med_q],
            NO_SWITCH_rnd_ASR_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"][med_q],
            NO_SWITCH_rnd_ASR_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"][med_q],

            SWITCH_other_ASR_CIFAR10=result["CIFAR-10"]["SWITCH_other"]["success_rate"],
            SWITCH_other_AVGQ_CIFAR10=result["CIFAR-10"]["SWITCH_other"][avg_q],
            SWITCH_other_MEDQ_CIFAR10=result["CIFAR-10"]["SWITCH_other"][med_q],
            SWITCH_other_ASR_CIFAR100=result["CIFAR-100"]["SWITCH_other"]["success_rate"],
            SWITCH_other_AVGQ_CIFAR100=result["CIFAR-100"]["SWITCH_other"][avg_q],
            SWITCH_other_MEDQ_CIFAR100=result["CIFAR-100"]["SWITCH_other"][med_q],
            SWITCH_other_ASR_TinyImageNet=result["TinyImageNet"]["SWITCH_other"]["success_rate"],
            SWITCH_other_AVGQ_TinyImageNet=result["TinyImageNet"]["SWITCH_other"][avg_q],
            SWITCH_other_MEDQ_TinyImageNet=result["TinyImageNet"]["SWITCH_other"][med_q],
        )
              )
    else:
        print("""
        & & RGF & {RGF_ASR_CIFAR10}\% & {RGF_AVGQ_CIFAR10} & {RGF_MEDQ_CIFAR10} &  & {RGF_ASR_CIFAR100}\% & {RGF_AVGQ_CIFAR100} & {RGF_MEDQ_CIFAR100} & &  {RGF_ASR_TinyImageNet}\% & {RGF_AVGQ_TinyImageNet} & {RGF_MEDQ_TinyImageNet} \\\\
        & & P-RGF & {PRGF_ASR_CIFAR10}\% & {PRGF_AVGQ_CIFAR10} & {PRGF_MEDQ_CIFAR10} &  & {PRGF_ASR_CIFAR100}\% & {PRGF_AVGQ_CIFAR100} & {PRGF_MEDQ_CIFAR100} & &  {PRGF_ASR_TinyImageNet}\% & {PRGF_AVGQ_TinyImageNet} & {PRGF_MEDQ_TinyImageNet} \\\\
        & & Bandits & {Bandits_ASR_CIFAR10}\% & {Bandits_AVGQ_CIFAR10} & {Bandits_MEDQ_CIFAR10} &  & {Bandits_ASR_CIFAR100}\% & {Bandits_AVGQ_CIFAR100} & {Bandits_MEDQ_CIFAR100} & &  {Bandits_ASR_TinyImageNet}\% & {Bandits_AVGQ_TinyImageNet} & {Bandits_MEDQ_TinyImageNet} \\\\
        & & PPBA & {PPBA_ASR_CIFAR10}\% & {PPBA_AVGQ_CIFAR10} & {PPBA_MEDQ_CIFAR10} &  & {PPBA_ASR_CIFAR100}\% & {PPBA_AVGQ_CIFAR100} & {PPBA_MEDQ_CIFAR100} & &  {PPBA_ASR_TinyImageNet}\% & {PPBA_AVGQ_TinyImageNet} & {PPBA_MEDQ_TinyImageNet} \\\\
        & & SignHunter & {SignHunter_ASR_CIFAR10}\% & {SignHunter_AVGQ_CIFAR10} & {SignHunter_MEDQ_CIFAR10} &  & {SignHunter_ASR_CIFAR100}\% & {SignHunter_AVGQ_CIFAR100} & {SignHunter_MEDQ_CIFAR100} & &  {SignHunter_ASR_TinyImageNet}\% & {SignHunter_AVGQ_TinyImageNet} & {SignHunter_MEDQ_TinyImageNet} \\\\
        & & Square Attack & {Square_ASR_CIFAR10}\% & {Square_AVGQ_CIFAR10} & {Square_MEDQ_CIFAR10} &  & {Square_ASR_CIFAR100}\% & {Square_AVGQ_CIFAR100} & {Square_MEDQ_CIFAR100} & &  {Square_ASR_TinyImageNet}\% & {Square_AVGQ_TinyImageNet} & {Square_MEDQ_TinyImageNet} \\\\
        & & NO SWITCH & {NO_SWITCH_ASR_CIFAR10}\% & {NO_SWITCH_AVGQ_CIFAR10} & {NO_SWITCH_MEDQ_CIFAR10} &  & {NO_SWITCH_ASR_CIFAR100}\% & {NO_SWITCH_AVGQ_CIFAR100} & {NO_SWITCH_MEDQ_CIFAR100} & &  {NO_SWITCH_ASR_TinyImageNet}\% & {NO_SWITCH_AVGQ_TinyImageNet} & {NO_SWITCH_MEDQ_TinyImageNet} \\\\
        & & SWITCH_neg & {SWITCH_neg_ASR_CIFAR10}\% & {SWITCH_neg_AVGQ_CIFAR10} & {SWITCH_neg_MEDQ_CIFAR10} &  & {SWITCH_neg_ASR_CIFAR100}\% & {SWITCH_neg_AVGQ_CIFAR100} & {SWITCH_neg_MEDQ_CIFAR100} & &  {SWITCH_neg_ASR_TinyImageNet}\% & {SWITCH_neg_AVGQ_TinyImageNet} & {SWITCH_neg_MEDQ_TinyImageNet} \\\\
        & & NO SWITCH$_rnd$ & {NO_SWITCH_rnd_ASR_CIFAR10}\% & {NO_SWITCH_rnd_AVGQ_CIFAR10} & {NO_SWITCH_rnd_MEDQ_CIFAR10} &  & {NO_SWITCH_rnd_ASR_CIFAR100}\% & {NO_SWITCH_rnd_AVGQ_CIFAR100} & {NO_SWITCH_rnd_MEDQ_CIFAR100} & &  {NO_SWITCH_rnd_ASR_TinyImageNet}\% & {NO_SWITCH_rnd_AVGQ_TinyImageNet} & {NO_SWITCH_rnd_MEDQ_TinyImageNet} \\\\
        & & SWITCH$_other$ & {SWITCH_other_ASR_CIFAR10}\% & {SWITCH_other_AVGQ_CIFAR10} & {SWITCH_other_MEDQ_CIFAR10} &  & {SWITCH_other_ASR_CIFAR100}\% & {SWITCH_other_AVGQ_CIFAR100} & {SWITCH_other_MEDQ_CIFAR100} & &  {SWITCH_other_ASR_TinyImageNet}\% & {SWITCH_other_AVGQ_TinyImageNet} & {SWITCH_other_MEDQ_TinyImageNet} \\\\
                """.format(
            RGF_ASR_CIFAR10=result["CIFAR-10"]["RGF"]["success_rate"],
            RGF_AVGQ_CIFAR10=result["CIFAR-10"]["RGF"][avg_q], RGF_MEDQ_CIFAR10=result["CIFAR-10"]["RGF"][med_q],
            RGF_ASR_CIFAR100=result["CIFAR-100"]["RGF"]["success_rate"],
            RGF_AVGQ_CIFAR100=result["CIFAR-100"]["RGF"][avg_q], RGF_MEDQ_CIFAR100=result["CIFAR-100"]["RGF"][med_q],
            RGF_ASR_TinyImageNet=result["TinyImageNet"]["RGF"]["success_rate"],
            RGF_AVGQ_TinyImageNet=result["TinyImageNet"]["RGF"][avg_q],
            RGF_MEDQ_TinyImageNet=result["TinyImageNet"]["RGF"][med_q],

            PRGF_ASR_CIFAR10=result["CIFAR-10"]["PRGF"]["success_rate"],
            PRGF_AVGQ_CIFAR10=result["CIFAR-10"]["PRGF"][avg_q], PRGF_MEDQ_CIFAR10=result["CIFAR-10"]["PRGF"][med_q],
            PRGF_ASR_CIFAR100=result["CIFAR-100"]["PRGF"]["success_rate"],
            PRGF_AVGQ_CIFAR100=result["CIFAR-100"]["PRGF"][avg_q],
            PRGF_MEDQ_CIFAR100=result["CIFAR-100"]["PRGF"][med_q],
            PRGF_ASR_TinyImageNet=result["TinyImageNet"]["PRGF"]["success_rate"],
            PRGF_AVGQ_TinyImageNet=result["TinyImageNet"]["PRGF"][avg_q],
            PRGF_MEDQ_TinyImageNet=result["TinyImageNet"]["PRGF"][med_q],

            Bandits_ASR_CIFAR10=result["CIFAR-10"]["Bandits"]["success_rate"],
            Bandits_AVGQ_CIFAR10=result["CIFAR-10"]["Bandits"][avg_q],
            Bandits_MEDQ_CIFAR10=result["CIFAR-10"]["Bandits"][med_q],
            Bandits_ASR_CIFAR100=result["CIFAR-100"]["Bandits"]["success_rate"],
            Bandits_AVGQ_CIFAR100=result["CIFAR-100"]["Bandits"][avg_q],
            Bandits_MEDQ_CIFAR100=result["CIFAR-100"]["Bandits"][med_q],
            Bandits_ASR_TinyImageNet=result["TinyImageNet"]["Bandits"]["success_rate"],
            Bandits_AVGQ_TinyImageNet=result["TinyImageNet"]["Bandits"][avg_q],
            Bandits_MEDQ_TinyImageNet=result["TinyImageNet"]["Bandits"][med_q],

            PPBA_ASR_CIFAR10=result["CIFAR-10"]["PPBA"]["success_rate"],
            PPBA_AVGQ_CIFAR10=result["CIFAR-10"]["PPBA"][avg_q], PPBA_MEDQ_CIFAR10=result["CIFAR-10"]["PPBA"][med_q],
            PPBA_ASR_CIFAR100=result["CIFAR-100"]["PPBA"]["success_rate"],
            PPBA_AVGQ_CIFAR100=result["CIFAR-100"]["PPBA"][avg_q],
            PPBA_MEDQ_CIFAR100=result["CIFAR-100"]["PPBA"][med_q],
            PPBA_ASR_TinyImageNet=result["TinyImageNet"]["PPBA"]["success_rate"],
            PPBA_AVGQ_TinyImageNet=result["TinyImageNet"]["PPBA"][avg_q],
            PPBA_MEDQ_TinyImageNet=result["TinyImageNet"]["PPBA"][med_q],

            SignHunter_ASR_CIFAR10=result["CIFAR-10"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_CIFAR10=result["CIFAR-10"]["SignHunter"][avg_q],
            SignHunter_MEDQ_CIFAR10=result["CIFAR-10"]["SignHunter"][med_q],
            SignHunter_ASR_CIFAR100=result["CIFAR-100"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_CIFAR100=result["CIFAR-100"]["SignHunter"][avg_q],
            SignHunter_MEDQ_CIFAR100=result["CIFAR-100"]["SignHunter"][med_q],
            SignHunter_ASR_TinyImageNet=result["TinyImageNet"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_TinyImageNet=result["TinyImageNet"]["SignHunter"][avg_q],
            SignHunter_MEDQ_TinyImageNet=result["TinyImageNet"]["SignHunter"][med_q],

            Square_ASR_CIFAR10=result["CIFAR-10"]["Square"]["success_rate"],
            Square_AVGQ_CIFAR10=result["CIFAR-10"]["Square"][avg_q],
            Square_MEDQ_CIFAR10=result["CIFAR-10"]["Square"][med_q],
            Square_ASR_CIFAR100=result["CIFAR-100"]["Square"]["success_rate"],
            Square_AVGQ_CIFAR100=result["CIFAR-100"]["Square"][avg_q],
            Square_MEDQ_CIFAR100=result["CIFAR-100"]["Square"][med_q],
            Square_ASR_TinyImageNet=result["TinyImageNet"]["Square"]["success_rate"],
            Square_AVGQ_TinyImageNet=result["TinyImageNet"]["Square"][avg_q],
            Square_MEDQ_TinyImageNet=result["TinyImageNet"]["Square"][med_q],

            NO_SWITCH_ASR_CIFAR10=result["CIFAR-10"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH"][med_q],
            NO_SWITCH_ASR_CIFAR100=result["CIFAR-100"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH"][med_q],
            NO_SWITCH_ASR_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"][med_q],

            SWITCH_neg_ASR_CIFAR10=result["CIFAR-10"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_CIFAR10=result["CIFAR-10"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_CIFAR10=result["CIFAR-10"]["SWITCH_neg"][med_q],
            SWITCH_neg_ASR_CIFAR100=result["CIFAR-100"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_CIFAR100=result["CIFAR-100"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_CIFAR100=result["CIFAR-100"]["SWITCH_neg"][med_q],
            SWITCH_neg_ASR_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"][med_q],

            NO_SWITCH_rnd_ASR_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"][med_q],
            NO_SWITCH_rnd_ASR_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"][med_q],
            NO_SWITCH_rnd_ASR_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"][med_q],

            SWITCH_other_ASR_CIFAR10=result["CIFAR-10"]["SWITCH_other"]["success_rate"],
            SWITCH_other_AVGQ_CIFAR10=result["CIFAR-10"]["SWITCH_other"][avg_q],
            SWITCH_other_MEDQ_CIFAR10=result["CIFAR-10"]["SWITCH_other"][med_q],
            SWITCH_other_ASR_CIFAR100=result["CIFAR-100"]["SWITCH_other"]["success_rate"],
            SWITCH_other_AVGQ_CIFAR100=result["CIFAR-100"]["SWITCH_other"][avg_q],
            SWITCH_other_MEDQ_CIFAR100=result["CIFAR-100"]["SWITCH_other"][med_q],
            SWITCH_other_ASR_TinyImageNet=result["TinyImageNet"]["SWITCH_other"]["success_rate"],
            SWITCH_other_AVGQ_TinyImageNet=result["TinyImageNet"]["SWITCH_other"][avg_q],
            SWITCH_other_MEDQ_TinyImageNet=result["TinyImageNet"]["SWITCH_other"][med_q],
        )
        )



def draw_tables_without_SWITCH_other(norm, result):
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    if norm == "linf":
        print("""
& & RGF & {RGF_ASR_CIFAR10}\% & {RGF_AVGQ_CIFAR10} & {RGF_MEDQ_CIFAR10} &  & {RGF_ASR_CIFAR100}\% & {RGF_AVGQ_CIFAR100} & {RGF_MEDQ_CIFAR100} & &  {RGF_ASR_TinyImageNet}\% & {RGF_AVGQ_TinyImageNet} & {RGF_MEDQ_TinyImageNet} \\\\
& & P-RGF & {PRGF_ASR_CIFAR10}\% & {PRGF_AVGQ_CIFAR10} & {PRGF_MEDQ_CIFAR10} &  & {PRGF_ASR_CIFAR100}\% & {PRGF_AVGQ_CIFAR100} & {PRGF_MEDQ_CIFAR100} & &  {PRGF_ASR_TinyImageNet}\% & {PRGF_AVGQ_TinyImageNet} & {PRGF_MEDQ_TinyImageNet} \\\\
& & Bandits & {Bandits_ASR_CIFAR10}\% & {Bandits_AVGQ_CIFAR10} & {Bandits_MEDQ_CIFAR10} &  & {Bandits_ASR_CIFAR100}\% & {Bandits_AVGQ_CIFAR100} & {Bandits_MEDQ_CIFAR100} & &  {Bandits_ASR_TinyImageNet}\% & {Bandits_AVGQ_TinyImageNet} & {Bandits_MEDQ_TinyImageNet} \\\\
& & PPBA & {PPBA_ASR_CIFAR10}\% & {PPBA_AVGQ_CIFAR10} & {PPBA_MEDQ_CIFAR10} &  & {PPBA_ASR_CIFAR100}\% & {PPBA_AVGQ_CIFAR100} & {PPBA_MEDQ_CIFAR100} & &  {PPBA_ASR_TinyImageNet}\% & {PPBA_AVGQ_TinyImageNet} & {PPBA_MEDQ_TinyImageNet} \\\\
& & Parsimonious & {Parsimonious_ASR_CIFAR10}\% & {Parsimonious_AVGQ_CIFAR10} & {Parsimonious_MEDQ_CIFAR10} &  & {Parsimonious_ASR_CIFAR100}\% & {Parsimonious_AVGQ_CIFAR100} & {Parsimonious_MEDQ_CIFAR100} & &  {Parsimonious_ASR_TinyImageNet}\% & {Parsimonious_AVGQ_TinyImageNet} & {Parsimonious_MEDQ_TinyImageNet} \\\\ 
& & SignHunter & {SignHunter_ASR_CIFAR10}\% & {SignHunter_AVGQ_CIFAR10} & {SignHunter_MEDQ_CIFAR10} &  & {SignHunter_ASR_CIFAR100}\% & {SignHunter_AVGQ_CIFAR100} & {SignHunter_MEDQ_CIFAR100} & &  {SignHunter_ASR_TinyImageNet}\% & {SignHunter_AVGQ_TinyImageNet} & {SignHunter_MEDQ_TinyImageNet} \\\\
& & Square Attack & {Square_ASR_CIFAR10}\% & {Square_AVGQ_CIFAR10} & {Square_MEDQ_CIFAR10} &  & {Square_ASR_CIFAR100}\% & {Square_AVGQ_CIFAR100} & {Square_MEDQ_CIFAR100} & &  {Square_ASR_TinyImageNet}\% & {Square_AVGQ_TinyImageNet} & {Square_MEDQ_TinyImageNet} \\\\
& & NO SWITCH & {NO_SWITCH_ASR_CIFAR10}\% & {NO_SWITCH_AVGQ_CIFAR10} & {NO_SWITCH_MEDQ_CIFAR10} &  & {NO_SWITCH_ASR_CIFAR100}\% & {NO_SWITCH_AVGQ_CIFAR100} & {NO_SWITCH_MEDQ_CIFAR100} & &  {NO_SWITCH_ASR_TinyImageNet}\% & {NO_SWITCH_AVGQ_TinyImageNet} & {NO_SWITCH_MEDQ_TinyImageNet} \\\\
& & SWITCH_neg & {SWITCH_neg_ASR_CIFAR10}\% & {SWITCH_neg_AVGQ_CIFAR10} & {SWITCH_neg_MEDQ_CIFAR10} &  & {SWITCH_neg_ASR_CIFAR100}\% & {SWITCH_neg_AVGQ_CIFAR100} & {SWITCH_neg_MEDQ_CIFAR100} & &  {SWITCH_neg_ASR_TinyImageNet}\% & {SWITCH_neg_AVGQ_TinyImageNet} & {SWITCH_neg_MEDQ_TinyImageNet} \\\\
        """.format(
            RGF_ASR_CIFAR10=result["CIFAR-10"]["RGF"]["success_rate"],RGF_AVGQ_CIFAR10=result["CIFAR-10"]["RGF"][avg_q],RGF_MEDQ_CIFAR10=result["CIFAR-10"]["RGF"][med_q],
            RGF_ASR_CIFAR100=result["CIFAR-100"]["RGF"]["success_rate"], RGF_AVGQ_CIFAR100=result["CIFAR-100"]["RGF"][avg_q], RGF_MEDQ_CIFAR100=result["CIFAR-100"]["RGF"][med_q],
            RGF_ASR_TinyImageNet=result["TinyImageNet"]["RGF"]["success_rate"], RGF_AVGQ_TinyImageNet=result["TinyImageNet"]["RGF"][avg_q], RGF_MEDQ_TinyImageNet=result["TinyImageNet"]["RGF"][med_q],

            PRGF_ASR_CIFAR10=result["CIFAR-10"]["PRGF"]["success_rate"],
            PRGF_AVGQ_CIFAR10=result["CIFAR-10"]["PRGF"][avg_q], PRGF_MEDQ_CIFAR10=result["CIFAR-10"]["PRGF"][med_q],
            PRGF_ASR_CIFAR100=result["CIFAR-100"]["PRGF"]["success_rate"],
            PRGF_AVGQ_CIFAR100=result["CIFAR-100"]["PRGF"][avg_q],
            PRGF_MEDQ_CIFAR100=result["CIFAR-100"]["PRGF"][med_q],
            PRGF_ASR_TinyImageNet=result["TinyImageNet"]["PRGF"]["success_rate"],
            PRGF_AVGQ_TinyImageNet=result["TinyImageNet"]["PRGF"][avg_q],
            PRGF_MEDQ_TinyImageNet=result["TinyImageNet"]["PRGF"][med_q],

            Bandits_ASR_CIFAR10=result["CIFAR-10"]["Bandits"]["success_rate"],
            Bandits_AVGQ_CIFAR10=result["CIFAR-10"]["Bandits"][avg_q],
            Bandits_MEDQ_CIFAR10=result["CIFAR-10"]["Bandits"][med_q],
            Bandits_ASR_CIFAR100=result["CIFAR-100"]["Bandits"]["success_rate"],
            Bandits_AVGQ_CIFAR100=result["CIFAR-100"]["Bandits"][avg_q],
            Bandits_MEDQ_CIFAR100=result["CIFAR-100"]["Bandits"][med_q],
            Bandits_ASR_TinyImageNet=result["TinyImageNet"]["Bandits"]["success_rate"],
            Bandits_AVGQ_TinyImageNet=result["TinyImageNet"]["Bandits"][avg_q],
            Bandits_MEDQ_TinyImageNet=result["TinyImageNet"]["Bandits"][med_q],

            PPBA_ASR_CIFAR10=result["CIFAR-10"]["PPBA"]["success_rate"],
            PPBA_AVGQ_CIFAR10=result["CIFAR-10"]["PPBA"][avg_q], PPBA_MEDQ_CIFAR10=result["CIFAR-10"]["PPBA"][med_q],
            PPBA_ASR_CIFAR100=result["CIFAR-100"]["PPBA"]["success_rate"],
            PPBA_AVGQ_CIFAR100=result["CIFAR-100"]["PPBA"][avg_q],
            PPBA_MEDQ_CIFAR100=result["CIFAR-100"]["PPBA"][med_q],
            PPBA_ASR_TinyImageNet=result["TinyImageNet"]["PPBA"]["success_rate"],
            PPBA_AVGQ_TinyImageNet=result["TinyImageNet"]["PPBA"][avg_q],
            PPBA_MEDQ_TinyImageNet=result["TinyImageNet"]["PPBA"][med_q],

            Parsimonious_ASR_CIFAR10=result["CIFAR-10"]["Parsimonious"]["success_rate"],
            Parsimonious_AVGQ_CIFAR10=result["CIFAR-10"]["Parsimonious"][avg_q],
            Parsimonious_MEDQ_CIFAR10=result["CIFAR-10"]["Parsimonious"][med_q],
            Parsimonious_ASR_CIFAR100=result["CIFAR-100"]["Parsimonious"]["success_rate"],
            Parsimonious_AVGQ_CIFAR100=result["CIFAR-100"]["Parsimonious"][avg_q],
            Parsimonious_MEDQ_CIFAR100=result["CIFAR-100"]["Parsimonious"][med_q],
            Parsimonious_ASR_TinyImageNet=result["TinyImageNet"]["Parsimonious"]["success_rate"],
            Parsimonious_AVGQ_TinyImageNet=result["TinyImageNet"]["Parsimonious"][avg_q],
            Parsimonious_MEDQ_TinyImageNet=result["TinyImageNet"]["Parsimonious"][med_q],

            SignHunter_ASR_CIFAR10=result["CIFAR-10"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_CIFAR10=result["CIFAR-10"]["SignHunter"][avg_q],
            SignHunter_MEDQ_CIFAR10=result["CIFAR-10"]["SignHunter"][med_q],
            SignHunter_ASR_CIFAR100=result["CIFAR-100"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_CIFAR100=result["CIFAR-100"]["SignHunter"][avg_q],
            SignHunter_MEDQ_CIFAR100=result["CIFAR-100"]["SignHunter"][med_q],
            SignHunter_ASR_TinyImageNet=result["TinyImageNet"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_TinyImageNet=result["TinyImageNet"]["SignHunter"][avg_q],
            SignHunter_MEDQ_TinyImageNet=result["TinyImageNet"]["SignHunter"][med_q],

            Square_ASR_CIFAR10=result["CIFAR-10"]["Square"]["success_rate"],
            Square_AVGQ_CIFAR10=result["CIFAR-10"]["Square"][avg_q],
            Square_MEDQ_CIFAR10=result["CIFAR-10"]["Square"][med_q],
            Square_ASR_CIFAR100=result["CIFAR-100"]["Square"]["success_rate"],
            Square_AVGQ_CIFAR100=result["CIFAR-100"]["Square"][avg_q],
            Square_MEDQ_CIFAR100=result["CIFAR-100"]["Square"][med_q],
            Square_ASR_TinyImageNet=result["TinyImageNet"]["Square"]["success_rate"],
            Square_AVGQ_TinyImageNet=result["TinyImageNet"]["Square"][avg_q],
            Square_MEDQ_TinyImageNet=result["TinyImageNet"]["Square"][med_q],

            NO_SWITCH_ASR_CIFAR10=result["CIFAR-10"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH"][med_q],
            NO_SWITCH_ASR_CIFAR100=result["CIFAR-100"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH"][med_q],
            NO_SWITCH_ASR_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"][med_q],

            SWITCH_neg_ASR_CIFAR10=result["CIFAR-10"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_CIFAR10=result["CIFAR-10"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_CIFAR10=result["CIFAR-10"]["SWITCH_neg"][med_q],
            SWITCH_neg_ASR_CIFAR100=result["CIFAR-100"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_CIFAR100=result["CIFAR-100"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_CIFAR100=result["CIFAR-100"]["SWITCH_neg"][med_q],
            SWITCH_neg_ASR_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"][med_q],

            NO_SWITCH_rnd_ASR_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"][med_q],
            NO_SWITCH_rnd_ASR_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"][med_q],
            NO_SWITCH_rnd_ASR_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"][med_q],


        )
              )
    else:
        print("""
        & & RGF & {RGF_ASR_CIFAR10}\% & {RGF_AVGQ_CIFAR10} & {RGF_MEDQ_CIFAR10} &  & {RGF_ASR_CIFAR100}\% & {RGF_AVGQ_CIFAR100} & {RGF_MEDQ_CIFAR100} & &  {RGF_ASR_TinyImageNet}\% & {RGF_AVGQ_TinyImageNet} & {RGF_MEDQ_TinyImageNet} \\\\
        & & P-RGF & {PRGF_ASR_CIFAR10}\% & {PRGF_AVGQ_CIFAR10} & {PRGF_MEDQ_CIFAR10} &  & {PRGF_ASR_CIFAR100}\% & {PRGF_AVGQ_CIFAR100} & {PRGF_MEDQ_CIFAR100} & &  {PRGF_ASR_TinyImageNet}\% & {PRGF_AVGQ_TinyImageNet} & {PRGF_MEDQ_TinyImageNet} \\\\
        & & Bandits & {Bandits_ASR_CIFAR10}\% & {Bandits_AVGQ_CIFAR10} & {Bandits_MEDQ_CIFAR10} &  & {Bandits_ASR_CIFAR100}\% & {Bandits_AVGQ_CIFAR100} & {Bandits_MEDQ_CIFAR100} & &  {Bandits_ASR_TinyImageNet}\% & {Bandits_AVGQ_TinyImageNet} & {Bandits_MEDQ_TinyImageNet} \\\\
        & & PPBA & {PPBA_ASR_CIFAR10}\% & {PPBA_AVGQ_CIFAR10} & {PPBA_MEDQ_CIFAR10} &  & {PPBA_ASR_CIFAR100}\% & {PPBA_AVGQ_CIFAR100} & {PPBA_MEDQ_CIFAR100} & &  {PPBA_ASR_TinyImageNet}\% & {PPBA_AVGQ_TinyImageNet} & {PPBA_MEDQ_TinyImageNet} \\\\
        & & SignHunter & {SignHunter_ASR_CIFAR10}\% & {SignHunter_AVGQ_CIFAR10} & {SignHunter_MEDQ_CIFAR10} &  & {SignHunter_ASR_CIFAR100}\% & {SignHunter_AVGQ_CIFAR100} & {SignHunter_MEDQ_CIFAR100} & &  {SignHunter_ASR_TinyImageNet}\% & {SignHunter_AVGQ_TinyImageNet} & {SignHunter_MEDQ_TinyImageNet} \\\\
        & & Square Attack & {Square_ASR_CIFAR10}\% & {Square_AVGQ_CIFAR10} & {Square_MEDQ_CIFAR10} &  & {Square_ASR_CIFAR100}\% & {Square_AVGQ_CIFAR100} & {Square_MEDQ_CIFAR100} & &  {Square_ASR_TinyImageNet}\% & {Square_AVGQ_TinyImageNet} & {Square_MEDQ_TinyImageNet} \\\\
        & & NO SWITCH & {NO_SWITCH_ASR_CIFAR10}\% & {NO_SWITCH_AVGQ_CIFAR10} & {NO_SWITCH_MEDQ_CIFAR10} &  & {NO_SWITCH_ASR_CIFAR100}\% & {NO_SWITCH_AVGQ_CIFAR100} & {NO_SWITCH_MEDQ_CIFAR100} & &  {NO_SWITCH_ASR_TinyImageNet}\% & {NO_SWITCH_AVGQ_TinyImageNet} & {NO_SWITCH_MEDQ_TinyImageNet} \\\\
        & & SWITCH_neg & {SWITCH_neg_ASR_CIFAR10}\% & {SWITCH_neg_AVGQ_CIFAR10} & {SWITCH_neg_MEDQ_CIFAR10} &  & {SWITCH_neg_ASR_CIFAR100}\% & {SWITCH_neg_AVGQ_CIFAR100} & {SWITCH_neg_MEDQ_CIFAR100} & &  {SWITCH_neg_ASR_TinyImageNet}\% & {SWITCH_neg_AVGQ_TinyImageNet} & {SWITCH_neg_MEDQ_TinyImageNet} \\\\
        & & NO SWITCH$_rnd$ & {NO_SWITCH_rnd_ASR_CIFAR10}\% & {NO_SWITCH_rnd_AVGQ_CIFAR10} & {NO_SWITCH_rnd_MEDQ_CIFAR10} &  & {NO_SWITCH_rnd_ASR_CIFAR100}\% & {NO_SWITCH_rnd_AVGQ_CIFAR100} & {NO_SWITCH_rnd_MEDQ_CIFAR100} & &  {NO_SWITCH_rnd_ASR_TinyImageNet}\% & {NO_SWITCH_rnd_AVGQ_TinyImageNet} & {NO_SWITCH_rnd_MEDQ_TinyImageNet} \\\\
                """.format(
            RGF_ASR_CIFAR10=result["CIFAR-10"]["RGF"]["success_rate"],
            RGF_AVGQ_CIFAR10=result["CIFAR-10"]["RGF"][avg_q], RGF_MEDQ_CIFAR10=result["CIFAR-10"]["RGF"][med_q],
            RGF_ASR_CIFAR100=result["CIFAR-100"]["RGF"]["success_rate"],
            RGF_AVGQ_CIFAR100=result["CIFAR-100"]["RGF"][avg_q], RGF_MEDQ_CIFAR100=result["CIFAR-100"]["RGF"][med_q],
            RGF_ASR_TinyImageNet=result["TinyImageNet"]["RGF"]["success_rate"],
            RGF_AVGQ_TinyImageNet=result["TinyImageNet"]["RGF"][avg_q],
            RGF_MEDQ_TinyImageNet=result["TinyImageNet"]["RGF"][med_q],

            PRGF_ASR_CIFAR10=result["CIFAR-10"]["PRGF"]["success_rate"],
            PRGF_AVGQ_CIFAR10=result["CIFAR-10"]["PRGF"][avg_q], PRGF_MEDQ_CIFAR10=result["CIFAR-10"]["PRGF"][med_q],
            PRGF_ASR_CIFAR100=result["CIFAR-100"]["PRGF"]["success_rate"],
            PRGF_AVGQ_CIFAR100=result["CIFAR-100"]["PRGF"][avg_q],
            PRGF_MEDQ_CIFAR100=result["CIFAR-100"]["PRGF"][med_q],
            PRGF_ASR_TinyImageNet=result["TinyImageNet"]["PRGF"]["success_rate"],
            PRGF_AVGQ_TinyImageNet=result["TinyImageNet"]["PRGF"][avg_q],
            PRGF_MEDQ_TinyImageNet=result["TinyImageNet"]["PRGF"][med_q],

            Bandits_ASR_CIFAR10=result["CIFAR-10"]["Bandits"]["success_rate"],
            Bandits_AVGQ_CIFAR10=result["CIFAR-10"]["Bandits"][avg_q],
            Bandits_MEDQ_CIFAR10=result["CIFAR-10"]["Bandits"][med_q],
            Bandits_ASR_CIFAR100=result["CIFAR-100"]["Bandits"]["success_rate"],
            Bandits_AVGQ_CIFAR100=result["CIFAR-100"]["Bandits"][avg_q],
            Bandits_MEDQ_CIFAR100=result["CIFAR-100"]["Bandits"][med_q],
            Bandits_ASR_TinyImageNet=result["TinyImageNet"]["Bandits"]["success_rate"],
            Bandits_AVGQ_TinyImageNet=result["TinyImageNet"]["Bandits"][avg_q],
            Bandits_MEDQ_TinyImageNet=result["TinyImageNet"]["Bandits"][med_q],

            PPBA_ASR_CIFAR10=result["CIFAR-10"]["PPBA"]["success_rate"],
            PPBA_AVGQ_CIFAR10=result["CIFAR-10"]["PPBA"][avg_q], PPBA_MEDQ_CIFAR10=result["CIFAR-10"]["PPBA"][med_q],
            PPBA_ASR_CIFAR100=result["CIFAR-100"]["PPBA"]["success_rate"],
            PPBA_AVGQ_CIFAR100=result["CIFAR-100"]["PPBA"][avg_q],
            PPBA_MEDQ_CIFAR100=result["CIFAR-100"]["PPBA"][med_q],
            PPBA_ASR_TinyImageNet=result["TinyImageNet"]["PPBA"]["success_rate"],
            PPBA_AVGQ_TinyImageNet=result["TinyImageNet"]["PPBA"][avg_q],
            PPBA_MEDQ_TinyImageNet=result["TinyImageNet"]["PPBA"][med_q],

            SignHunter_ASR_CIFAR10=result["CIFAR-10"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_CIFAR10=result["CIFAR-10"]["SignHunter"][avg_q],
            SignHunter_MEDQ_CIFAR10=result["CIFAR-10"]["SignHunter"][med_q],
            SignHunter_ASR_CIFAR100=result["CIFAR-100"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_CIFAR100=result["CIFAR-100"]["SignHunter"][avg_q],
            SignHunter_MEDQ_CIFAR100=result["CIFAR-100"]["SignHunter"][med_q],
            SignHunter_ASR_TinyImageNet=result["TinyImageNet"]["SignHunter"]["success_rate"],
            SignHunter_AVGQ_TinyImageNet=result["TinyImageNet"]["SignHunter"][avg_q],
            SignHunter_MEDQ_TinyImageNet=result["TinyImageNet"]["SignHunter"][med_q],

            Square_ASR_CIFAR10=result["CIFAR-10"]["Square"]["success_rate"],
            Square_AVGQ_CIFAR10=result["CIFAR-10"]["Square"][avg_q],
            Square_MEDQ_CIFAR10=result["CIFAR-10"]["Square"][med_q],
            Square_ASR_CIFAR100=result["CIFAR-100"]["Square"]["success_rate"],
            Square_AVGQ_CIFAR100=result["CIFAR-100"]["Square"][avg_q],
            Square_MEDQ_CIFAR100=result["CIFAR-100"]["Square"][med_q],
            Square_ASR_TinyImageNet=result["TinyImageNet"]["Square"]["success_rate"],
            Square_AVGQ_TinyImageNet=result["TinyImageNet"]["Square"][avg_q],
            Square_MEDQ_TinyImageNet=result["TinyImageNet"]["Square"][med_q],

            NO_SWITCH_ASR_CIFAR10=result["CIFAR-10"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH"][med_q],
            NO_SWITCH_ASR_CIFAR100=result["CIFAR-100"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH"][med_q],
            NO_SWITCH_ASR_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"]["success_rate"],
            NO_SWITCH_AVGQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"][avg_q],
            NO_SWITCH_MEDQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH"][med_q],

            SWITCH_neg_ASR_CIFAR10=result["CIFAR-10"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_CIFAR10=result["CIFAR-10"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_CIFAR10=result["CIFAR-10"]["SWITCH_neg"][med_q],
            SWITCH_neg_ASR_CIFAR100=result["CIFAR-100"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_CIFAR100=result["CIFAR-100"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_CIFAR100=result["CIFAR-100"]["SWITCH_neg"][med_q],
            SWITCH_neg_ASR_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"]["success_rate"],
            SWITCH_neg_AVGQ_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"][avg_q],
            SWITCH_neg_MEDQ_TinyImageNet=result["TinyImageNet"]["SWITCH_neg"][med_q],

            NO_SWITCH_rnd_ASR_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_CIFAR10=result["CIFAR-10"]["NO_SWITCH_rnd"][med_q],
            NO_SWITCH_rnd_ASR_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_CIFAR100=result["CIFAR-100"]["NO_SWITCH_rnd"][med_q],
            NO_SWITCH_rnd_ASR_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"]["success_rate"],
            NO_SWITCH_rnd_AVGQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"][avg_q],
            NO_SWITCH_rnd_MEDQ_TinyImageNet=result["TinyImageNet"]["NO_SWITCH_rnd"][med_q],

        )
        )


if __name__ == "__main__":
    datasets = ["CIFAR-10", "CIFAR-100", "TinyImageNet"]
    results = {}
    norm = "linf"
    defense_method = "jpeg"
    for dataset in datasets:
        targeted = False
        if "CIFAR" in dataset:
            arch = "resnet-50"
        else:
            arch = "resnet50"
        result_archs={}
        result = fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, defense_method)
        results[dataset] = result

    draw_tables_without_SWITCH_other(norm, results)
    print("THIS IS {} {} on {}".format(norm, "untargeted" if not targeted else "targeted", defense_method))
