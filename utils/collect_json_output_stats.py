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


def read_json_and_extract(json_path):
    with open(json_path, "r") as file_obj:
        json_content = json.load(file_obj)
        switch_ratio = new_round(json_content["switch_ratio"] * 100, 1)
        improved_from_last_iter_after_switched = new_round(json_content["ratio_improved_after_switch"]*100, 1)
        swtiched_loss_bigger_than_x_temp = new_round(json_content["ratio_swtiched_loss_bigger_than_orig"]*100, 1)
        return switch_ratio, improved_from_last_iter_after_switched, swtiched_loss_bigger_than_x_temp

def get_file_name_list(dataset, norm, targeted):
    folder_path_dict = {}

    for method in ["SWITCH_neg"]:
        if method == "SWITCH_neg":
            file_path = "/home1/machen/query_based_black_box_attack/logs/" + get_SWITCH_neg_dir_name(dataset, norm, targeted, "increment", False)
        elif method == "SWITCH_other":
            file_path = "/home1/machen/query_based_black_box_attack/logs/" + get_SWITCH_rnd_dir_name(dataset, norm,
                                                                                                     targeted,
                                                                                                     "increment", False)
        assert os.path.exists(file_path), "{} does not exist".format(file_path)
        folder_path_dict[method] = file_path
    return folder_path_dict

def fetch_all_json_content(datasets, targeted, archs):
    norms = ["l2", "linf"]
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for dataset in datasets:
        for norm in norms:
            folder_list = get_file_name_list(dataset, norm, targeted)
            for method, folder in folder_list.items():
                for arch in archs:
                    file_path = folder + "/{}_result_stats.json".format(arch)
                    assert os.path.exists(file_path), "{} does not exist!".format(file_path)
                    switch_ratio, improved_from_last_iter_after_switched, swtiched_loss_bigger_than_x_temp = read_json_and_extract(file_path)
                    result[dataset][norm][method][arch] = {"switch_ratio":switch_ratio, "switched_loss_increased_from_lastiter_ratio":improved_from_last_iter_after_switched,
                                      "switched_loss_improved_x_temp_ratio": swtiched_loss_bigger_than_x_temp}
    return result



def get_SWITCH_rnd_dir_name(dataset,  norm, targeted, target_type, attack_defense):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    surrogate_models = "resnet-110,densenet-bc-100-12" if "CIFAR" in dataset else "resnet101,resnet152"
    loss = 'cw' if not targeted else "xent"

    if attack_defense:
        dirname = 'SWITCH_rnd_save_stats_attack_on_defensive_model_using_{}-{}-{}_loss-{}-{}'.format(surrogate_models, dataset, loss, norm, target_str)
    else:
        # SWITCH_rnd_save_stats_attack_using_resnet-110,densenet-bc-100-12-CIFAR-10-cw_loss-linf-untargeted
        dirname = 'SWITCH_rnd_save_stats_attack_using_{}-{}-{}_loss-{}-{}'.format(surrogate_models, dataset, loss, norm, target_str)
    return dirname

def get_SWITCH_neg_dir_name(dataset,  norm, targeted, target_type, attack_defense):
    # SWITCH_neg_save-CIFAR-10-xent_lr_0.003-loss-linf-targeted_increment
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    loss = 'cw' if not targeted else "xent"

    if attack_defense:
        dirname = 'SWITCH_neg_save_stats_attack_on_defensive_model-{}-{}_loss-{}-{}'.format(dataset, loss, norm, target_str)
    else:
        # SWITCH_neg_save_stats_attack-CIFAR-10-xent_loss-linf-targeted_increment
        dirname = 'SWITCH_neg_save_stats_attack-{}-{}_loss-{}-{}'.format(dataset, loss, norm, target_str)
    return dirname


def draw_tables_CIFAR_only_SWITCH_neg(result):
    print("""
    CIFAR-10 & $\ell_2$ & {CIFAR10_l2_SWITCH_neg_PN_switch_ratio}\% & {CIFAR10_l2_SWITCH_neg_GDAS_switch_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN28_switch_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN40_switch_ratio}\% & {CIFAR10_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\% \\\\
& $\ell_\infty$ & {CIFAR10_linf_SWITCH_neg_PN_switch_ratio}\% & {CIFAR10_linf_SWITCH_neg_GDAS_switch_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN28_switch_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN40_switch_ratio}\% & {CIFAR10_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\% \\\\
\midrule
CIFAR-100 & $\ell_2$  & {CIFAR100_l2_SWITCH_neg_PN_switch_ratio}\% & {CIFAR100_l2_SWITCH_neg_GDAS_switch_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN28_switch_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN40_switch_ratio}\% & {CIFAR100_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\% \\\\ 
   & $\ell_\infty$ & {CIFAR100_linf_SWITCH_neg_PN_switch_ratio}\% & {CIFAR100_linf_SWITCH_neg_GDAS_switch_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN28_switch_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN40_switch_ratio}\% & {CIFAR100_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\% \\\\
    """.format(
        CIFAR10_l2_SWITCH_neg_PN_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switch_ratio"],
        CIFAR10_l2_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR10_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR10_linf_SWITCH_neg_PN_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR10_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_l2_SWITCH_neg_PN_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switch_ratio"],
        CIFAR100_l2_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_linf_SWITCH_neg_PN_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],
    ))

def draw_tables_CIFAR(result):
    print("""
CIFAR-10 & SWITCH$_neg$ & \ell_2 & {CIFAR10_l2_SWITCH_neg_PN_switch_ratio}\% & {CIFAR10_l2_SWITCH_neg_GDAS_switch_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN28_switch_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN40_switch_ratio}\% & {CIFAR10_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\% & \ell_\infty & {CIFAR10_linf_SWITCH_neg_PN_switch_ratio}\% & {CIFAR10_linf_SWITCH_neg_GDAS_switch_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN28_switch_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN40_switch_ratio}\% & {CIFAR10_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\% \\\\
        & SWITCH$_other$ &  & {CIFAR10_l2_SWITCH_other_PN_switch_ratio}\% & {CIFAR10_l2_SWITCH_other_GDAS_switch_ratio}\% & {CIFAR10_l2_SWITCH_other_WRN28_switch_ratio}\% & {CIFAR10_l2_SWITCH_other_WRN40_switch_ratio}\% & {CIFAR10_l2_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_l2_SWITCH_other_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_l2_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_l2_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_l2_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio}\% & & {CIFAR10_linf_SWITCH_other_PN_switch_ratio}\% & {CIFAR10_linf_SWITCH_other_GDAS_switch_ratio}\% & {CIFAR10_linf_SWITCH_other_WRN28_switch_ratio}\% & {CIFAR10_linf_SWITCH_other_WRN40_switch_ratio}\% & {CIFAR10_linf_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR10_linf_SWITCH_other_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_linf_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_linf_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR10_linf_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio}\% \\\\
\midrule
CIFAR-100 & SWITCH$_neg$ & \ell_2 & {CIFAR100_l2_SWITCH_neg_PN_switch_ratio}\% & {CIFAR100_l2_SWITCH_neg_GDAS_switch_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN28_switch_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN40_switch_ratio}\% & {CIFAR100_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\% & \ell_\infty & {CIFAR100_linf_SWITCH_neg_PN_switch_ratio}\% & {CIFAR100_linf_SWITCH_neg_GDAS_switch_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN28_switch_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN40_switch_ratio}\% & {CIFAR100_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\% \\\\
          & SWITCH$_other$ &  & {CIFAR100_l2_SWITCH_other_PN_switch_ratio}\% & {CIFAR100_l2_SWITCH_other_GDAS_switch_ratio}\% & {CIFAR100_l2_SWITCH_other_WRN28_switch_ratio}\% & {CIFAR100_l2_SWITCH_other_WRN40_switch_ratio}\% & {CIFAR100_l2_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_l2_SWITCH_other_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_l2_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_l2_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_l2_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio}\% & & {CIFAR100_linf_SWITCH_other_PN_switch_ratio}\% & {CIFAR100_linf_SWITCH_other_GDAS_switch_ratio}\% & {CIFAR100_linf_SWITCH_other_WRN28_switch_ratio}\% & {CIFAR100_linf_SWITCH_other_WRN40_switch_ratio}\% & {CIFAR100_linf_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio}\% & {CIFAR100_linf_SWITCH_other_PN_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_linf_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_linf_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio}\% & {CIFAR100_linf_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio}\% \\\\ 
    """.format(
        CIFAR10_l2_SWITCH_neg_PN_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switch_ratio"], CIFAR10_l2_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switch_ratio"], CIFAR10_l2_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switch_ratio"],

        CIFAR10_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR10_linf_SWITCH_neg_PN_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR10_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_l2_SWITCH_neg_PN_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switch_ratio"],
        CIFAR100_l2_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_linf_SWITCH_neg_PN_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR10_l2_SWITCH_other_PN_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_other"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR10_l2_SWITCH_other_GDAS_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_other"]["gdas"]["switch_ratio"],
        CIFAR10_l2_SWITCH_other_WRN28_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR10_l2_SWITCH_other_WRN40_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR10_l2_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_l2_SWITCH_other_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR10_linf_SWITCH_other_PN_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_other"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_other_GDAS_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_other"]["gdas"]["switch_ratio"],
        CIFAR10_linf_SWITCH_other_WRN28_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_other_WRN40_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR10_linf_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_linf_SWITCH_other_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_l2_SWITCH_other_PN_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_other"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR100_l2_SWITCH_other_GDAS_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_other"]["gdas"]["switch_ratio"],
        CIFAR100_l2_SWITCH_other_WRN28_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_l2_SWITCH_other_WRN40_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_l2_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_l2_SWITCH_other_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_linf_SWITCH_other_PN_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_other"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_other_GDAS_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_other"]["gdas"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_other_WRN28_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_other_WRN40_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_linf_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_linf_SWITCH_other_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"]

               )
          )

def draw_tables_CIFAR_with_slash(result):
    print("""
CIFAR-10 & SWITCH$_\\text{{neg}}$ & {CIFAR10_l2_SWITCH_neg_PN_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_PN_switch_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_GDAS_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_GDAS_switch_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_WRN28_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_WRN28_switch_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_WRN40_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_WRN40_switch_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\%}} \\\\
        & SWITCH$_\\text{{other}}$ & {CIFAR10_l2_SWITCH_other_PN_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_PN_switch_ratio}\%}} & {CIFAR10_l2_SWITCH_other_GDAS_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_GDAS_switch_ratio}\%}} & {CIFAR10_l2_SWITCH_other_WRN28_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_WRN28_switch_ratio}\%}} & {CIFAR10_l2_SWITCH_other_WRN40_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_WRN40_switch_ratio}\%}} & {CIFAR10_l2_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR10_l2_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR10_l2_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR10_l2_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR10_l2_SWITCH_other_PN_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_PN_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR10_l2_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR10_l2_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR10_l2_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR10_linf_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio}\%}} \\\\
\midrule
CIFAR-100 &  {CIFAR100_l2_SWITCH_neg_PN_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_PN_switch_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_GDAS_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_GDAS_switch_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_WRN28_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_WRN28_switch_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_WRN40_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_WRN40_switch_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio}\%}} \\\\
          & SWITCH$_\\text{{other}}$ &  {CIFAR100_l2_SWITCH_other_PN_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_PN_switch_ratio}\%}} & {CIFAR100_l2_SWITCH_other_GDAS_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_GDAS_switch_ratio}\%}} & {CIFAR100_l2_SWITCH_other_WRN28_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_WRN28_switch_ratio}\%}} & {CIFAR100_l2_SWITCH_other_WRN40_switch_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_WRN40_switch_ratio}\%}} & {CIFAR100_l2_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR100_l2_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR100_l2_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR100_l2_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio}\%}} & {CIFAR100_l2_SWITCH_other_PN_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_PN_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR100_l2_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR100_l2_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio}\%}} & {CIFAR100_l2_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio}\%/\\textcolor{{blue}}{{{CIFAR100_linf_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio}\%}} \\\\ 
    """.format(
        CIFAR10_l2_SWITCH_neg_PN_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switch_ratio"], CIFAR10_l2_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switch_ratio"], CIFAR10_l2_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switch_ratio"],

        CIFAR10_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=result["CIFAR-10"]["l2"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR10_linf_SWITCH_neg_PN_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR10_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_l2_SWITCH_neg_PN_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switch_ratio"],
        CIFAR100_l2_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_l2_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_l2_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=result["CIFAR-100"]["l2"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_linf_SWITCH_neg_PN_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_neg_GDAS_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["gdas"]["switch_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN28_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN40_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_linf_SWITCH_neg_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_linf_SWITCH_neg_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_neg_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_neg_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_neg"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR10_l2_SWITCH_other_PN_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_other"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR10_l2_SWITCH_other_GDAS_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_other"]["gdas"]["switch_ratio"],
        CIFAR10_l2_SWITCH_other_WRN28_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR10_l2_SWITCH_other_WRN40_switch_ratio=result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR10_l2_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_l2_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_l2_SWITCH_other_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_l2_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["l2"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR10_linf_SWITCH_other_PN_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_other"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_other_GDAS_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_other"]["gdas"]["switch_ratio"],
        CIFAR10_linf_SWITCH_other_WRN28_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR10_linf_SWITCH_other_WRN40_switch_ratio=result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR10_linf_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR10_linf_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR10_linf_SWITCH_other_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR10_linf_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-10"]["linf"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_l2_SWITCH_other_PN_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_other"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR100_l2_SWITCH_other_GDAS_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_other"]["gdas"]["switch_ratio"],
        CIFAR100_l2_SWITCH_other_WRN28_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_l2_SWITCH_other_WRN40_switch_ratio=result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_l2_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_l2_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_l2_SWITCH_other_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_l2_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["l2"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"],

        CIFAR100_linf_SWITCH_other_PN_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_other"]["pyramidnet272"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_other_GDAS_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_other"]["gdas"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_other_WRN28_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-28-10-drop"][
            "switch_ratio"],
        CIFAR100_linf_SWITCH_other_WRN40_switch_ratio=result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-40-10-drop"][
            "switch_ratio"],

        CIFAR100_linf_SWITCH_other_PN_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["pyramidnet272"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_other_GDAS_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["gdas"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_other_WRN28_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_increased_from_lastiter_ratio"],
        CIFAR100_linf_SWITCH_other_WRN40_switched_loss_increased_from_lastiter_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_increased_from_lastiter_ratio"],

        CIFAR100_linf_SWITCH_other_PN_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["pyramidnet272"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_other_GDAS_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["gdas"][
            "switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_other_WRN28_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-28-10-drop"]["switched_loss_improved_x_temp_ratio"],
        CIFAR100_linf_SWITCH_other_WRN40_switched_loss_improved_x_temp_ratio=
        result["CIFAR-100"]["linf"]["SWITCH_other"]["WRN-40-10-drop"]["switched_loss_improved_x_temp_ratio"]

               )
          )

def draw_tables_TinyImageNet(result):
    print("""
SWITCH$_neg$ & {TinyImageNet_l2_SWITCH_neg_D121_switch_ratio}\% & {TinyImageNet_l2_SWITCH_neg_R32_switch_ratio}\% & {TinyImageNet_l2_SWITCH_neg_R64_switch_ratio}\% & {TinyImageNet_l2_SWITCH_neg_D121_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_l2_SWITCH_neg_R32_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_l2_SWITCH_neg_R64_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_l2_SWITCH_neg_D121_switched_loss_improved_x_temp_ratio}\% & {TinyImageNet_l2_SWITCH_neg_R32_switched_loss_improved_x_temp_ratio}\% & {TinyImageNet_l2_SWITCH_neg_R64_switched_loss_improved_x_temp_ratio}\% \\\\
			SWITCH$_other$ & {TinyImageNet_l2_SWITCH_other_D121_switch_ratio}\% & {TinyImageNet_l2_SWITCH_other_R32_switch_ratio}\% & {TinyImageNet_l2_SWITCH_other_R64_switch_ratio}\% & {TinyImageNet_l2_SWITCH_other_D121_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_l2_SWITCH_other_R32_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_l2_SWITCH_other_R64_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_l2_SWITCH_other_D121_switched_loss_improved_x_temp_ratio}\% & {TinyImageNet_l2_SWITCH_other_R32_switched_loss_improved_x_temp_ratio}\% & {TinyImageNet_l2_SWITCH_other_R64_switched_loss_improved_x_temp_ratio}\% \\\\
			\midrule
SWITCH$_neg$ & {TinyImageNet_linf_SWITCH_neg_D121_switch_ratio}\% & {TinyImageNet_linf_SWITCH_neg_R32_switch_ratio}\% & {TinyImageNet_linf_SWITCH_neg_R64_switch_ratio}\% & {TinyImageNet_linf_SWITCH_neg_D121_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_linf_SWITCH_neg_R32_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_linf_SWITCH_neg_R64_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_linf_SWITCH_neg_D121_switched_loss_improved_x_temp_ratio}\% & {TinyImageNet_linf_SWITCH_neg_R32_switched_loss_improved_x_temp_ratio}\% & {TinyImageNet_linf_SWITCH_neg_R64_switched_loss_improved_x_temp_ratio}\% \\\\
			SWITCH$_other$ & {TinyImageNet_linf_SWITCH_other_D121_switch_ratio}\% & {TinyImageNet_linf_SWITCH_other_R32_switch_ratio}\% & {TinyImageNet_linf_SWITCH_other_R64_switch_ratio}\% & {TinyImageNet_linf_SWITCH_other_D121_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_linf_SWITCH_other_R32_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_linf_SWITCH_other_R64_switched_loss_increased_from_lastiter_ratio}\% & {TinyImageNet_linf_SWITCH_other_D121_switched_loss_improved_x_temp_ratio}\% & {TinyImageNet_linf_SWITCH_other_R32_switched_loss_improved_x_temp_ratio}\% & {TinyImageNet_linf_SWITCH_other_R64_switched_loss_improved_x_temp_ratio}\% \\\\
    """.format(
        TinyImageNet_l2_SWITCH_neg_D121_switch_ratio=result["TinyImageNet"]["l2"]["SWITCH_neg"]["densenet121"]["switch_ratio"], TinyImageNet_l2_SWITCH_neg_R32_switch_ratio=result["TinyImageNet"]["l2"]["SWITCH_neg"]["resnext32_4"]["switch_ratio"],
        TinyImageNet_l2_SWITCH_neg_R64_switch_ratio=result["TinyImageNet"]["l2"]["SWITCH_neg"]["resnext64_4"][
            "switch_ratio"],
        TinyImageNet_l2_SWITCH_other_D121_switch_ratio=result["TinyImageNet"]["l2"]["SWITCH_other"]["densenet121"][
            "switch_ratio"],
        TinyImageNet_l2_SWITCH_other_R32_switch_ratio=result["TinyImageNet"]["l2"]["SWITCH_other"]["resnext32_4"][
            "switch_ratio"],
        TinyImageNet_l2_SWITCH_other_R64_switch_ratio=result["TinyImageNet"]["l2"]["SWITCH_other"]["resnext64_4"][
            "switch_ratio"],
        TinyImageNet_linf_SWITCH_neg_D121_switch_ratio=result["TinyImageNet"]["linf"]["SWITCH_neg"]["densenet121"][
            "switch_ratio"],
        TinyImageNet_linf_SWITCH_neg_R32_switch_ratio=result["TinyImageNet"]["linf"]["SWITCH_neg"]["resnext32_4"][
            "switch_ratio"],
        TinyImageNet_linf_SWITCH_neg_R64_switch_ratio=result["TinyImageNet"]["linf"]["SWITCH_neg"]["resnext64_4"][
            "switch_ratio"],
        TinyImageNet_linf_SWITCH_other_D121_switch_ratio=result["TinyImageNet"]["linf"]["SWITCH_other"]["densenet121"][
            "switch_ratio"],
        TinyImageNet_linf_SWITCH_other_R32_switch_ratio=result["TinyImageNet"]["linf"]["SWITCH_other"]["resnext32_4"][
            "switch_ratio"],
        TinyImageNet_linf_SWITCH_other_R64_switch_ratio=result["TinyImageNet"]["linf"]["SWITCH_other"]["resnext64_4"][
            "switch_ratio"],

        TinyImageNet_l2_SWITCH_neg_D121_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_neg"]["densenet121"]["switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_l2_SWITCH_neg_R32_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_neg"]["resnext32_4"]["switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_l2_SWITCH_neg_R64_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_neg"]["resnext64_4"][
            "switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_l2_SWITCH_other_D121_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_other"]["densenet121"][
            "switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_l2_SWITCH_other_R32_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_other"]["resnext32_4"][
            "switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_l2_SWITCH_other_R64_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_other"]["resnext64_4"][
            "switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_linf_SWITCH_neg_D121_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_neg"]["densenet121"][
            "switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_linf_SWITCH_neg_R32_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_neg"]["resnext32_4"][
            "switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_linf_SWITCH_neg_R64_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_neg"]["resnext64_4"][
            "switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_linf_SWITCH_other_D121_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_other"]["densenet121"][
            "switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_linf_SWITCH_other_R32_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_other"]["resnext32_4"][
            "switched_loss_increased_from_lastiter_ratio"],
        TinyImageNet_linf_SWITCH_other_R64_switched_loss_increased_from_lastiter_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_other"]["resnext64_4"][
            "switched_loss_increased_from_lastiter_ratio"],

        TinyImageNet_l2_SWITCH_neg_D121_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_neg"]["densenet121"]["switched_loss_improved_x_temp_ratio"],
        TinyImageNet_l2_SWITCH_neg_R32_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_neg"]["resnext32_4"]["switched_loss_improved_x_temp_ratio"],
        TinyImageNet_l2_SWITCH_neg_R64_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_neg"]["resnext64_4"][
            "switched_loss_improved_x_temp_ratio"],
        TinyImageNet_l2_SWITCH_other_D121_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_other"]["densenet121"][
            "switched_loss_improved_x_temp_ratio"],
        TinyImageNet_l2_SWITCH_other_R32_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_other"]["resnext32_4"][
            "switched_loss_improved_x_temp_ratio"],
        TinyImageNet_l2_SWITCH_other_R64_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["l2"]["SWITCH_other"]["resnext64_4"][
            "switched_loss_improved_x_temp_ratio"],
        TinyImageNet_linf_SWITCH_neg_D121_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_neg"]["densenet121"][
            "switched_loss_improved_x_temp_ratio"],
        TinyImageNet_linf_SWITCH_neg_R32_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_neg"]["resnext32_4"][
            "switched_loss_improved_x_temp_ratio"],
        TinyImageNet_linf_SWITCH_neg_R64_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_neg"]["resnext64_4"][
            "switched_loss_improved_x_temp_ratio"],
        TinyImageNet_linf_SWITCH_other_D121_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_other"]["densenet121"][
            "switched_loss_improved_x_temp_ratio"],
        TinyImageNet_linf_SWITCH_other_R32_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_other"]["resnext32_4"][
            "switched_loss_improved_x_temp_ratio"],
        TinyImageNet_linf_SWITCH_other_R64_switched_loss_improved_x_temp_ratio=
        result["TinyImageNet"]["linf"]["SWITCH_other"]["resnext64_4"][
            "switched_loss_improved_x_temp_ratio"],
               )
          )

if __name__ == "__main__":
    datasets = ["CIFAR-10","CIFAR-100"]
    targeted = True
    if "CIFAR" in datasets[0]:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        archs = ["densenet121", "resnext32_4", "resnext64_4"]
    result = fetch_all_json_content(datasets, targeted, archs)  # result[dataset][norm][method][arch]
    if "TinyImageNet" == datasets[0]:
        draw_tables_TinyImageNet(result)
    else:
        draw_tables_CIFAR_only_SWITCH_neg(result)
    print("THIS IS {} {} result".format(datasets[0], targeted))
    # if "CIFAR" in dataset:
    #     draw_tables_for_CIFAR(norm, result_archs)
    # elif "TinyImageNet" in dataset:
    #     draw_tables_for_TinyImageNet(norm, result_archs)
    # print("THIS IS {} {} {}".format(dataset, norm, "untargeted" if not targeted else "targeted"))
