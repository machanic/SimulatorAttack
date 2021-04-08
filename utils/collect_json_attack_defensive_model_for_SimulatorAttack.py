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
                        "MetaAttack":"MetaAttack",
                       "simulate_bandits_shrink_attack":"MetaSimulator",
                         "NES": "NES"}

def from_method_to_dir_path(dataset, method, norm, targeted):
    # methods = ["bandits_attack", "MetaGradAttack", "NES-attack","P-RGF_biased","P-RGF_uniform","ZOO","simulate_bandits_shrink_attack"]
    if method == "bandits_attack":
        path = "{method}_on_defensive_model-{dataset}-cw_loss-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                        norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NES":
        path = "{method}-attack_on_defensive_model-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
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
    elif method == "MetaAttack":
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
    print(method)
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
        file_path = "/home1/machen/meta_perturbations_black_box_attack/logs/" + from_method_to_dir_path(dataset, method, norm, targeted)
        assert os.path.exists(file_path), "{} does not exist".format(file_path)
        folder_path_dict[paper_method_name] = file_path
    return folder_path_dict

def fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, defense_method, limited_queries=10000):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result = {}
    for method, folder in folder_list.items():
        if defense_method == "PCL":
            file_path = folder + "/pcl_{}_pcl_loss_result.json".format(arch)
        elif defense_method == "AdvTrain":
            file_path = folder + "/{}_{}_result.json".format(arch, "adv_train")
        else:
            file_path = folder + "/{}_{}_result.json".format(arch, defense_method)
        assert os.path.exists(file_path), "{} does not exist!".format(file_path)
        success_rate, avg_query_over_successful_samples, median_query_over_successful_samples, \
        correct_all, query_all, not_done_all, json_content = read_json_and_extract(file_path)
        not_done_all[query_all > limited_queries] = 1
        failure_rate = np.mean(not_done_all[correct_all.astype(np.bool)]).item()
        success_rate = new_round((1 - failure_rate) * 100, 1)
        if success_rate.is_integer():
            success_rate = int(success_rate)
        success_all = (1 - not_done_all.astype(np.int32)) * correct_all.astype(np.int32)
        success_query_all = success_all * query_all
        success_all = success_all.astype(np.bool)
        avg_query_over_successful_samples = int(new_round(success_query_all[success_all].mean().item(), 0))
        median_query_over_successful_samples = int(
            new_round(np.median(success_query_all[success_all]).item(), 0))
        avg_query_over_all_samples = int(new_round(np.mean(query_all[correct_all.astype(np.bool)]).item(), 0))
        median_query_over_all_samples = int(
            new_round(np.median(query_all[correct_all.astype(np.bool)]).item(), 0))
        result[method] = {"success_rate":success_rate, "avg_query_over_successful_samples":avg_query_over_successful_samples,
                          "median_query_over_successful_samples": median_query_over_successful_samples,
                        "avg_query_over_all_samples": avg_query_over_all_samples, "median_query_over_all_samples":median_query_over_all_samples}
    return result



def draw_tables_for_defense(archs_result):
    result = archs_result
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    print("""
             NES \cite{{ilyas2018blackbox}} & {ComDefend_NES_ASR}\% & {PCL_NES_ASR}\% & {FeatureDistillation_NES_ASR}\% &  {AdvTrain_NES_ASR}\% & {ComDefend_NES_AVGQ} & {PCL_NES_AVGQ} & {FeatureDistillation_NES_AVGQ} & {AdvTrain_NES_AVGQ}  & {ComDefend_NES_MEDQ} & {PCL_NES_MEDQ} & {FeatureDistillation_NES_MEDQ} & {AdvTrain_NES_MEDQ}  \\\\ 
            & RGF \cite{{2017RGF}} & {ComDefend_RGF_ASR}\% & {PCL_RGF_ASR}\% & {FeatureDistillation_RGF_ASR}\% &  {AdvTrain_RGF_ASR}\% &  {ComDefend_RGF_AVGQ} & {PCL_RGF_AVGQ} & {FeatureDistillation_RGF_AVGQ} & {AdvTrain_RGF_AVGQ}  & {ComDefend_RGF_MEDQ} & {PCL_RGF_MEDQ} & {FeatureDistillation_RGF_MEDQ} & {AdvTrain_RGF_MEDQ}  \\\\
            & P-RGF \cite{{cheng2019improving}} & {ComDefend_PRGF_ASR}\% & {PCL_PRGF_ASR}\% & {FeatureDistillation_PRGF_ASR}\% &  {AdvTrain_PRGF_ASR}\% &  {ComDefend_PRGF_AVGQ} & {PCL_PRGF_AVGQ} & {FeatureDistillation_PRGF_AVGQ} & {AdvTrain_PRGF_AVGQ}  & {ComDefend_PRGF_MEDQ} & {PCL_PRGF_MEDQ} & {FeatureDistillation_PRGF_MEDQ} & {AdvTrain_PRGF_MEDQ}  \\\\
            & Meta Attack \cite{{du2020queryefficient}} & {ComDefend_MetaAttack_ASR}\% & {PCL_MetaAttack_ASR}\% & {FeatureDistillation_MetaAttack_ASR}\% &  {AdvTrain_MetaAttack_ASR}\% &  {ComDefend_MetaAttack_AVGQ} & {PCL_MetaAttack_AVGQ} & {FeatureDistillation_MetaAttack_AVGQ} & {AdvTrain_MetaAttack_AVGQ}  & {ComDefend_MetaAttack_MEDQ} & {PCL_MetaAttack_MEDQ} & {FeatureDistillation_MetaAttack_MEDQ} & {AdvTrain_MetaAttack_MEDQ}  \\\\
            & Bandits \cite{{ilyas2018prior}} & {ComDefend_Bandits_ASR}\% & {PCL_Bandits_ASR}\% & {FeatureDistillation_Bandits_ASR}\% &  {AdvTrain_Bandits_ASR}\% &  {ComDefend_Bandits_AVGQ} & {PCL_Bandits_AVGQ} & {FeatureDistillation_Bandits_AVGQ} & {AdvTrain_Bandits_AVGQ}  & {ComDefend_Bandits_MEDQ} & {PCL_Bandits_MEDQ} & {FeatureDistillation_Bandits_MEDQ} & {AdvTrain_Bandits_MEDQ}  \\\\
            & Simulator Attack & {ComDefend_MetaSimulator_ASR}\% & {PCL_MetaSimulator_ASR}\% & {FeatureDistillation_MetaSimulator_ASR}\% &  {AdvTrain_MetaSimulator_ASR}\% &  {ComDefend_MetaSimulator_AVGQ} & {PCL_MetaSimulator_AVGQ} & {FeatureDistillation_MetaSimulator_AVGQ} & {AdvTrain_MetaSimulator_AVGQ}  & {ComDefend_MetaSimulator_MEDQ} & {PCL_MetaSimulator_MEDQ} & {FeatureDistillation_MetaSimulator_MEDQ} & {AdvTrain_MetaSimulator_MEDQ}  \\\\
                    """.format(
        ComDefend_NES_ASR=result["com_defend"]["NES"]["success_rate"],
        PCL_NES_ASR=result["PCL"]["NES"]["success_rate"],
        FeatureDistillation_NES_ASR=result["feature_distillation"]["NES"]["success_rate"],
        AdvTrain_NES_ASR=result["AdvTrain"]["NES"]["success_rate"],

        ComDefend_NES_AVGQ=result["com_defend"]["NES"][avg_q],
        PCL_NES_AVGQ=result["PCL"]["NES"][avg_q],
        FeatureDistillation_NES_AVGQ=result["feature_distillation"]["NES"][avg_q],
        AdvTrain_NES_AVGQ=result["AdvTrain"]["NES"][avg_q],

        ComDefend_NES_MEDQ=result["com_defend"]["NES"][med_q],
        PCL_NES_MEDQ=result["PCL"]["NES"][med_q],
        FeatureDistillation_NES_MEDQ=result["feature_distillation"]["NES"][med_q],
        AdvTrain_NES_MEDQ=result["AdvTrain"]["NES"][med_q],


        ComDefend_RGF_ASR=result["com_defend"]["RGF"]["success_rate"],
        PCL_RGF_ASR=result["PCL"]["RGF"]["success_rate"],
        FeatureDistillation_RGF_ASR=result["feature_distillation"]["RGF"]["success_rate"],
        AdvTrain_RGF_ASR=result["AdvTrain"]["RGF"]["success_rate"],

        ComDefend_RGF_AVGQ=result["com_defend"]["RGF"][avg_q],
        PCL_RGF_AVGQ=result["PCL"]["RGF"][avg_q],
        FeatureDistillation_RGF_AVGQ=result["feature_distillation"]["RGF"][avg_q],
        AdvTrain_RGF_AVGQ=result["AdvTrain"]["RGF"][avg_q],

        ComDefend_RGF_MEDQ=result["com_defend"]["RGF"][med_q],
        PCL_RGF_MEDQ=result["PCL"]["RGF"][med_q],
        FeatureDistillation_RGF_MEDQ=result["feature_distillation"]["RGF"][med_q],
        AdvTrain_RGF_MEDQ=result["AdvTrain"]["RGF"][med_q],

        ComDefend_PRGF_ASR=result["com_defend"]["PRGF"]["success_rate"],
        PCL_PRGF_ASR=result["PCL"]["PRGF"]["success_rate"],
        FeatureDistillation_PRGF_ASR=result["feature_distillation"]["PRGF"]["success_rate"],
        AdvTrain_PRGF_ASR=result["AdvTrain"]["PRGF"]["success_rate"],

        ComDefend_PRGF_AVGQ=result["com_defend"]["PRGF"][avg_q],
        PCL_PRGF_AVGQ=result["PCL"]["PRGF"][avg_q],
        FeatureDistillation_PRGF_AVGQ=result["feature_distillation"]["PRGF"][avg_q],
        AdvTrain_PRGF_AVGQ=result["AdvTrain"]["PRGF"][avg_q],

        ComDefend_PRGF_MEDQ=result["com_defend"]["PRGF"][med_q],
        PCL_PRGF_MEDQ=result["PCL"]["PRGF"][med_q],
        FeatureDistillation_PRGF_MEDQ=result["feature_distillation"]["PRGF"][med_q],
        AdvTrain_PRGF_MEDQ=result["AdvTrain"]["PRGF"][med_q],

        ComDefend_MetaAttack_ASR=result["com_defend"]["MetaAttack"]["success_rate"],
        PCL_MetaAttack_ASR=result["PCL"]["MetaAttack"]["success_rate"],
        FeatureDistillation_MetaAttack_ASR=result["feature_distillation"]["MetaAttack"]["success_rate"],
        AdvTrain_MetaAttack_ASR=result["AdvTrain"]["MetaAttack"]["success_rate"],

        ComDefend_MetaAttack_AVGQ=result["com_defend"]["MetaAttack"][avg_q],
        PCL_MetaAttack_AVGQ=result["PCL"]["MetaAttack"][avg_q],
        FeatureDistillation_MetaAttack_AVGQ=result["feature_distillation"]["MetaAttack"][avg_q],
        AdvTrain_MetaAttack_AVGQ=result["AdvTrain"]["MetaAttack"][avg_q],

        ComDefend_MetaAttack_MEDQ=result["com_defend"]["MetaAttack"][med_q],
        PCL_MetaAttack_MEDQ=result["PCL"]["MetaAttack"][med_q],
        FeatureDistillation_MetaAttack_MEDQ=result["feature_distillation"]["MetaAttack"][med_q],
        AdvTrain_MetaAttack_MEDQ=result["AdvTrain"]["MetaAttack"][med_q],

        ComDefend_Bandits_ASR=result["com_defend"]["Bandits"]["success_rate"],
        PCL_Bandits_ASR=result["PCL"]["Bandits"]["success_rate"],
        FeatureDistillation_Bandits_ASR=result["feature_distillation"]["Bandits"]["success_rate"],
        AdvTrain_Bandits_ASR=result["AdvTrain"]["Bandits"]["success_rate"],

        ComDefend_Bandits_AVGQ=result["com_defend"]["Bandits"][avg_q],
        PCL_Bandits_AVGQ=result["PCL"]["Bandits"][avg_q],
        FeatureDistillation_Bandits_AVGQ=result["feature_distillation"]["Bandits"][avg_q],
        AdvTrain_Bandits_AVGQ=result["AdvTrain"]["Bandits"][avg_q],

        ComDefend_Bandits_MEDQ=result["com_defend"]["Bandits"][med_q],
        PCL_Bandits_MEDQ=result["PCL"]["Bandits"][med_q],
        FeatureDistillation_Bandits_MEDQ=result["feature_distillation"]["Bandits"][med_q],
        AdvTrain_Bandits_MEDQ=result["AdvTrain"]["Bandits"][med_q],

        ComDefend_MetaSimulator_ASR=result["com_defend"]["MetaSimulator"]["success_rate"],
        PCL_MetaSimulator_ASR=result["PCL"]["MetaSimulator"]["success_rate"],
        FeatureDistillation_MetaSimulator_ASR=result["feature_distillation"]["MetaSimulator"]["success_rate"],
        AdvTrain_MetaSimulator_ASR=result["AdvTrain"]["MetaSimulator"]["success_rate"],

        ComDefend_MetaSimulator_AVGQ=result["com_defend"]["MetaSimulator"][avg_q],
        PCL_MetaSimulator_AVGQ=result["PCL"]["MetaSimulator"][avg_q],
        FeatureDistillation_MetaSimulator_AVGQ=result["feature_distillation"]["MetaSimulator"][avg_q],
        AdvTrain_MetaSimulator_AVGQ=result["AdvTrain"]["MetaSimulator"][avg_q],

        ComDefend_MetaSimulator_MEDQ=result["com_defend"]["MetaSimulator"][med_q],
        PCL_MetaSimulator_MEDQ=result["PCL"]["MetaSimulator"][med_q],
        FeatureDistillation_MetaSimulator_MEDQ=result["feature_distillation"]["MetaSimulator"][med_q],
        AdvTrain_MetaSimulator_MEDQ=result["AdvTrain"]["MetaSimulator"][med_q],

    )
    )


if __name__ == "__main__":
    datasets = ["CIFAR-10", "CIFAR-100", "TinyImageNet"]
    norm = "linf"
    dataset = "TinyImageNet"
    targeted = False
    if "CIFAR" in dataset:
        arch = "resnet-50"
    else:
        arch = "resnet50"
    result_archs={}
    for defense_method in ["com_defend","PCL","feature_distillation","AdvTrain"]:
        result = fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, defense_method)
        result_archs[defense_method] = result
    draw_tables_for_defense(result_archs)
    print("THIS IS {} {} {}".format(dataset, norm, "untargeted" if not targeted else "targeted"))
