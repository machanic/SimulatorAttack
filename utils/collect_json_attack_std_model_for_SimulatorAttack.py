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
    if method == "bandits_attack":
        path = "{method}-{dataset}-cw_loss-{norm}-{target_str}".format(method=method, dataset=dataset,
                                                                        norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
    elif method == "NES":
        path = "{method}-attack-{dataset}-{norm}-{target_str}".format(method=method, dataset=dataset,
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
    elif method == "MetaAttack":
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

def fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, limited_queries):
    folder_list = get_file_name_list(dataset, method_name_to_paper, norm, targeted)
    result = {}
    for method, folder in folder_list.items():
        if method == "MetaSimulator" and targeted and dataset!="TinyImageNet":
            for m in ["3", "5"]:
                file_path = folder + "/{}_meta_interval_{}_result.json".format(arch, m)
                assert os.path.exists(file_path), "{} does not exist!".format(file_path)
                print("Read {}".format(file_path))
                _, _, _, correct_all, query_all, not_done_all, json_content = read_json_and_extract(file_path)
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
                result[method + "_m{}".format(m)] = {"success_rate": success_rate,
                                  "avg_query_over_successful_samples": avg_query_over_successful_samples,
                                  "median_query_over_successful_samples": median_query_over_successful_samples,
                                  "avg_query_over_all_samples": avg_query_over_all_samples,
                                  "median_query_over_all_samples": median_query_over_all_samples}
        else:
            file_path = folder + "/{}_result.json".format(arch)
            assert os.path.exists(file_path), "{} does not exist!".format(file_path)
            print("Read {}".format(file_path))
            _, _, _, correct_all, query_all, not_done_all, json_content = read_json_and_extract(file_path)
            not_done_all[query_all>limited_queries] = 1
            # query_all[query_all > limited_queries] = limited_queries
            # query_all[not_done_all==1] = limited_queries
            failure_rate = np.mean(not_done_all[correct_all.astype(np.bool)]).item()
            success_rate = new_round((1 - failure_rate) * 100, 1)
            if success_rate.is_integer():
                success_rate = int(success_rate)
            success_all = (1-not_done_all.astype(np.int32)) * correct_all.astype(np.int32)
            success_query_all = success_all * query_all
            success_all = success_all.astype(np.bool)
            avg_query_over_successful_samples = int(new_round(success_query_all[success_all].mean().item(),0))
            median_query_over_successful_samples = int(new_round(np.median(success_query_all[success_all]).item(),0))
            avg_query_over_all_samples = int(new_round(np.mean(query_all[correct_all.astype(np.bool)]).item(),0))
            median_query_over_all_samples = int(new_round(np.median(query_all[correct_all.astype(np.bool)]).item(),0))
            result[method] = {"success_rate":success_rate, "avg_query_over_successful_samples":avg_query_over_successful_samples,
                              "median_query_over_successful_samples": median_query_over_successful_samples,
                            "avg_query_over_all_samples": avg_query_over_all_samples, "median_query_over_all_samples":median_query_over_all_samples}
    return result

def draw_tables_for_TinyImageNet(archs_result):
    result = archs_result
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    print("""
& NES \cite{{ilyas2018blackbox}} & {D121_NES_ASR}\% & {R32_NES_ASR}\% & {R64_NES_ASR}\% & {D121_NES_AVGQ} & {R32_NES_AVGQ} & {R64_NES_AVGQ} & {D121_NES_MEDQ} & {R32_NES_MEDQ} & {R64_NES_MEDQ} \\\\
& & RGF \cite{{2017RGF}} & {D121_RGF_ASR}\% & {R32_RGF_ASR}\% & {R64_RGF_ASR}\% & {D121_RGF_AVGQ} & {R32_RGF_AVGQ} & {R64_RGF_AVGQ} & {D121_RGF_MEDQ} & {R32_RGF_MEDQ} & {R64_RGF_MEDQ} \\\\
& & P-RGF \cite{{cheng2019improving}} & {D121_PRGF_ASR}\% & {R32_PRGF_ASR}\% & {R64_PRGF_ASR}\% & {D121_PRGF_AVGQ} & {R32_PRGF_AVGQ} & {R64_PRGF_AVGQ} & {D121_PRGF_MEDQ} & {R32_PRGF_MEDQ} & {R64_PRGF_MEDQ} \\\\
& & Meta Attack \cite{{du2020queryefficient}} & {D121_MetaAttack_ASR}\% & {R32_MetaAttack_ASR}\% & {R64_MetaAttack_ASR}\% & {D121_MetaAttack_AVGQ} & {R32_MetaAttack_AVGQ} & {R64_MetaAttack_AVGQ} & {D121_MetaAttack_MEDQ} & {R32_MetaAttack_MEDQ} & {R64_MetaAttack_MEDQ} \\\\
& & Bandits  \cite{{ilyas2018prior}} & {D121_Bandits_ASR}\% & {R32_Bandits_ASR}\% & {R64_Bandits_ASR}\% & {D121_Bandits_AVGQ} & {R32_Bandits_AVGQ} & {R64_Bandits_AVGQ} & {D121_Bandits_MEDQ} & {R32_Bandits_MEDQ} & {R64_Bandits_MEDQ} \\\\
& & Simulator Attack & {D121_MetaSimulator_ASR}\% & {R32_MetaSimulator_ASR}\% & {R64_MetaSimulator_ASR}\% & {D121_MetaSimulator_AVGQ} & {R32_MetaSimulator_AVGQ} & {R64_MetaSimulator_AVGQ} & {D121_MetaSimulator_MEDQ} & {R32_MetaSimulator_MEDQ} & {R64_MetaSimulator_MEDQ} \\\\
        """.format(
            D121_NES_ASR=result["densenet121"]["NES"]["success_rate"],R32_NES_ASR=result["resnext32_4"]["NES"]["success_rate"],R64_NES_ASR=result["resnext64_4"]["NES"]["success_rate"],
            D121_NES_AVGQ=result["densenet121"]["NES"][avg_q], R32_NES_AVGQ=result["resnext32_4"]["NES"][avg_q], R64_NES_AVGQ=result["resnext64_4"]["NES"][avg_q],
            D121_NES_MEDQ=result["densenet121"]["NES"][med_q], R32_NES_MEDQ=result["resnext32_4"]["NES"][med_q], R64_NES_MEDQ=result["resnext64_4"]["NES"][med_q],

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

            D121_MetaAttack_ASR=result["densenet121"]["MetaAttack"]["success_rate"],R32_MetaAttack_ASR=result["resnext32_4"]["MetaAttack"]["success_rate"],R64_MetaAttack_ASR=result["resnext64_4"]["MetaAttack"]["success_rate"],
            D121_MetaAttack_AVGQ=result["densenet121"]["MetaAttack"][avg_q], R32_MetaAttack_AVGQ=result["resnext32_4"]["MetaAttack"][avg_q], R64_MetaAttack_AVGQ=result["resnext64_4"]["MetaAttack"][avg_q],
            D121_MetaAttack_MEDQ=result["densenet121"]["MetaAttack"][med_q], R32_MetaAttack_MEDQ=result["resnext32_4"]["MetaAttack"][med_q], R64_MetaAttack_MEDQ=result["resnext64_4"]["MetaAttack"][med_q],

            D121_Bandits_ASR=result["densenet121"]["Bandits"]["success_rate"],
            R32_Bandits_ASR=result["resnext32_4"]["Bandits"]["success_rate"],
            R64_Bandits_ASR=result["resnext64_4"]["Bandits"]["success_rate"],
            D121_Bandits_AVGQ=result["densenet121"]["Bandits"][avg_q],
            R32_Bandits_AVGQ=result["resnext32_4"]["Bandits"][avg_q],
            R64_Bandits_AVGQ=result["resnext64_4"]["Bandits"][avg_q],
            D121_Bandits_MEDQ=result["densenet121"]["Bandits"][med_q],
            R32_Bandits_MEDQ=result["resnext32_4"]["Bandits"][med_q],
            R64_Bandits_MEDQ=result["resnext64_4"]["Bandits"][med_q],

            D121_MetaSimulator_ASR=result["densenet121"]["MetaSimulator"]["success_rate"],R32_MetaSimulator_ASR=result["resnext32_4"]["MetaSimulator"]["success_rate"],R64_MetaSimulator_ASR=result["resnext64_4"]["MetaSimulator"]["success_rate"],
            D121_MetaSimulator_AVGQ=result["densenet121"]["MetaSimulator"][avg_q], R32_MetaSimulator_AVGQ=result["resnext32_4"]["MetaSimulator"][avg_q], R64_MetaSimulator_AVGQ=result["resnext64_4"]["MetaSimulator"][avg_q],
            D121_MetaSimulator_MEDQ=result["densenet121"]["MetaSimulator"][med_q], R32_MetaSimulator_MEDQ=result["resnext32_4"]["MetaSimulator"][med_q], R64_MetaSimulator_MEDQ=result["resnext64_4"]["MetaSimulator"][med_q],
        )
              )

def draw_tables_for_CIFAR(targeted, archs_result):
    result = archs_result
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    if not targeted:
        print("""
& & NES \cite{{ilyas2018blackbox}} & {pyramidnet272_NES_ASR}\% & {gdas_NES_ASR}\% & {WRN28_NES_ASR}\% & {WRN40_NES_ASR}\% & {pyramidnet272_NES_AVGQ} & {gdas_NES_AVGQ} & {WRN28_NES_AVGQ} & {WRN40_NES_AVGQ} & {pyramidnet272_NES_MEDQ} & {gdas_NES_MEDQ} & {WRN28_NES_MEDQ} & {WRN40_NES_MEDQ} \\\\
& & RGF \cite{{2017RGF}} & {pyramidnet272_RGF_ASR}\% & {gdas_RGF_ASR}\% & {WRN28_RGF_ASR}\% & {WRN40_RGF_ASR}\% & {pyramidnet272_RGF_AVGQ} & {gdas_RGF_AVGQ} & {WRN28_RGF_AVGQ} & {WRN40_RGF_AVGQ} & {pyramidnet272_RGF_MEDQ} & {gdas_RGF_MEDQ} & {WRN28_RGF_MEDQ} & {WRN40_RGF_MEDQ} \\\\
& & P-RGF \cite{{cheng2019improving}} & {pyramidnet272_PRGF_ASR}\% & {gdas_PRGF_ASR}\% & {WRN28_PRGF_ASR}\% & {WRN40_PRGF_ASR}\% & {pyramidnet272_PRGF_AVGQ} & {gdas_PRGF_AVGQ} & {WRN28_PRGF_AVGQ} & {WRN40_PRGF_AVGQ} & {pyramidnet272_PRGF_MEDQ} & {gdas_PRGF_MEDQ} & {WRN28_PRGF_MEDQ} & {WRN40_PRGF_MEDQ} \\\\
& & Meta Attack \cite{{du2020queryefficient}} & {pyramidnet272_MetaAttack_ASR}\% & {gdas_MetaAttack_ASR}\% & {WRN28_MetaAttack_ASR}\% & {WRN40_MetaAttack_ASR}\% & {pyramidnet272_MetaAttack_AVGQ} & {gdas_MetaAttack_AVGQ} & {WRN28_MetaAttack_AVGQ} & {WRN40_MetaAttack_AVGQ} & {pyramidnet272_MetaAttack_MEDQ} & {gdas_MetaAttack_MEDQ} & {WRN28_MetaAttack_MEDQ} & {WRN40_MetaAttack_MEDQ} \\\\
& & Bandits \cite{{ilyas2018prior}} & {pyramidnet272_Bandits_ASR}\% & {gdas_Bandits_ASR}\% & {WRN28_Bandits_ASR}\% & {WRN40_Bandits_ASR}\% & {pyramidnet272_Bandits_AVGQ} & {gdas_Bandits_AVGQ} & {WRN28_Bandits_AVGQ} & {WRN40_Bandits_AVGQ} & {pyramidnet272_Bandits_MEDQ} & {gdas_Bandits_MEDQ} & {WRN28_Bandits_MEDQ} & {WRN40_Bandits_MEDQ} \\\\
& & Simulator Attack & {pyramidnet272_MetaSimulator_ASR}\% & {gdas_MetaSimulator_ASR}\% & {WRN28_MetaSimulator_ASR}\% & {WRN40_MetaSimulator_ASR}\% & {pyramidnet272_MetaSimulator_AVGQ} & {gdas_MetaSimulator_AVGQ} & {WRN28_MetaSimulator_AVGQ} & {WRN40_MetaSimulator_AVGQ} & {pyramidnet272_MetaSimulator_MEDQ} & {gdas_MetaSimulator_MEDQ} & {WRN28_MetaSimulator_MEDQ} & {WRN40_MetaSimulator_MEDQ} \\\\
        """.format(
                   pyramidnet272_NES_ASR=result["pyramidnet272"]["NES"]["success_rate"], gdas_NES_ASR=result["gdas"]["NES"]["success_rate"],
                   WRN28_NES_ASR=result["WRN-28-10-drop"]["NES"]["success_rate"], WRN40_NES_ASR=result["WRN-40-10-drop"]["NES"]["success_rate"],
                   pyramidnet272_NES_AVGQ=result["pyramidnet272"]["NES"][avg_q],
                   gdas_NES_AVGQ=result["gdas"]["NES"][avg_q], WRN28_NES_AVGQ=result["WRN-28-10-drop"]["NES"][avg_q],  WRN40_NES_AVGQ=result["WRN-40-10-drop"]["NES"][avg_q],
                   pyramidnet272_NES_MEDQ=result["pyramidnet272"]["NES"][med_q],
                   gdas_NES_MEDQ=result["gdas"]["NES"][med_q], WRN28_NES_MEDQ=result["WRN-28-10-drop"]["NES"][med_q],
                   WRN40_NES_MEDQ=result["WRN-40-10-drop"]["NES"][med_q],

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

                    pyramidnet272_MetaAttack_ASR=result["pyramidnet272"]["MetaAttack"]["success_rate"],
                    gdas_MetaAttack_ASR=result["gdas"]["MetaAttack"]["success_rate"],
                    WRN28_MetaAttack_ASR=result["WRN-28-10-drop"]["MetaAttack"]["success_rate"],
                    WRN40_MetaAttack_ASR=result["WRN-40-10-drop"]["MetaAttack"]["success_rate"],
                    pyramidnet272_MetaAttack_AVGQ=result["pyramidnet272"]["MetaAttack"][avg_q],
                    gdas_MetaAttack_AVGQ=result["gdas"]["MetaAttack"][avg_q],
                    WRN28_MetaAttack_AVGQ=result["WRN-28-10-drop"]["MetaAttack"][avg_q],
                    WRN40_MetaAttack_AVGQ=result["WRN-40-10-drop"]["MetaAttack"][avg_q],
                    pyramidnet272_MetaAttack_MEDQ=result["pyramidnet272"]["MetaAttack"][med_q],
                    gdas_MetaAttack_MEDQ=result["gdas"]["MetaAttack"][med_q],
                    WRN28_MetaAttack_MEDQ=result["WRN-28-10-drop"]["MetaAttack"][med_q],
                    WRN40_MetaAttack_MEDQ=result["WRN-40-10-drop"]["MetaAttack"][med_q],

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

            pyramidnet272_MetaSimulator_ASR=result["pyramidnet272"]["MetaSimulator"]["success_rate"],
            gdas_MetaSimulator_ASR=result["gdas"]["MetaSimulator"]["success_rate"],
            WRN28_MetaSimulator_ASR=result["WRN-28-10-drop"]["MetaSimulator"]["success_rate"],
            WRN40_MetaSimulator_ASR=result["WRN-40-10-drop"]["MetaSimulator"]["success_rate"],
            pyramidnet272_MetaSimulator_AVGQ=result["pyramidnet272"]["MetaSimulator"][avg_q],
            gdas_MetaSimulator_AVGQ=result["gdas"]["MetaSimulator"][avg_q],
            WRN28_MetaSimulator_AVGQ=result["WRN-28-10-drop"]["MetaSimulator"][avg_q],
            WRN40_MetaSimulator_AVGQ=result["WRN-40-10-drop"]["MetaSimulator"][avg_q],
            pyramidnet272_MetaSimulator_MEDQ=result["pyramidnet272"]["MetaSimulator"][med_q],
            gdas_MetaSimulator_MEDQ=result["gdas"]["MetaSimulator"][med_q],
            WRN28_MetaSimulator_MEDQ=result["WRN-28-10-drop"]["MetaSimulator"][med_q],
            WRN40_MetaSimulator_MEDQ=result["WRN-40-10-drop"]["MetaSimulator"][med_q],

                   )
              )
    else:
        print("""
& & NES \cite{{ilyas2018blackbox}} & {pyramidnet272_NES_ASR}\% & {gdas_NES_ASR}\% & {WRN28_NES_ASR}\% & {WRN40_NES_ASR}\% & {pyramidnet272_NES_AVGQ} & {gdas_NES_AVGQ} & {WRN28_NES_AVGQ} & {WRN40_NES_AVGQ} & {pyramidnet272_NES_MEDQ} & {gdas_NES_MEDQ} & {WRN28_NES_MEDQ} & {WRN40_NES_MEDQ} \\\\
& & Meta Attack \cite{{du2020queryefficient}} & {pyramidnet272_MetaAttack_ASR}\% & {gdas_MetaAttack_ASR}\% & {WRN28_MetaAttack_ASR}\% & {WRN40_MetaAttack_ASR}\% & {pyramidnet272_MetaAttack_AVGQ} & {gdas_MetaAttack_AVGQ} & {WRN28_MetaAttack_AVGQ} & {WRN40_MetaAttack_AVGQ} & {pyramidnet272_MetaAttack_MEDQ} & {gdas_MetaAttack_MEDQ} & {WRN28_MetaAttack_MEDQ} & {WRN40_MetaAttack_MEDQ} \\\\
& & Bandits \cite{{ilyas2018prior}} & {pyramidnet272_Bandits_ASR}\% & {gdas_Bandits_ASR}\% & {WRN28_Bandits_ASR}\% & {WRN40_Bandits_ASR}\% & {pyramidnet272_Bandits_AVGQ} & {gdas_Bandits_AVGQ} & {WRN28_Bandits_AVGQ} & {WRN40_Bandits_AVGQ} & {pyramidnet272_Bandits_MEDQ} & {gdas_Bandits_MEDQ} & {WRN28_Bandits_MEDQ} & {WRN40_Bandits_MEDQ} \\\\
& & Simulator Attack (m=3) & {pyramidnet272_MetaSimulator_m3_ASR}\% & {gdas_MetaSimulator_m3_ASR}\% & {WRN28_MetaSimulator_m3_ASR}\% & {WRN40_MetaSimulator_m3_ASR}\% & {pyramidnet272_MetaSimulator_m3_AVGQ} & {gdas_MetaSimulator_m3_AVGQ} & {WRN28_MetaSimulator_m3_AVGQ} & {WRN40_MetaSimulator_m3_AVGQ} & {pyramidnet272_MetaSimulator_m3_MEDQ} & {gdas_MetaSimulator_m3_MEDQ} & {WRN28_MetaSimulator_m3_MEDQ} & {WRN40_MetaSimulator_m3_MEDQ} \\\\
& & Simulator Attack (m=5) & {pyramidnet272_MetaSimulator_m5_ASR}\% & {gdas_MetaSimulator_m5_ASR}\% & {WRN28_MetaSimulator_m5_ASR}\% & {WRN40_MetaSimulator_m5_ASR}\% & {pyramidnet272_MetaSimulator_m5_AVGQ} & {gdas_MetaSimulator_m5_AVGQ} & {WRN28_MetaSimulator_m5_AVGQ} & {WRN40_MetaSimulator_m5_AVGQ} & {pyramidnet272_MetaSimulator_m5_MEDQ} & {gdas_MetaSimulator_m5_MEDQ} & {WRN28_MetaSimulator_m5_MEDQ} & {WRN40_MetaSimulator_m5_MEDQ} \\\\
                """.format(
            pyramidnet272_NES_ASR=result["pyramidnet272"]["NES"]["success_rate"],
            gdas_NES_ASR=result["gdas"]["NES"]["success_rate"],
            WRN28_NES_ASR=result["WRN-28-10-drop"]["NES"]["success_rate"],
            WRN40_NES_ASR=result["WRN-40-10-drop"]["NES"]["success_rate"],
            pyramidnet272_NES_AVGQ=result["pyramidnet272"]["NES"][avg_q],
            gdas_NES_AVGQ=result["gdas"]["NES"][avg_q], WRN28_NES_AVGQ=result["WRN-28-10-drop"]["NES"][avg_q],
            WRN40_NES_AVGQ=result["WRN-40-10-drop"]["NES"][avg_q],
            pyramidnet272_NES_MEDQ=result["pyramidnet272"]["NES"][med_q],
            gdas_NES_MEDQ=result["gdas"]["NES"][med_q], WRN28_NES_MEDQ=result["WRN-28-10-drop"]["NES"][med_q],
            WRN40_NES_MEDQ=result["WRN-40-10-drop"]["NES"][med_q],

            pyramidnet272_MetaAttack_ASR=result["pyramidnet272"]["MetaAttack"]["success_rate"],
            gdas_MetaAttack_ASR=result["gdas"]["MetaAttack"]["success_rate"],
            WRN28_MetaAttack_ASR=result["WRN-28-10-drop"]["MetaAttack"]["success_rate"],
            WRN40_MetaAttack_ASR=result["WRN-40-10-drop"]["MetaAttack"]["success_rate"],
            pyramidnet272_MetaAttack_AVGQ=result["pyramidnet272"]["MetaAttack"][avg_q],
            gdas_MetaAttack_AVGQ=result["gdas"]["MetaAttack"][avg_q],
            WRN28_MetaAttack_AVGQ=result["WRN-28-10-drop"]["MetaAttack"][avg_q],
            WRN40_MetaAttack_AVGQ=result["WRN-40-10-drop"]["MetaAttack"][avg_q],
            pyramidnet272_MetaAttack_MEDQ=result["pyramidnet272"]["MetaAttack"][med_q],
            gdas_MetaAttack_MEDQ=result["gdas"]["MetaAttack"][med_q],
            WRN28_MetaAttack_MEDQ=result["WRN-28-10-drop"]["MetaAttack"][med_q],
            WRN40_MetaAttack_MEDQ=result["WRN-40-10-drop"]["MetaAttack"][med_q],

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

            pyramidnet272_MetaSimulator_m3_ASR=result["pyramidnet272"]["MetaSimulator_m3"]["success_rate"],
            gdas_MetaSimulator_m3_ASR=result["gdas"]["MetaSimulator_m3"]["success_rate"],
            WRN28_MetaSimulator_m3_ASR=result["WRN-28-10-drop"]["MetaSimulator_m3"]["success_rate"],
            WRN40_MetaSimulator_m3_ASR=result["WRN-40-10-drop"]["MetaSimulator_m3"]["success_rate"],
            pyramidnet272_MetaSimulator_m3_AVGQ=result["pyramidnet272"]["MetaSimulator_m3"][avg_q],
            gdas_MetaSimulator_m3_AVGQ=result["gdas"]["MetaSimulator_m3"][avg_q],
            WRN28_MetaSimulator_m3_AVGQ=result["WRN-28-10-drop"]["MetaSimulator_m3"][avg_q],
            WRN40_MetaSimulator_m3_AVGQ=result["WRN-40-10-drop"]["MetaSimulator_m3"][avg_q],
            pyramidnet272_MetaSimulator_m3_MEDQ=result["pyramidnet272"]["MetaSimulator_m3"][med_q],
            gdas_MetaSimulator_m3_MEDQ=result["gdas"]["MetaSimulator_m3"][med_q],
            WRN28_MetaSimulator_m3_MEDQ=result["WRN-28-10-drop"]["MetaSimulator_m3"][med_q],
            WRN40_MetaSimulator_m3_MEDQ=result["WRN-40-10-drop"]["MetaSimulator_m3"][med_q],

            pyramidnet272_MetaSimulator_m5_ASR=result["pyramidnet272"]["MetaSimulator_m5"]["success_rate"],
            gdas_MetaSimulator_m5_ASR=result["gdas"]["MetaSimulator_m5"]["success_rate"],
            WRN28_MetaSimulator_m5_ASR=result["WRN-28-10-drop"]["MetaSimulator_m5"]["success_rate"],
            WRN40_MetaSimulator_m5_ASR=result["WRN-40-10-drop"]["MetaSimulator_m5"]["success_rate"],
            pyramidnet272_MetaSimulator_m5_AVGQ=result["pyramidnet272"]["MetaSimulator_m5"][avg_q],
            gdas_MetaSimulator_m5_AVGQ=result["gdas"]["MetaSimulator_m5"][avg_q],
            WRN28_MetaSimulator_m5_AVGQ=result["WRN-28-10-drop"]["MetaSimulator_m5"][avg_q],
            WRN40_MetaSimulator_m5_AVGQ=result["WRN-40-10-drop"]["MetaSimulator_m5"][avg_q],
            pyramidnet272_MetaSimulator_m5_MEDQ=result["pyramidnet272"]["MetaSimulator_m5"][med_q],
            gdas_MetaSimulator_m5_MEDQ=result["gdas"]["MetaSimulator_m5"][med_q],
            WRN28_MetaSimulator_m5_MEDQ=result["WRN-28-10-drop"]["MetaSimulator_m5"][med_q],
            WRN40_MetaSimulator_m5_MEDQ=result["WRN-40-10-drop"]["MetaSimulator_m5"][med_q],
        )
        )



if __name__ == "__main__":
    dataset = "TinyImageNet"
    norm = "l2"
    targeted = True
    limited_queries_big = 10000
    if "CIFAR" in dataset:
        archs = ['pyramidnet272',"gdas","WRN-28-10-drop", "WRN-40-10-drop"]
    else:
        archs = ["densenet121", "resnext32_4", "resnext64_4"]
    result_archs_big_queries = {}
    for arch in archs:
        result = fetch_all_json_content_given_contraint(dataset, norm, targeted, arch, limited_queries_big)
        result_archs_big_queries[arch] = result
    if "CIFAR" in dataset:
        draw_tables_for_CIFAR(targeted, result_archs_big_queries)
    elif "TinyImageNet" in dataset:
        draw_tables_for_TinyImageNet(result_archs_big_queries)
    print("THIS IS {} {} {}".format(dataset, norm, "untargeted" if not targeted else "targeted"))
