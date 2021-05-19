import numpy as np
import json
import os
import re
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
                        "SWITCH_rnd_save":'SWITCH_other',
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
        path = "{method}_{dataset}_surrogate_arch_{surrogate_arch}_{norm}_{target_str}".format(method=method, dataset=dataset,
                                                                                               surrogate_arch=surrogate_arch,
                                                                                               norm=norm, target_str="untargeted" if not targeted else "targeted_increment")
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
        cosine = new_round(json_content["avg_cosine_similarity"],3)
        return success_rate, avg_query_over_successful_samples, median_query_over_successful_samples, \
               correct_all, query_all, not_done_all, cosine

def get_file_name_list(target_model):
    folder_path_dict = {}
    root_dir = "/home1/machen/query_based_black_box_attack/logs/SWITCH_cosine_gradients/"
    pattern = re.compile("SWITCH_grad_cosine_stats-(.*?)-CIFAR-10-cw-loss-l2-(.*)")
    pattern_random =re.compile("SWITCH_random_grad_grad_cosine_stats-CIFAR-10-cw-loss-l2-(.*)")
    for folder in os.listdir(root_dir):
        ma = re.match(pattern,folder)
        if ma is None:
            ma = re.match(pattern_random, folder)
            target_str = ma.group(1)
            arch = "random_grad"
        else:
            arch = ma.group(1)
            target_str = ma.group(2)
        file_path = root_dir+"/" +folder + "/{}_result.json".format(target_model)
        targeted = True if target_str == "targeted_increment" else False
        folder_path_dict[(arch, targeted)] = file_path
    return folder_path_dict

def fetch_all_json_content_given_contraint(targeted, target_model):
    folder_list = get_file_name_list(target_model)
    result = {}
    for (surrogate_model, targeted_extract), file_path in folder_list.items():
        if targeted != targeted_extract:
            continue
        if not os.path.exists(file_path):
            print("{} does not exist!".format(file_path))
            result[surrogate_model] = {"success_rate": 0,
                                       "avg_query_over_successful_samples": 0,
                                       "median_query_over_successful_samples": 0,
                                       "avg_query_over_all_samples": 0,
                                       "median_query_over_all_samples": 0,
                                       "cosine_grad": 0}
        else:
            # assert os.path.exists(file_path), "{} does not exist!".format(file_path)
            success_rate, avg_query_over_successful_samples, median_query_over_successful_samples, \
            correct_all, query_all, not_done_all, cosine_grad = read_json_and_extract(file_path)
            not_done_all[query_all>10000] = 1
            query_all[query_all > 10000] = 10000
            query_all[not_done_all==1] = 10000
            avg_query_over_all_samples = int(new_round(np.mean(query_all[correct_all.astype(np.bool)]).item(),0))
            median_query_over_all_samples = int(new_round(np.median(query_all[correct_all.astype(np.bool)]).item(),0))
            result[surrogate_model] = {"success_rate":success_rate, "avg_query_over_successful_samples":avg_query_over_successful_samples,
                              "median_query_over_successful_samples": median_query_over_successful_samples,
                            "avg_query_over_all_samples": avg_query_over_all_samples, "median_query_over_all_samples":median_query_over_all_samples,
                            "cosine_grad":cosine_grad}
    return result



def draw_tables_for_CIFAR( archs_result):
    result = archs_result
    avg_q = "avg_query_over_successful_samples"
    med_q = "median_query_over_successful_samples"
    # archs_result = {"pyramidnet272" : {"NES": {}, "P-RGF": {}, }, "gdas": { 各种方法}}
    print("""
        &  Success Rate  & {random_grad_ASR}\% & {vgg19_bn_ASR}\% & {resnet_50_ASR}\% &  {resnet_110_ASR}\% & {preresnet_110_ASR}\% & {resnext_8x64d_ASR}\% & {resnext_16x64d_ASR}\% & {densenet_bc_100_12_ASR}\% & {densenet_bc_L190_k40_ASR}\%  \\\\
        &  Avg. Query  & {random_grad_AVGQ} & {vgg19_bn_AVGQ} & {resnet_50_AVGQ} &  {resnet_110_AVGQ} & {preresnet_110_AVGQ} & {resnext_8x64d_AVGQ} & {resnext_16x64d_AVGQ} & {densenet_bc_100_12_AVGQ} & {densenet_bc_L190_k40_AVGQ}  \\\\
        &  cosine grad  & {random_grad_cosine_grad} & {vgg19_bn_cosine_grad} & {resnet_50_cosine_grad} & {resnet_110_cosine_grad} & {preresnet_110_cosine_grad} & {resnext_8x64d_cosine_grad} & {resnext_16x64d_cosine_grad} & {densenet_bc_100_12_cosine_grad} & {densenet_bc_L190_k40_cosine_grad} \\\\
    """.format(
        random_grad_ASR=result["random_grad"]["success_rate"],
        vgg19_bn_ASR=result["vgg19_bn"]["success_rate"],
        resnet_50_ASR=result["resnet-50"]["success_rate"],
        resnet_110_ASR = result["resnet-110"]["success_rate"],
        preresnet_110_ASR = result["preresnet-110"]["success_rate"],
        resnext_8x64d_ASR = result["resnext-8x64d"]["success_rate"],
        resnext_16x64d_ASR=result["resnext-16x64d"]["success_rate"],
        densenet_bc_100_12_ASR = result["densenet-bc-100-12"]["success_rate"],
        densenet_bc_L190_k40_ASR=result["densenet-bc-L190-k40"]["success_rate"],

        random_grad_AVGQ=result["random_grad"][avg_q],
        vgg19_bn_AVGQ=result["vgg19_bn"][avg_q],
        resnet_50_AVGQ=result["resnet-50"][avg_q],
        resnet_110_AVGQ=result["resnet-110"][avg_q],
        preresnet_110_AVGQ=result["preresnet-110"][avg_q],
        resnext_8x64d_AVGQ=result["resnext-8x64d"][avg_q],
        resnext_16x64d_AVGQ=result["resnext-16x64d"][avg_q],
        densenet_bc_100_12_AVGQ=result["densenet-bc-100-12"][avg_q],
        densenet_bc_L190_k40_AVGQ=result["densenet-bc-L190-k40"][avg_q],

        random_grad_cosine_grad=result["random_grad"]["cosine_grad"],
        vgg19_bn_cosine_grad=result["vgg19_bn"]["cosine_grad"],
        resnet_50_cosine_grad=result["resnet-50"]["cosine_grad"],
        resnet_110_cosine_grad=result["resnet-110"]["cosine_grad"],
        preresnet_110_cosine_grad=result["preresnet-110"]["cosine_grad"],
        resnext_8x64d_cosine_grad=result["resnext-8x64d"]["cosine_grad"],
        resnext_16x64d_cosine_grad=result["resnext-16x64d"]["cosine_grad"],
        densenet_bc_100_12_cosine_grad=result["densenet-bc-100-12"]["cosine_grad"],
        densenet_bc_L190_k40_cosine_grad=result["densenet-bc-L190-k40"]["cosine_grad"],
               )
          )

if __name__ == "__main__":
    dataset = "CIFAR-10"
    norm = "l2"
    targeted = False
    result = fetch_all_json_content_given_contraint(targeted, 'WRN-28-10-drop')
    draw_tables_for_CIFAR(result)
    print("THIS IS {} {} {}".format(dataset, norm, "untargeted" if not targeted else "targeted"))
