import torch
import numpy as np
from collections import OrderedDict


def success_rate_and_query_coorelation(query_all, not_done_indexes, correct_indexes):
    # query为0的是还没攻击就分类错的（当然最好没有这种情况), not_done_indexes是未攻击成功的，还有很多事没有攻击成功的，query很大达到100000
    query_threshold_success_rate = OrderedDict()  # accumulative query->success rate
    query_success_rate = OrderedDict()
    if isinstance(query_all, torch.Tensor):
        query_all = query_all.detach().cpu().numpy().astype(np.int32)
    if isinstance(not_done_indexes, torch.Tensor):
        not_done_indexes = not_done_indexes.detach().cpu().numpy().astype(np.int32)
    query_all = query_all[np.nonzero(correct_indexes)[0]]
    not_done_indexes = not_done_indexes[np.nonzero(correct_indexes)[0]]
    assert len(query_all) == len(not_done_indexes)
    query_all = query_all[np.nonzero(query_all)[0]]  # 选择出非0的, 排除原本模型就能分类错位的图片index
    not_done_indexes = not_done_indexes[np.nonzero(query_all)[0]]
    total_samples = len(query_all)
    # 未攻击成功的图片，不能算作统计count，因此将这些query改成一个特殊的值-1
    if len(np.nonzero(not_done_indexes)[0]) > 0:
        query_all[np.nonzero(not_done_indexes)[0]] = -1
    unique_query, query_count =np.unique(query_all, return_counts=True)
    accumulate_count = 0
    for query, count in zip(unique_query, query_count):
        if query.item() == -1:
            continue
        accumulate_count += count
        query_threshold_success_rate[query.item()] = float(accumulate_count) / total_samples  #  success rate 就是攻击成功的个数占总体的个数
        query_success_rate[query.item()] = float(count) / total_samples
    return query_threshold_success_rate, query_success_rate

def success_rate_avg_query(query_all, not_done_indexes, correct_indexes, success_rate_threhold):
    # query为0的是还没攻击就分类错的（当然最好没有这种情况), not_done_indexes是未攻击成功的，还有很多事没有攻击成功的，query很大达到100000
    if isinstance(query_all, torch.Tensor):
        query_all = query_all.detach().cpu().numpy().astype(np.int32)
    if isinstance(not_done_indexes, torch.Tensor):
        not_done_indexes = not_done_indexes.detach().cpu().numpy().astype(np.int32)
    assert len(query_all) == len(not_done_indexes)
    query_all = query_all[np.nonzero(correct_indexes)[0]]
    not_done_indexes = not_done_indexes[np.nonzero(correct_indexes)[0]]
    query_all = query_all[np.where(query_all < 10000)[0]]  # 选择出非0的, 排除原本模型就能分类错位的图片index
    not_done_indexes = not_done_indexes[np.where(query_all < 10000)[0]]
    query_all = query_all.astype(np.float32)
    success_indexes = np.nonzero(not_done_indexes == 0)[0]
    query_all = query_all[success_indexes]
    success_rate_list = list(range(1,101))
    sucess_rate_avg_query_dict = OrderedDict()
    for success_rate in success_rate_list:
        if success_rate > success_rate_threhold:
            break
        threshold = np.percentile(query_all, float(success_rate))
        avg_query = np.mean(query_all[np.where(query_all <= threshold)[0]])
        sucess_rate_avg_query_dict[success_rate] = avg_query.item()
    assert len(sucess_rate_avg_query_dict) > 0, "success_rate_threhold is {}".format(success_rate_threhold)
    return sucess_rate_avg_query_dict


def query_to_bins(query_all):
    return np.histogram(query_all, bins=20, range=(0,10000), density=False)