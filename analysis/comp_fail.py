import json
import numpy as np
from typing import List
from trace_format import CompTrace, InstTrace
from pydantic import ValidationError
from sklearn.neighbors import NearestNeighbors

def comp_analyzer(filename: str) -> CompTrace:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = CompTrace.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())

# 计算每一层切分后算子在各个PE的平均执行时间
def calc_pe_exetime(trace: List[InstTrace]):
    cur_inst_num = 0
    tot_exetime = 0
    average_exetime = []

    cur_pe_id = -1

    index2pe = {}

    for id, inst_trace in enumerate(trace):
        if cur_pe_id != -1 and inst_trace.pe_id != cur_pe_id:
            layer_average_exetime = tot_exetime / cur_inst_num
            index2pe[len(average_exetime)] = cur_pe_id
            average_exetime.append(layer_average_exetime)

            cur_pe_id = inst_trace.pe_id
            tot_exetime = inst_trace.end_time - inst_trace.ready_time
            cur_inst_num = 1
        else:
            cur_pe_id = inst_trace.pe_id
            tot_exetime += inst_trace.end_time - inst_trace.ready_time
            cur_inst_num += 1

    # 处理最后一层
    layer_average_exetime = tot_exetime / cur_inst_num
    index2pe[len(average_exetime)] = cur_pe_id
    average_exetime.append(layer_average_exetime)

    return average_exetime, index2pe

# 输入为同一层的所有指令
def layer_failslow_detect(trace: List[InstTrace], k: int, threshold: int):
    average_exetime, index2pe = calc_pe_exetime(trace)
    average_exetime = np.array(average_exetime).reshape(-1, 1)

    # print(len(average_exetime))
    
    # if len(average_exetime) < k + 1:
    #     return 
    
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(average_exetime)
    distances, indices = nbrs.kneighbors(average_exetime)

    k_distances = distances[:, k]
    outliers = k_distances > threshold
    outlier_indices = np.where(outliers)[0]

    pe_ids = tuple(index2pe.get(item, None) for item in outlier_indices)
    print(f"failslow pes are: {pe_ids}")

k = 1
threshold = 1000

if __name__ == '__main__':
    comp_trace = comp_analyzer("data/darknet19/tpu/comp_trace.json")
    
    comp_trace_layer = [[] for _ in range(25)]

    for inst_trace in comp_trace.trace:
        comp_trace_layer[inst_trace.layer_id].append(inst_trace)

    # print(len(comp_trace_layer[0]))

    for layer_trace in comp_trace_layer:
        print(f"layer {layer_trace[0].layer_id}")
        layer_failslow_detect(layer_trace, k, threshold)