import os
import sys
import json
import math
import numpy as np
from typing import List
from trace_format import CompTrace, InstTrace
from pydantic import ValidationError
from sklearn.neighbors import NearestNeighbors

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from tools.geninst_new import json_analyzer, config_analyzer

def comp_analyzer(filename: str) -> CompTrace:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = CompTrace.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())

# 计算每一层切分后算子在各个PE的平均执行时间
# 返回average_exetime，按layer_mapping的pe顺序排列
def calc_pe_exetime(trace: List[InstTrace], layer_mapping: List[int]):
    tot_exetime = {}
    inst_num = {}
    average_exetime = []

    for id in layer_mapping:
        tot_exetime[id] = 0
        inst_num[id] = 0

    for id, inst_trace in enumerate(trace):
        tot_exetime[inst_trace.pe_id] += inst_trace.end_time - inst_trace.start_time
        inst_num[inst_trace.pe_id] += 1

    for id in layer_mapping:
        average_exetime.append(tot_exetime[id] / inst_num[id])

    # print(f"average exetime: {average_exetime}")
    return average_exetime

# trace为同一层的所有指令
def layer_failslow_detect(trace: List[InstTrace], percent: float, layer_mapping: List[int]):
    average_exetime = calc_pe_exetime(trace, layer_mapping)
    average_exetime = np.array(average_exetime).reshape(-1, 1)

    k = int(math.sqrt(len(average_exetime)+1))
    if k <= 1:
        print("Could not detect failslow pe under current mapping!")
        return
    
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(average_exetime)
    distances, indices = nbrs.kneighbors(average_exetime)

    threshold = average_exetime.mean() * percent
    k_distances = distances[:, k]
    outliers = k_distances > threshold
    outlier_indices = np.where(outliers)[0]

    print(f"failslow pes are: {[layer_mapping[i] for i in outlier_indices]}")

percent = 1
arch_configs = config_analyzer("arch/gemini4_4.json")

def get_id(x: int, y: int):
    return x * arch_configs.core.y + y

if __name__ == '__main__':
    net = json_analyzer("tests/darknet19/mapping.json")
    
    comp_trace = comp_analyzer("data/darknet19/tpu/comp_trace.json")

    layer_mapping = [[] for _ in range(len(net.layers))]
    # 统计每层的映射情况
    for id, layer in enumerate(net.layers):
        for output in layer.output_feature:
            for block in output.blocks:
                for pe in block.cores:
                    layer_mapping[id].append(get_id(pe.x, pe.y))
        
        layer_mapping[id] = sorted(layer_mapping[id])
    
    comp_trace_layer = [[] for _ in range(len(net.layers))]

    for inst_trace in comp_trace.trace:
        comp_trace_layer[inst_trace.layer_id].append(inst_trace)

    # print(len(comp_trace_layer[0]))

    for layer_trace in comp_trace_layer:
        print(f"layer {layer_trace[0].layer_id} was computed on pe: {layer_mapping[layer_trace[0].layer_id]}")
        layer_failslow_detect(layer_trace, percent, layer_mapping[layer_trace[0].layer_id])