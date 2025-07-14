import os
import sys
import copy
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.sim_type import Task, Probe

compute_code = ["Conv", "Pool", "FC", "Elem", "GConv", "PTP", "Trans"]
communication_code = ["Send", "Recv"]

# 输入原始 Task 序列，根据用户配置插入 收集代码片段
def fail_probing(
    pe_id: int,
    # 原始 Task 序列
    tasks: list[Task],
    mode: str = "both",
    # 插入指令类型
    target_inst_types: list[str] = None,
    # 插入层 id
    target_layer_ids: list[int] = None,
) -> list:
    assert mode in ("before", "after", "both", "none"), f"Unsupported mode: {mode}"

    if mode == "none":
        return tasks

    for inst in tasks:
        inst_type = inst.opcode
        layer_id = inst.layer_id
        index = inst.index

        # 判断是否应插入 probe
        if (target_inst_types is not None and inst_type not in target_inst_types):
            continue
        if (target_layer_ids is not None and layer_id not in target_layer_ids):
            continue

        # 插入前置 probe
        if mode in ("before", "both"):
            probe_metric = {}
            if inst_type in compute_code:
                probe_metric = {
                    "start_time": -1,
                    "flops": -1
                }
            elif inst_type in communication_code:
                probe_metric = {
                    "start_time": -1,
                    "data_size": -1,
                    "src_id": -1
                }
            inst.probe_st = Probe(
                flag = 0,
                metric = probe_metric
            )

        # 插入后置 probe
        if mode in ("after", "both"):
            probe_metric = {}
            if inst_type in compute_code:
                probe_metric = {
                    "end_time": -1,
                }
            elif inst_type in communication_code:
                probe_metric = {
                    "end_time": -1,
                    "dst_id": -1
                }
            inst.probe_ed = Probe(
                flag = 1,
                metric = probe_metric
            )

    return tasks
