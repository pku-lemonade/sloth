import os
import sys
import copy
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def fail_probing(
    instructions: list,
    mode: str = "both",
    target_inst_types: list = None,
    target_layer_ids: list = None,
) -> list:
    assert mode in ("before", "after", "both", "none"), f"Unsupported mode: {mode}"

    if mode == "none":
        return instructions

    new_instructions = []
    inserted_probes = []

    for inst in instructions:
        inst_type = inst.get("inst_type", -1)
        layer_id = inst.get("layer_id", -1)
        index = inst.get("index", -1)

        # 判断是否应插入 probe
        if (target_inst_types is not None and inst_type not in target_inst_types):
            new_instructions.append(inst)
            continue
        if (target_layer_ids is not None and layer_id not in target_layer_ids):
            new_instructions.append(inst)
            continue

        # 插入前置 probe
        if mode in ("before", "both"):
            probe_before = {
                "instruction_id": index,
                "instruction_type": inst_type,
                "layer_id": layer_id,
                "pe_id": -1,
                "start_time": -1,
                "inference_time": -1,
                "data_size": -1,
                "src_id": -1,
                "dst_id": -1
            }
            inserted_probes.append(copy.deepcopy(probe_before))
            new_instructions.append(probe_before)

        # 插入原始指令
        new_instructions.append(copy.deepcopy(inst))

        # 插入后置 probe
        if mode in ("after", "both"):
            probe_after = {
                "instruction_id": index,
                "instruction_type": inst_type,
                "layer_id": layer_id,
                "pe_id": -1,
                "end_time": -1,
                "inference_time": -1,
            }
            inserted_probes.append(copy.deepcopy(probe_after))
            new_instructions.append(probe_after)

    return new_instructions
