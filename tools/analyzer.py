import re
import sys
import os
from typing import List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.sim_type import Instruction, PEworkload, Workload

pe_pattern = r'PE (\d+)'
inst_pattern = r'(READ|WRITE|SEND|RECV|COMP) (\d+) (CONV|POOL|FC) (\d+) (PARA|FEAT|WGT) (-?\d+) (\d+)'

def inst_analyzer(filename: str) -> List[List[Instruction]]:
    all_pes = []

    with open(filename, "r", encoding="utf-8") as file:
        pe_id = -1
        pe_instructions = []

        for line in file:
            pe_match = re.search(pe_pattern, line)
            if pe_match:
                if pe_id != -1:
                    pe_instructions = sorted(pe_instructions, key=lambda x:x.layer_id)
                    pe_inst = PEworkload(id=pe_id, insts=pe_instructions)
                    all_pes.append(pe_inst)
                    pe_instructions.clear()
                pe_id = int(pe_match.group(1))
                continue
            
            inst_match = re.search(inst_pattern, line)
            if inst_match:
                inst_type, index, operation, layer_id, data_type, position, size = inst_match.groups()
                instruction = Instruction(
                    inst_type = inst_type,
                    index = int(index),
                    operation = operation,
                    layer_id = int(layer_id),
                    data_type = data_type,
                    position = int(position),
                    size = int(size)
                )
                pe_instructions.append(instruction)

        pe_instructions = sorted(pe_instructions, key=lambda x:x.layer_id)
        pe_inst = PEworkload(id=pe_id, insts=pe_instructions)
        all_pes.append(pe_inst)
        workload = Workload(name="resnet50", pes=all_pes)
        return workload


workload = inst_analyzer("many-core-sim/tools/instructions.txt")
workload_json = workload.model_dump_json(indent=4)

with open("many-core-sim/tools/workload.json", "w") as file:
    print(workload_json, file=file)