import re
import sys
import os
from typing import List

inst2int = {"READ": 0, "WRITE": 1, "SEND": 2, "RECV": 3, "STAY": 4, "COMP": 5}
op2int = {"CONV": 0, "POOL": 1, "FC": 2}
dt2int = {"PARA": 0, "FEAT": 1, "WGT": 2}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.sim_type import Instruction, PEworkload, Workload, TaskType

pe_pattern = r'PE (\d+)'
inst_pattern = r'(READ|WRITE|SEND|RECV|COMP|STAY) (\d+) (CONV|POOL|FC) (\d+) (PARA|FEAT|WGT) (-?\d+) (\d+)'

def build_trigger(program):
    last_layer = -1
    last_layer_ind = -1

    inputs = []
    outputs = []
    comp_pos = -1

    tri_program = program
    
    for ind, inst in enumerate(program):
        if inst.layer_id != last_layer or ind == len(program)-1:
            if ind == len(program)-1:
                inst_start = last_layer_ind
                inst_end = len(program)
            elif last_layer != -1:
                inst_start = last_layer_ind
                inst_end = ind
            else:
                inst_start = 0
                inst_end = ind

            inputs.clear()
            outputs.clear()
            comp_pos = -1

            for idx, layer_inst in enumerate(program[inst_start:inst_end], start=inst_start):
                match layer_inst.inst_type:
                    case TaskType.READ:
                        inputs.append(idx)
                    case TaskType.RECV:
                        inputs.append(idx)
                    case TaskType.STAY:
                        inputs.append(idx)
                    case TaskType.WRITE:
                        outputs.append(idx)
                    case TaskType.SEND:
                        outputs.append(idx)
                    case TaskType.COMP:
                        comp_pos = idx
                    
            for input in inputs:
                # print(f"insert {input}->{program[comp_pos].index}")
                tri_program[input].trigger_index.append(program[comp_pos].index)

            for output in outputs:
                # print(f"insert {comp_pos}->{program[output].index}")
                tri_program[comp_pos].trigger_index.append(program[output].index)
                
            last_layer = inst.layer_id
            last_layer_ind = ind
    
    return tri_program


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
                    pe_instructions_tri = build_trigger(pe_instructions)
                    pe_inst = PEworkload(id=pe_id, insts=pe_instructions)

                    all_pes.append(pe_inst)
                    pe_instructions.clear()
                pe_id = int(pe_match.group(1))
                continue
            
            inst_match = re.search(inst_pattern, line)
            if inst_match:
                inst_type, index, operation, layer_id, data_type, position, size = inst_match.groups()
                instruction = Instruction(
                    inst_type = inst2int[inst_type],
                    index = int(index),
                    operation = op2int[operation],
                    layer_id = int(layer_id),
                    data_type = dt2int[data_type],
                    position = int(position),
                    size = int(size)
                )
                pe_instructions.append(instruction)

        pe_instructions = sorted(pe_instructions, key=lambda x:x.layer_id)
        pe_instructions_tri = build_trigger(pe_instructions)
        pe_inst = PEworkload(id=pe_id, insts=pe_instructions_tri)

        all_pes.append(pe_inst)
        workload = Workload(name="resnet50", pes=all_pes)
        return workload


workload = inst_analyzer("tools/instructions.txt")
workload_json = workload.model_dump_json(indent=4)

with open("tools/workload.json", "w") as file:
    print(workload_json, file=file)