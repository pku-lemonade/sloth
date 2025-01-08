import re
import json
import argparse
from typing import List
from config import ArchConfig
from sim_type import Instruction
from architecture import Arch
from pydantic import ValidationError

pe_pattern = r'PE (\d+)'
inst_pattern = r'(READ|WRITE|SEND|RECV|COMP) (\d+) (CONV|POOL|FC) (\d+) (PARA|FEAT|WGT) (-?\d+) (\d+)'

def inst_analyzer(filename: str) -> List[List[Instruction]]:
    all_instructions = []

    with open(filename, "r", encoding="utf-8") as file:
        pe_id = -1
        pe_instructions = []

        for line in file:
            pe_match = re.search(pe_pattern, line)
            if pe_match:
                if pe_id != -1:
                    all_instructions.append(pe_instructions)
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

        all_instructions.append(pe_instructions)
    return all_instructions

def config_analyzer(filename: str) -> ArchConfig:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            config = ArchConfig.model_validate(data)
            return config
        except ValidationError as e:
            print(e.json())

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inst", type=str, default="configs/instructions.txt")
    parser.add_argument("--arch", type=str, default="configs/arch.json")

    args = parser.parse_args()

    print("Reading architecture config.")
    arch_config = config_analyzer(args.arch)
    print("Finished.")

    print("Loading simulation workload.")
    workload = inst_analyzer(args.inst)
    print("Finished.")

    arch = Arch(arch_config, workload)

    arch.run()

if __name__ == '__main__':
    main()
