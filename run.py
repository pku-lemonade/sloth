import json
import logging
import argparse
from src.config import ArchConfig
from src.sim_type import Workload
from src.architecture import Arch
from pydantic import ValidationError

def config_analyzer(filename: str) -> ArchConfig:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            config = ArchConfig.model_validate(data)
            return config
        except ValidationError as e:
            print(e.json())

def workload_analyzer(filename: str) -> Workload:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            workload = Workload.model_validate(data)
            return workload
        except ValidationError as e:
            print(e.json())

def setup_logging(filename, level):
    logging.basicConfig(
        level = level,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        filename = filename,
        filemode = 'w'
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--workload", type=str, default="programs/resnet50/resnet50.json")
    parser.add_argument("--arch", type=str, default="configs/arch.json")
    parser.add_argument("--log", type=str, default="logging/simulation.log")
    parser.add_argument("--level", type=str, default="info")

    args = parser.parse_args()

    print("Reading architecture config.")
    arch_config = config_analyzer(args.arch)
    print("Finished.")

    print("Loading simulation workload.")
    workload = workload_analyzer(args.workload)
    print(f"Finished loading {workload.name} as simulation workload.")

    print("Setting up logging.")
    
    if args.level == "debug":
        level = logging.DEBUG
    elif args.level == "info":
        level = logging.INFO
    setup_logging(args.log, level)

    print("Finished.")

    arch = Arch(arch_config, [pe.insts for pe in workload.pes])

    arch.run()

if __name__ == '__main__':
    main()
