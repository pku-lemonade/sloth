import json
import time
import logging
import argparse
from src.arch_config import ArchConfig
from src.sim_type import Workload, FailSlow
from src.architecture import Arch
from pydantic import ValidationError

batch_size = 16
def fail_analyzer(filename: str) -> FailSlow:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = FailSlow.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())

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

    parser.add_argument("--workload", type=str, default="tests/resnet50/workload.json")
    parser.add_argument("--arch", type=str, default="arch/mesh4_4.json")
    parser.add_argument("--fail", type=str, default="failslow/base.json")
    parser.add_argument("--log", type=str, default="logging/simulation.log")
    parser.add_argument("--level", type=str, default="info")

    args = parser.parse_args()

    print("Reading architecture config.")
    arch_config = config_analyzer(args.arch)
    print("Finished.")

    print("Loading fail-slow setting.")
    fail_slow = fail_analyzer(args.fail)
    print(f"Finished loading fail-slow setting.")

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

    arch = Arch(arch_config, [pe.insts for pe in workload.pes], fail_slow)

    start_time = time.time()
    result = arch.run()
    end_time = time.time()
    simulation_time = end_time - start_time


    print("="*40)
    print(f"Simulation time is {simulation_time}.")
    print(f"Total simulation cycles is {result.now * batch_size}.")


if __name__ == '__main__':
    main()
