import json
import time
import logging
import argparse
from src.arch_config import ArchConfig
from src.sim_type import Workload, FailSlow
from src.architecture import Arch
from pydantic import ValidationError
from src.common import *

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

    stage = None
    noc_model = args.model
    inference_time = args.times

    arch = Arch(arch_config, [pe.insts for pe in workload.pes], fail_slow, workload.name, args.fail, noc_model, inference_time, stage)

    start_time = time.time()
    result = arch.run()
    end_time = time.time()
    simulation_time = end_time - start_time

    if stage == "pre_analysis":
        stage = "post_analysis"
        arch = Arch(arch_config, [pe.insts for pe in workload.pes], fail_slow, stage)
        result = arch.run()
        end_time = time.time()
        simulation_time = end_time - start_time

    # arch.debug()

    print("="*40)
    print(f"Simulation time is {simulation_time}.")
    print(f"Total simulation cycles is {arch.end_time:.0f}.")


if __name__ == '__main__':
    main()
