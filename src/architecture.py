import simpy
from src.core import Core
from src.noc import NoC, Link
from src.config import CoreConfig, NoCConfig, ArchConfig, LinkConfig, MemConfig
from src.sim_type import Instruction
from typing import List

class Arch:
    def __init__(self, arch: ArchConfig, program: List[List[Instruction]]):
        print("Constructing hardware architecture.")
        self.env = simpy.Environment()
        self.noc = self.build_noc(arch.noc)
        self.cores = self.build_cores(arch.core, program)
        print("Construction finished.")

    def build_cores(self, config: CoreConfig, program: List[List[Instruction]]) -> List[Core]:
        cores = []
        for id in range(config.x * config.y):
            core = Core(self.env, config, program[id], id)
            link = Link(self.env, LinkConfig(width=128, delay=1))
            core.connect(link, self.noc.routers[id])
            self.noc.routers[id].bound_with_core(link, core)
            cores.append(core)
        return cores

    def build_noc(self, config: NoCConfig) -> NoC:
        print("Building NoC architecture.")
        NoC_temp = NoC(self.env, config)
        NoC_temp.build_connection()
        print("Finished.")
        return NoC_temp
    
    # def build_mem(self, config: MemConfig) -> Mem:
    #     return Mem(self.env, config)

    def run(self):
        print("Start simulation.")
        self.env.run()
        print("Simulation finished.")