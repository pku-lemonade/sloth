import simpy
from src.core import Core
from src.noc_new import NoC, Link, Direction
from src.arch_config import CoreConfig, NoCConfig, ArchConfig, LinkConfig, MemConfig
from src.sim_type import Instruction, FailSlow
from typing import List

class Arch:
    def __init__(self, arch: ArchConfig, program: List[List[Instruction]], fail: FailSlow):
        print("Constructing hardware architecture.")
        self.env = simpy.Environment()
        self.noc = self.build_noc(arch.noc)
        self.cores = self.build_cores(arch.core, program)
        self.fail_slow = fail
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

    def run_fail_slow(self, link):
        yield self.env.timeout(link.start_time)
        match link.direction:
            case Direction.NORTH:
                self.noc.routers[link.router_id].north_link.change_width(link.times)
            case Direction.SOUTH:
                self.noc.routers[link.router_id].south_link.change_width(link.times)
            case Direction.EAST:
                self.noc.routers[link.router_id].east_link.change_width(link.times)
            case Direction.WEST:
                self.noc.routers[link.router_id].west_link.change_width(link.times)

        yield self.env.timeout(link.end_time - link.start_time)
        match link.direction:
            case Direction.NORTH:
                self.noc.routers[link.router_id].north_link.change_width(link.times)
            case Direction.SOUTH:
                self.noc.routers[link.router_id].south_link.change_width(link.times)
            case Direction.EAST:
                self.noc.routers[link.router_id].east_link.change_width(link.times)
            case Direction.WEST:
                self.noc.routers[link.router_id].west_link.change_width(link.times)

    def run(self):
        print("Start simulation.")
        self.env.run()
        
        for link in self.fail_slow.link:
            self.env.process(self.run_fail_slow(link))

        print("Simulation finished.")
        return self.env