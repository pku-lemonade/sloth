import simpy
import json
import time
import logging
import argparse
from pydantic import ValidationError

class CFG:
    def __init__(self, args):
        self.simstart = args.simstart
        self.simend = args.simend
        self.flow = args.flow

parser = argparse.ArgumentParser()

parser.add_argument("--simstart", type=int, default=0,
                    help="Simulation start cycle, default is 0")
parser.add_argument("--simend", type=int, default=int((1<<31)-1),
                    help="Simulation end cycle, default is None (natural end)")
parser.add_argument("--flow", action="store_true", help="enable flow flag")
parser.add_argument("--workload", type=str, default="data/workload_example.json")
parser.add_argument("--arch", type=str, default="data/arch_example.json")
parser.add_argument("--fail", type=str, default="data/fail_example.json")
parser.add_argument("--fragment", type=str)
parser.add_argument("--type", type=str)
parser.add_argument("--location", type=str)
parser.add_argument("--plevel", type=str)
parser.add_argument("--structure", type=str)
parser.add_argument("--log", type=str, default="logging/simulation.log")
parser.add_argument("--level", type=str, default="debug")
parser.add_argument("--model", type=str, default="basic")
parser.add_argument("--times", type=int, default=1)

args = parser.parse_args()
cfg = CFG(args)

class MonitoredResource(simpy.Resource):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []
        
    def checkneed(self):
        if self._env.now >= cfg.simstart and self._env.now <= cfg.simend:
            return True
        else:
            return False

    def exe(self, task, delay, ins, v=None, core=None, index=None, opcode=None, flops=None, attributes=None):
        req = super().request()
        yield req
        if v is not None:
            v.run(core, index, ins.layer_id, opcode, flops=flops)
        if self.checkneed():
            if attributes is None:
                self.data.append((task, self._env.now, len(self.queue), "req", "B"))
            else:
                self.data.append((task, self._env.now, len(self.queue), "req", "B", attributes))
        if v is None:
            yield self._env.timeout(delay)
        else:
            yield self._env.timeout(delay, value=v)
        if self.checkneed():
            if attributes is None:
                self.data.append((task, self._env.now, len(self.queue), "req", "E"))
            else:
                self.data.append((task, self._env.now, len(self.queue), "req", "E", attributes))
                
        super().release(req)

    def execute(self, task, delay, ins, v=None, core=None, index=None, opcode=None, flops=None, attributes=None):
        return self._env.process(self.exe(task, delay, ins, v=v, core=core, index=index, opcode=opcode, flops=flops, attributes=attributes))