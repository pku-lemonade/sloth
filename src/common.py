import simpy
#monitor resource such as lsu and tpu
import json
import time
import logging
import argparse
from pydantic import ValidationError

batch_size = 64

class CFG:
    def __init__(self,args):
        self.simstart = args.simstart
        self.simend = args.simend

parser = argparse.ArgumentParser()

#set simulation start and end cycle
parser.add_argument("--simstart", type=int, default=0,
                    help="Simulation start cycle, default is 0")
parser.add_argument("--simend", type=int, default=int((1<<31)-1),
                    help="Simulation end cycle, default is None (natural end)")
parser.add_argument("--workload", type=str, default="tests/resnet50/workload.json")
parser.add_argument("--arch", type=str, default="arch/mesh4_4.json")
parser.add_argument("--fail", type=str, default="failslow/base.json")
parser.add_argument("--log", type=str, default="logging/simulation.log")
parser.add_argument("--level", type=str, default="info")

args = parser.parse_args()
cfg=CFG(args)



class MonitoredResource(simpy.Resource):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []
        
    def checkneed(self):
        if self._env.now >= cfg.simstart and self._env.now <= cfg.simend:
            return True
        else:
            return False

    def exe(self,delay,v=None):
        req=super().request()
        yield req
        self.data.append((self._env.now, len(self.queue),"req"))
        if v is None:
            yield self._env.timeout(delay)
        else:
            yield self._env.timeout(delay, value=v)
        self.data.append((self._env.now, len(self.queue),"release"))
        super().release(req)

    def execute(self,delay,v=None):
        return self._env.process(self.exe(delay,v))
        

