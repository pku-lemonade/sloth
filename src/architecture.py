import simpy
import os
import logging
from functools import partial, wraps
from src.core import Core
from src.noc_new import NoC, Link, Direction
from src.arch_config import CoreConfig, NoCConfig, ArchConfig, LinkConfig, MemConfig
from src.sim_type import Instruction, FailSlow, Data, Message
from src.common import cfg
from typing import List
logger = logging.getLogger("Arch")

def trace(env, callback):
    """Replace the ``step()`` method of *env* with a tracing function
    that calls *callbacks* with an events time, priority, ID and its
    instance just before it is processed.

    """
    def get_wrapper(env_step, callback):
        """Generate the wrapper for env.step()."""
        @wraps(env_step)
        def tracing_step():
            """Call *callback* for the next event if one exist before
            calling ``env.step()``."""
            if len(env._queue):
                t, prio, eid, event = env._queue[0]
                callback(t, prio, eid, event)
            return env_step()
        return tracing_step

    env.step = get_wrapper(env.step, callback)

data = []

def monitor(data, t, prio, eid, event):
    if isinstance(event, simpy.resources.store.StoreGet) or isinstance(event, simpy.resources.store.StorePut):
        logger.info((t, eid, event.proc._generator, type(event), event.resource, event.value))
    else:
        logger.info((t, eid, type(event), event.value))
    # data.append((t, eid, type(event)))

monitor = partial(monitor, data)

def patch_resource(resource, pre=None, post=None):
    """Patch *resource* so that it calls the callable *pre* before each
    put/get/request/release operation and the callable *post* after each
    operation.  The only argument to these functions is the resource
    instance.

    """
    def get_wrapper(func):
        # Generate a wrapper for put/get/request/release
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is the actual wrapper
            # Call "pre" callback
            if pre:
                pre(resource, func)

            # Perform actual operation
            ret = func(*args, **kwargs)

            # Call "post" callback
            if post:
                post(resource, ret)

            return ret
        return wrapper

    # Replace the original operations with our wrapper
    for name in ['put', 'get', 'request', 'release']:
        if hasattr(resource, name):
            setattr(resource, name, get_wrapper(getattr(resource, name)))

def monitor1(data, resource, func):
    """This is our monitoring callback."""
    item = (
        resource._env.now,  # The current simulation time
        "pre",
        len(resource.items),  # The number of queued processes
        resource.items,
        # resource,
        func,
    )
    logger.info(item)
    data.append(item)

def monitor2(data, resource, ret):
    """This is our monitoring callback."""
    item = (
        resource._env.now,  # The current simulation time
        "post",
        len(resource.items),  # The number of queued processes
        resource.items,
        # resource,
        ret,
    )
    logger.info(item)
    data.append(item)

monitor1 = partial(monitor1, data)
monitor2 = partial(monitor2, data)

class Arch:
    def __init__(self, arch: ArchConfig, program: List[List[Instruction]], fail: FailSlow):
        print("Constructing hardware architecture.")
        self.env = simpy.Environment()
        
        
        self.noc = self.build_noc(arch.noc)
        #print(len(self.noc.r2r_links))
        self.cores = self.build_cores(arch.core, program)
        
        trace(self.env, monitor)
        patch_resource(self.cores[9].data_in.store, pre=monitor1, post=monitor2)

        self.fail_slow = fail
        print("Construction finished.")

    def debug(self):
        for d in data:
            logger.info(d)

    def build_cores(self, config: CoreConfig, program: List[List[Instruction]]) -> List[Core]:
        cores = []
        for id in range(config.x * config.y):
            core = Core(self.env, config, program[id], id)

            link1 = Link(self.env, LinkConfig(width=128, delay=1))
            link2 = Link(self.env, LinkConfig(width=128, delay=1))
            self.noc.routers[id].bound_with_core(link1, link2)
            core.bound_with_router(link2, link1)
            cores.append(core)

        return cores

    def build_noc(self, config: NoCConfig) -> NoC:
        print("Building NoC architecture.")
        return NoC(self.env, config).build_connection()


    #输出可视化文件
    def make_print_lsu():
        #对于每个lsu
        count=0
        #req->count+=1 release->count-=1
    def processesmonitor(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            for idx, line in enumerate(data):
                task,ts, lenthqueue, ation, ph = line
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\": \"{task}\",\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"lenthqueue\":{lenthqueue}}}}}")

    #这个由学长来编号,对于每个编号(id)怎么处理的逻辑我已经写好了:
    def processesmonitorlink(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            for idx, line in enumerate(data):
                task,ts, lenthqueue, ation, ph,dest = line
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\": \"{task}\",\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"lenthqueue\":{lenthqueue},\"dest\":{dest}}}}}")

    def processesspm(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            for idx, line in enumerate(data):
                task, capacity, action, size ,ts , ph= line
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\":\"{task}\" ,\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"act\":\"{action}\",\"capacity\":{capacity},\"size\":{size}}}}}")

    def processesflow(self,data,file,id,source):
         if len(data)==0:
            return
         with open(file,"w") as f: 
            for idx, line in enumerate(data):
                index, _, action, ts= line
                task=action+str(index)
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\":\"{task}\" ,\"ph\":\"B\",\"ts\":{ts},\"id\":{index},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"act\":\"{action}\"}}}}")
                f.write(",\n")
                f.write(f"{{\"name\":\"{task}\" ,\"ph\":\"E\",\"ts\":{ts},\"id\":{index},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"act\":\"{action}\"}}}}")
                if action!="recv":
                    f.write(",\n")
                    f.write(f"{{\"name\":\"connect\" ,\"ph\":\"s\",\"ts\":{ts},\"id\":{index},\"pid\":{id},\"tid\":\"{source}\"}}")
                else:
                    f.write(",\n")
                    f.write(f"{{\"name\":\"connect\",\"ph\":\"f\",\"bp\":\"e\",\"id\":{index},\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\"}}")


                
                            
    def make_print(self):
        #print(self.cores[0].lsu.data)
        #print(self.cores[1].tpu.data)
        #print(self.cores[2].lsu.data)
        #print(self.noc.routers[0].core_out.linkentry.data)
        #print(self.cores[1].spm_manager.data)
        os.makedirs("gen", exist_ok=True)
        for i in range(len(self.cores)):
            self.processesmonitor(self.cores[i].lsu.data,"gen/lsu"+str(i)+".json",i,"lsu")
            self.processesmonitor(self.cores[i].tpu.data,"gen/tpu"+str(i)+".json",i,"tpu")
            self.processesspm(self.cores[i].spm_manager.data,"gen/spm"+str(i)+".json",i,"spm")

        for i in range(len(self.noc.r2r_links)):
            self.processesmonitorlink(self.noc.r2r_links[i].linkentry.data,"gen/link"+str(i)+".json",i,"link")

        
        if cfg.flow:
            for i in range(len(self.cores)):
                self.processesflow(self.cores[i].flow_in,"gen/flow_in"+str(i)+".json",i,"flow_in")
            for i in range(len(self.cores)):
                self.processesflow(self.cores[i].flow_out,"gen/flow_out"+str(i)+".json",i,"flow_out")


    def run(self):
        print("Start simulation.")
        self.env.run()
        for id in range(16):
            print(f"PE{id} processed [{self.cores[id].scheduler.inst_counter}/{len(self.cores[id].program)}] instructions.")
        print("Simulation finished.")
        #将值传入json文件
        # self.make_print()
        return self.env