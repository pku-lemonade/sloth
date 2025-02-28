import simpy
import logging
from functools import partial, wraps
from src.core import Core
from src.noc_new import NoC, Link, Direction
from src.arch_config import CoreConfig, NoCConfig, ArchConfig, LinkConfig, MemConfig
from src.sim_type import Instruction, FailSlow, Data, Message
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
    def make_print():
        print("begin print:")

    def run(self):
        print("Start simulation.")
        self.env.run()
        #print(self.cores[0].lsu.data)
        #print(self.cores[1].tpu.data)
        #print(self.cores[2].lsu.data)
        print(self.noc.routers[0].core_out.linkentry.data)
        print("Simulation finished.")
        #将值传入json文件
        #self.make_print()
        return self.env