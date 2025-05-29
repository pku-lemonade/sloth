import simpy
import os
import sys
import logging
from functools import partial, wraps
from src.core import Core
from src.noc_new import NoC, Link, Direction
from src.arch_config import CoreConfig, NoCConfig, ArchConfig, LinkConfig, MemConfig
from src.sim_type import *
from src.common import cfg,Timer, init_graph, ind2ins
from src.draw import draw_grid
from analysis.trace_format import *
from typing import List
logger = logging.getLogger("Arch")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

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
    if (isinstance(event, simpy.resources.store.StoreGet) or isinstance(event, simpy.resources.store.StorePut)) and False:
        logger.info((t, eid, event.proc._generator, type(event), event.resource, event.value))
    # else:
        # logger.info((t, eid, type(event), event.value))
    data.append((t, eid, type(event)))

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
    # logger.info(item)
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
    # logger.info(item)
    data.append(item)

monitor1 = partial(monitor1, data)
monitor2 = partial(monitor2, data)

class Arch:
    def __init__(self, arch: ArchConfig, program: List[List[Instruction]], fail: FailSlow, net_name: str, fail_kind: str, model: str, inference_time: int, stage=None):
        print("Constructing hardware architecture.")
        self.env = simpy.Environment()
        self.stage = stage
        self.arch = arch
        self.inference_time = inference_time
        
        self.noc = self.build_noc(arch.noc, model)
        #print(len(self.noc.r2r_links))
        if stage == "pre_analysis":
            init_graph(program)
        self.program = program

        self.cores = self.build_cores(arch.core, program, model, inference_time)

        self.layer_start = [-1 for _ in range(101)]
        self.layer_end = [-1 for _ in range(101)]

        self.net_name = net_name
        self.end_time = 0
        self.fail_kind = fail_kind
        
        trace(self.env, monitor)
        patch_resource(self.cores[9].data_in.store, pre=monitor1, post=monitor2)

        self.fail_slow = fail
        print("Construction finished.")

    def debug(self):
        for d in data:
            logger.info(d)

    def link_fail(self, fail: LinkFail):
        yield self.env.timeout(fail.start_time)
        match fail.direction:
            case Direction.NORTH:
                self.noc.routers[fail.router_id].north_in.change_delay(fail.times)
                self.noc.routers[fail.router_id].north_out.change_delay(fail.times)
            case Direction.SOUTH:
                self.noc.routers[fail.router_id].south_in.change_delay(fail.times)
                self.noc.routers[fail.router_id].south_out.change_delay(fail.times)
            case Direction.EAST:
                self.noc.routers[fail.router_id].east_in.change_delay(fail.times)
                self.noc.routers[fail.router_id].east_out.change_delay(fail.times)
            case Direction.WEST:
                self.noc.routers[fail.router_id].west_in.change_delay(fail.times)
                self.noc.routers[fail.router_id].west_out.change_delay(fail.times)
        
        yield self.env.timeout(fail.end_time-fail.start_time)
        match fail.direction:
            case Direction.NORTH:
                self.noc.routers[fail.router_id].north_in.recover_delay(fail.times)
                self.noc.routers[fail.router_id].north_out.recover_delay(fail.times)
            case Direction.SOUTH:
                self.noc.routers[fail.router_id].south_in.recover_delay(fail.times)
                self.noc.routers[fail.router_id].south_out.recover_delay(fail.times)
            case Direction.EAST:
                self.noc.routers[fail.router_id].east_in.recover_delay(fail.times)
                self.noc.routers[fail.router_id].east_out.recover_delay(fail.times)
            case Direction.WEST:
                self.noc.routers[fail.router_id].west_in.recover_delay(fail.times)
                self.noc.routers[fail.router_id].west_out.recover_delay(fail.times)

    def router_fail(self, fail: RouterFail):
        yield self.env.timeout(fail.start_time)
        self.noc.routers[fail.router_id].router_fail(fail.times)
        yield self.env.timeout(fail.end_time-fail.start_time)
        self.noc.routers[fail.router_id].router_recover(fail.times)

    def lsu_fail(self, fail: LsuFail):
        yield self.env.timeout(fail.start_time)
        self.cores[fail.pe_id].lsu_fail(fail.times)
        yield self.env.timeout(fail.end_time-fail.start_time)
        self.cores[fail.pe_id].lsu_recover(fail.times)

    def tpu_fail(self, fail: TpuFail):
        yield self.env.timeout(fail.start_time)
        self.cores[fail.pe_id].tpu_fail(fail.times)
        yield self.env.timeout(fail.end_time-fail.start_time)
        self.cores[fail.pe_id].tpu_recover(fail.times)

    def run_fail_slow(self):
        for link_fail in self.fail_slow.link:
            self.env.process(self.link_fail(link_fail))
        
        for router_fail in self.fail_slow.router:
            self.env.process(self.router_fail(router_fail))

        for lsu_fail in self.fail_slow.lsu:
            self.env.process(self.lsu_fail(lsu_fail))

        for tpu_fail in self.fail_slow.tpu:
            self.env.process(self.tpu_fail(tpu_fail))

    def build_cores(self, config: CoreConfig, program: List[List[Instruction]], model: str, inference_time: int) -> List[Core]:
        cores = []
        for id in range(config.x * config.y):
            link1 = Link(self.env, LinkConfig(width=128, delay=1))
            link2 = Link(self.env, LinkConfig(width=128, delay=1))
            core = Core(self.env, config, program[id], id, self, link1, link2, model, inference_time, self.stage)

            self.noc.routers[id].bound_with_core(link1, link2)
            cores.append(core)
            # TODO:timer should be in second stage
        if self.stage == "post_analysis":
            self.timer = Timer(self.env, 20000, cores)

        for id in range(config.x * config.y):
            cores[id].scheduler.bound_cores(cores)
            cores[id].core_bound_cores(cores)

        return cores

    def build_noc(self, config: NoCConfig, model: str) -> NoC:
        print("Building NoC architecture.")
        return NoC(self.env, config, model).build_connection()


    # 输出可视化文件
    def make_print_lsu():
        # 对于每个lsu
        count = 0
        # req->count+=1 release->count-=1
    def processesmonitor(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            f.write("[\n")
            for idx, line in enumerate(data):
                task,ts, lenthqueue, ation, ph = line
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\": \"{task}\",\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"lenthqueue\":{lenthqueue}}}}}")
            f.write("]\n")

    # 这个由学长来编号,对于每个编号(id)怎么处理的逻辑我已经写好了:
    def processesmonitorlink(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            f.write("[\n")
            for idx, line in enumerate(data):
                task,ts, lenthqueue, ation, ph,dest = line
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\": \"{task}\",\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"lenthqueue\":{lenthqueue},\"dest\":{dest}}}}}")
            f.write("]\n")

    def processesspm(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            f.write("[\n")
            for idx, line in enumerate(data):
                task, capacity, action, size ,ts , ph= line
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\":\"{task}\" ,\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"act\":\"{action}\",\"capacity\":{capacity},\"size\":{size}}}}}")
            f.write("]\n")

    def processesflow(self,data,file,id,source):
         if len(data)==0:
            return
         with open(file,"w") as f: 
            f.write("[\n")
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
            f.write("]\n")

                
                            
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

    def draw(self):
        L, H = 4, 4
        data = [
            [3, 1, 1, 2],
            [3, 3, 2, 3],
            [1, 1, 3, 3],
            [3, 3, 3, 1]
        ]
        links = []
        for i in self.noc.r2r_links:
            if i.tag == 1:
                links.append((i.corefrom, i.coreto, i.hop))
            else:
                print(i.tag)
        draw_grid(L, H, data, links)

    # 输出采集的数据（two stages）
    def process(self):
        for pos, pe in enumerate(self.program):
            for id, inst in enumerate(pe):
                print(pos, inst.index, inst.inst_type, inst.hot)

    def get_id(self, x: int, y: int):
        return x * self.arch.noc.y + y

    def output_data(self, net: str, fail: str):
        print("Writing performance data...")

        fail = fail.split("/")
        fail = fail[-1].split(".")[0]

        file_path = os.path.join("data/", net)
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        file_path = os.path.join(file_path, fail)
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        inst_file = os.path.join(file_path, "inst_info.txt")
        compute_trace = []
        
        # inst_index -> record 
        comm_record = {}
        comm_trace = []

        recv_insts = []

        with open(inst_file, "w") as file:
            # 指令时间数据
            for id, insts in enumerate(self.program):
                # 每个PE的指令
                for inst in insts:
                    # 没有被执行的指令（正常应该没有）
                    if inst.record.exe_start_time == []:
                        print(f"Instruction{inst.index} not executed.", file=file)
                    
                    if inst.inst_type in compute_task:
                        # assert len(inst.record.ready_run_time) > 0
                        # assert len(inst.record.exe_end_time) == 1
                        # assert len(inst.record.exe_start_time) == 1
                        
                        for time in range(self.inference_time):
                            print(f"Instruction{inst.index}: type {inst.inst_type}, layer_id {inst.layer_id}, pe_id {inst.record.pe_id}", file=file)
                            print(f"    ready_time {inst.record.ready_run_time[time][0]}, exe_time {inst.record.exe_end_time[time][0]-inst.record.exe_start_time[time][0]}, end_time {inst.record.exe_start_time[time][0]}", file=file)
                            print(f"    operands_time: {inst.record.mulins}", file=file)
                            
                            compute_trace.append(
                                CompInst(
                                    instruction_id = inst.index,
                                    instruction_type = inst.inst_type,
                                    layer_id = inst.layer_id,
                                    pe_id = inst.record.pe_id,
                                    start_time = inst.record.exe_start_time[time][0],
                                    end_time = inst.record.exe_end_time[time][0],
                                    flops = inst.record.flops,
                                    inference_time = time
                                )
                            )
                    elif inst.inst_type in io_task:
                        if len(inst.record.exe_start_time) == 0:
                            print(f"exe_start error:: {inst.index} on PE{id}")

                        for time in range(self.inference_time):
                            comm_trace.append(
                                CommInst(
                                    instruction_id = inst.index,
                                    instruction_type = inst.inst_type,
                                    # 当前层的id
                                    layer_id = inst.layer_id,
                                    pe_id = inst.record.pe_id,
                                    start_time = inst.record.exe_start_time[time][0],
                                    end_time = inst.record.exe_end_time[time][0],
                                    data_size = Slice(tensor_slice=inst.tensor_slice).size(),
                                    inference_time = time
                                )
                            )
                    else:
                        # 按pe遍历指令，可能存在某条RECV在SEND之前被访问到，所以先只处理SEND
                        # record里的数据是按推理次数排序的, 下标和推理次数一致
                        if inst.inst_type == TaskType.SEND:
                            comm_record[inst.index] = inst.record
                        else:
                            recv_insts.append(inst)
            
            # comm_record是send指令信息
            for recv_inst in recv_insts:
                for time in range(self.inference_time):
                    comm_trace.append(
                        CommInst(
                            instruction_id = recv_inst.index,
                            instruction_type = recv_inst.inst_type,
                            # 当前层的id
                            layer_id = recv_inst.layer_id,
                            pe_id = recv_inst.record.pe_id,
                            # send完成时间为数据包在noc中开始传输的时间
                            start_time = comm_record[recv_inst.index].exe_end_time[time][0],
                            # recv就绪时间为数据包完成noc传输的时间
                            end_time = recv_inst.record.ready_run_time[time][0],
                            data_size = Slice(tensor_slice=recv_inst.tensor_slice).size(),
                            src_id = comm_record[recv_inst.index].pe_id,
                            dst_id = recv_inst.record.pe_id,
                            inference_time = time
                        )
                    )

        compute_trace = CompTrace(trace=compute_trace)
        comp_json_file = os.path.join(file_path, "comp_trace.json")
        with open(comp_json_file, "w") as file:
            comp_json = compute_trace.model_dump_json(indent=4)
            print(comp_json, file=file)

        comm_trace = CommTrace(trace=comm_trace)
        comp_json_file = os.path.join(file_path, "comm_trace.json")
        with open(comp_json_file, "w") as file:
            comp_json = comm_trace.model_dump_json(indent=4)
            print(comp_json, file=file)

        layer_file = os.path.join(file_path, "layer_info.json") 
        with open(layer_file, "w") as file:
            layer_info = []
            for id, time in enumerate(self.layer_start):
                layer_info.append(
                    LayerGroupInfo(
                        start = time,
                        end = self.layer_end[id]
                    )
                )
            
            layer_info = LayerGroupsInfo(info=layer_info)
            layer_info = layer_info.model_dump_json(indent=4)
            print(layer_info, file=file)

        link_data = os.path.join(file_path, "link_data.txt")
        with open(link_data, "w") as file:
            link_output = {}
            for link in self.noc.r2r_links:
                src_core_id = self.get_id(link.corefrom[0], link.corefrom[1])
                dst_core_id = self.get_id(link.coreto[0], link.coreto[1])

                if src_core_id > dst_core_id:
                    src_core_id, dst_core_id = dst_core_id, src_core_id

                bandwidth = link.tot_size / self.end_time
                tag = (src_core_id, dst_core_id)
                if tag not in link_output:
                    link_output[tag] = bandwidth
                else:
                    link_output[tag] += bandwidth

            print("Global Link Bandwidth:", file=file)
            for key in link_output.keys():
                src_core_id = key[0]
                dst_core_id = key[1]
                bandwidth = link_output[key]
                print(f"Link_{src_core_id}_{dst_core_id}: ground_truth: {bandwidth} B/cycle.", file=file)

        layer_link_data_file = os.path.join(file_path, "layer_link_data.json")
        layer_link_data = []
        for link in self.noc.r2r_links:
            src_core_id = self.get_id(link.corefrom[0], link.corefrom[1])
            dst_core_id = self.get_id(link.coreto[0], link.coreto[1])

            for id, size in link.layer_size.items():
                layer_link_data.append(
                    LinkData(
                        src_id = src_core_id,
                        dst_id = dst_core_id,
                        layer_id = id,
                        data_size = size
                    )
                )

        layer_link_data = LinksData(data=layer_link_data)

        with open(layer_link_data_file, "w") as file:
            layer_link_data_json = layer_link_data.model_dump_json(indent=4)
            print(layer_link_data_json, file=file)


        print("Finished.")

    def run(self):
        print("Start simulation.")
        
        self.run_fail_slow()

        self.env.run()

        for id in range(len(self.cores)):
            self.end_time = max(self.end_time, self.cores[id].end_time[0])
            print(f"PE{id} processed [{self.cores[id].scheduler.task_counter}/{len(self.cores[id].scheduler.tasks)}] instructions.")
            print(f"Max buffer usage is {self.cores[id].spm_manager.max_buf}. [{self.cores[id].spm_manager.container.capacity-self.cores[id].spm_manager.container.level}/{self.cores[id].spm_manager.container.capacity}]")

        print("Simulation finished.")
        # 将值传入json文件
        self.make_print()
        if self.stage == "post_analysis":
            self.process()

        self.output_data(self.net_name, self.fail_kind)

        # self.draw()

        return self.env