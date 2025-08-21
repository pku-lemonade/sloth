import simpy
import os
import sys
import logging
from functools import partial, wraps
from evaluater.core import Core
from evaluater.noc import NoC, Link, Direction
from common.arch_config import CoreConfig, NoCConfig, ArchConfig, LinkConfig, MemConfig
from evaluater.sim_type import *
from common.common import cfg
from recorder.trace_format import *
from typing import List
logger = logging.getLogger("Arch")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class Arch:
    def __init__(self, arch: ArchConfig, program: List[List[Instruction]], fail: FailSlow, net_name: str, fail_kind: str, model: str, inference_time: int, probe, stage=None):
        print("Constructing hardware architecture.")
        self.env = simpy.Environment()
        self.stage = stage
        self.arch = arch
        self.inference_time = inference_time
        
        self.probe = probe
        self.noc = self.build_noc(arch.noc, model)
        self.program = program

        self.cores = self.build_cores(arch.core, program, model, inference_time)

        self.layer_start = [-1 for _ in range(200)]
        self.layer_end = [-1 for _ in range(200)]

        self.net_name = net_name
        self.end_time = 0
        self.fail_kind = fail_kind

        self.fail_slow = fail
        print("Construction finished.")

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
            core = Core(self.env, config, program[id], id, self, link1, link2, model, inference_time, self.probe, self.stage)

            self.noc.routers[id].bound_with_core(link1, link2)
            cores.append(core)

        for id in range(config.x * config.y):
            cores[id].scheduler.bound_cores(cores)
            cores[id].core_bound_cores(cores)

        return cores

    def build_noc(self, config: NoCConfig, model: str) -> NoC:
        print("Building NoC architecture.")
        return NoC(self.env, config, model).build_connection()


    def make_print_lsu():
        count = 0
    def processesmonitor(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            f.write("[\n")
            for idx, line in enumerate(data):
                task,ts, lenthqueue, ation, ph = line
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\": \"{task}\",\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"lenthqueue\":{lenthqueue}}}}}")
            f.write("]\n")

    def processesmonitorlink(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            f.write("[\n")
            for idx, line in enumerate(data):
                task,ts, lenthqueue, ation, ph,dest = line
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
        
        comm_record = {}
        comm_trace = []

        recv_insts = []

        with open(inst_file, "w") as file:
            for id, insts in enumerate(self.program):
                for inst in insts:
                    if inst.record.exe_start_time == []:
                        print(f"Instruction{inst.index} not executed.", file=file)
                    
                    if inst.inst_type in compute_task:
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
                                    layer_id = inst.layer_id,
                                    pe_id = inst.record.pe_id,
                                    start_time = inst.record.exe_start_time[time][0],
                                    end_time = inst.record.exe_end_time[time][0],
                                    data_size = Slice(tensor_slice=inst.tensor_slice).size(),
                                    inference_time = time
                                )
                            )
                    else:
                        if inst.inst_type == TaskType.SEND:
                            comm_record[inst.index] = inst.record
                        else:
                            recv_insts.append(inst)
            
            for recv_inst in recv_insts:
                for time in range(self.inference_time):
                    comm_trace.append(
                        CommInst(
                            instruction_id = recv_inst.index,
                            instruction_type = recv_inst.inst_type,
                            layer_id = recv_inst.layer_id,
                            pe_id = recv_inst.record.pe_id,
                            start_time = comm_record[recv_inst.index].exe_end_time[time][0],
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

    def probe_output(self, workload: str, fail: str):
        print("Writing performance data...")

        fail = fail.split("/")
        fail = fail[-1].split(".")[0]

        file_path = ("trace")

        file_path = os.path.join(file_path, workload)
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        file_path = os.path.join(file_path, fail)
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        compute_trace = []
        
        comm_record = {}
        comm_trace = []

        recv_probes = []

        for id, core in enumerate(self.cores):
            for inst_index, probe_data in core.probe_data.items():
                task_id = core.scheduler.index2taskid[inst_index]

                if core.scheduler.new_program[task_id].inst_type in compute_task:
                    inference_time = inst_index // INST_OFFSET
                    compute_trace.append(
                        CompInst(
                            instruction_id = probe_data.metric["instruction_id"],
                            instruction_type = probe_data.metric["instruction_type"],
                            layer_id = probe_data.metric["layer_id"],
                            pe_id = probe_data.metric["pe_id"],
                            start_time = probe_data.metric["start_time"],
                            end_time = probe_data.metric["end_time"],
                            inference_time = inference_time,
                            flops = probe_data.metric["flops"]
                        )
                    )
                elif core.scheduler.new_program[task_id].inst_type in communication_task:
                    if core.scheduler.new_program[task_id].inst_type == TaskType.SEND:
                        comm_record[inst_index] = probe_data
                    else:
                        recv_probes.append((inst_index, probe_data))

        for (inst_index, recv_probe) in recv_probes:
            inference_time = recv_probe.metric["instruction_id"] // INST_OFFSET
            comm_trace.append(
                CommInst(
                    instruction_id = inst_index,
                    instruction_type = recv_probe.metric["instruction_type"],
                    layer_id = recv_probe.metric["layer_id"],
                    pe_id = recv_probe.metric["pe_id"],
                    start_time = comm_record[inst_index].metric["end_time"],
                    end_time = recv_probe.metric["start_time"],
                    inference_time = inference_time,
                    data_size = comm_record[inst_index].metric["data_size"],
                    src_id = comm_record[inst_index].metric["src_id"],
                    dst_id = recv_probe.metric["dst_id"],
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

    def run(self):
        print("Start simulation.")
        
        self.run_fail_slow()

        self.env.run()

        for id in range(len(self.cores)):
            self.end_time = max(self.end_time, self.cores[id].end_time[0])
            print(f"PE{id} processed [{self.cores[id].scheduler.task_counter}/{len(self.cores[id].scheduler.tasks)}] instructions.")
            print(f"Max buffer usage is {self.cores[id].spm_manager.max_buf}. [{self.cores[id].spm_manager.container.capacity-self.cores[id].spm_manager.container.level}/{self.cores[id].spm_manager.container.capacity}]")

        print("Simulation finished.")
        self.make_print()

        self.probe_output(self.net_name, self.fail_kind)
        
        return self.env