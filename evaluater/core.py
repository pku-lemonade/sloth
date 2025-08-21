import copy
import heapq
import simpy
import logging
from queue import Queue
from common.common import MonitoredResource, cfg
from common.arch_config import CoreConfig, ScratchpadConfig
from evaluater.noc import Link, Router
from evaluater.sim_type import *
from typing import List
from common.distribution import CoreDist
from compiler.probing import fail_probing

logger = logging.getLogger("PE")

class SPMManager:
    def __init__(self, env, id, config: ScratchpadConfig):
        self.delay = config.delay
        self.env = env
        self.id = id
        self.container = simpy.Container(self.env, init=config.size, capacity=config.size)
        self.max_buf = 0
        self.data = []

    def allocate(self, string, size):
        if size > 0:
            logger.debug("Time %.2f: PE%d before-allocate: %d, [%d/%d]", self.env.now, self.id, size, self.container.level, self.container.capacity)
            yield self.container.get(size)
            logger.debug("Time %.2f: PE%d after-allocate: %d, [%d/%d]", self.env.now, self.id, size, self.container.level, self.container.capacity)

        self.max_buf = max(self.max_buf, self.container.capacity-self.container.level)
        self.data.append((string, self.container.level, "alloc", size, self.env.now, "B"))

    def free(self, string, size):
        if size > 0:
            logger.debug("Time %.2f: PE%d before-free: %d, [%d/%d]", self.env.now, self.id, size, self.container.level, self.container.capacity)
            yield self.container.put(size)
            logger.debug("Time %.2f: PE%d after-free: %d, [%d/%d]", self.env.now, self.id, size, self.container.level, self.container.capacity)

        self.data.append((string, self.container.level, "free", size, self.env.now, "E"))

class TableScheduler:
    def __init__(self, program, spm, block_size, id, env, arch, data_in, model, inference_time, probe, stage):
        self.program = program
        self.new_program = []
        self.spm = spm

        self.block_size = block_size
        self.block_ptr = -1
        self.block_counter = 0

        self.block_start = 0
        self.start_first = 0
        self.env = env
        
        self.block_time = []
        self.block_hot = []

        self.data_in = data_in
        self.start = 0
        self.end = 0

        self.id = id
        
        self.finish = False
        self.task_counter = 0

        self.stage = stage
        self.arch = arch
        self.model = model

        self.inference_time = inference_time

        self.tasks = []

        self.index2taskid = {}
        self.taskid2index = {}

        self.waiting_queue = Queue()

        self.tag = [True for _ in range(len(self.program)*inference_time)]
        self.comp_inst = [TaskType.CONV, TaskType.POOL, TaskType.ELEM, TaskType.FC, TaskType.GCONV, TaskType.PTP, TaskType.TRANS]

        for infe_time in range(self.inference_time):
            for id, inst in enumerate(self.program):
                true_index = inst.index + INST_OFFSET * infe_time
                task_id = id + len(self.program) * infe_time

                self.index2taskid[true_index] = task_id
                self.taskid2index[task_id] = true_index

                inference_end = False
                if id == len(self.program)-1:
                    inference_end = True

                new_inst = copy.deepcopy(inst)
                new_inst.index = true_index
                new_inst.inference_end = inference_end
                self.new_program.append(new_inst)

        for id, inst in enumerate(self.new_program):
            match inst.inst_type:
                case TaskType.STAY:
                    self.tasks.append(Stay(index=inst.index, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst))
                case TaskType.RECV:
                    self.tasks.append(Recv(index=inst.index, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst))
                case TaskType.READ:
                    self.tasks.append(Read(index=inst.index, feat_num=inst.feat_num, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst))
                case TaskType.WRITE:
                    self.tasks.append(Write(index=inst.index, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst))
                case TaskType.SEND:
                    self.tasks.append(Send(index=inst.index, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst, dst=inst.position))
                case TaskType.CONV:
                    self.tasks.append(Conv(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst, layer_id=inst.layer_id))
                case TaskType.POOL:
                    self.tasks.append(Pool(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst, layer_id=inst.layer_id))
                case TaskType.ELEM:
                    self.tasks.append(Elem(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst, layer_id=inst.layer_id))
                case TaskType.FC:
                    self.tasks.append(FC(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst, layer_id=inst.layer_id))
                case TaskType.GCONV:
                    self.tasks.append(GConv(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst, layer_id=inst.layer_id, group_num=inst.group_num))
                case TaskType.PTP:
                    self.tasks.append(PTP(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst, layer_id=inst.layer_id))
                case TaskType.TRANS:
                    self.tasks.append(Trans(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inference_end=inst.inference_end, inst=inst, layer_id=inst.layer_id))

        self.tasks = fail_probing(tasks=self.tasks, fragment=probe[0], type=probe[1], location=probe[2], level=probe[3], structure=probe[4])
        self.tasks = fail_probing(tasks=self.tasks, fragment="Route", type="Comm", location="Surround", level="Inst", structure="List")

        self.task_block_update()

    def bound_cores(self, cores):
        self.cores = []
        for core in cores:
            if core == None or core.id == self.id:
                self.cores.append(None)
            else:
                self.cores.append(core)

    def back_add(self, delta_time):
        i = 1
        while delta_time > 0:
            val = min(self.block_time[-i], delta_time)
            delta_time -= val
            self.block_hot[-i] += val
            i += 1

    def task_block_update(self):
        if self.start_first == 0:
            self.start_first = 1
        else:
            self.block_time.append(self.env.now-self.block_start)
            self.block_hot.append(0)
            self.block_start = self.env.now

        logger.debug("PE%d is task_block_updating", self.id)
        self.block_counter = 0
        self.block_ptr += 1
        
        self.start = self.block_ptr * self.block_size
        self.end = min((self.block_ptr + 1) * self.block_size, len(self.tasks))

        for id, task in enumerate(self.tasks[self.start:self.end], start=self.start):
            inst_id = id % len(self.program)
            if self.program[inst_id].start_time != -1:
                delta_time = self.env.now-self.program[inst_id].start_time
                self.back_add(delta_time)

            logger.debug("-"*30)
            logger.debug("inst_id is %d, index is %d, type is %d", id, self.program[inst_id].index, self.program[inst_id].inst_type)

            match self.program[inst_id].inst_type:
                case TaskType.SEND:
                    if task.feat:
                        if self.tag[id]:
                            self.tag[id] = False
                            logger.debug("insert %d into waiting queue", id)
                            self.waiting_queue.put(id)
                case TaskType.WRITE:
                    if task.feat:
                        if self.tag[id]:
                            self.tag[id] = False
                            logger.debug("insert %d into waiting queue", id)
                            self.waiting_queue.put(id)
                case TaskType.RECV:
                    if task.feat:
                        logger.debug("data%d has already arrived", self.program[inst_id].index)
                case TaskType.STAY:
                    if self.tag[id]:
                        self.tag[id] = False
                        self.waiting_queue.put(id)
                case _:
                    para = len(task.para)
                    feat = len(task.feat)
                    logger.debug("para is %d/%d, feat is %d/%d", para, task.para_num, feat, task.feat_num)
                    if para == task.para_num and feat == task.feat_num:
                        if self.tag[id]:
                            self.tag[id] = False
                            logger.debug("insert %d into waiting queue", id)
                            self.waiting_queue.put(id)

        if self.block_counter == self.block_size:
            self.task_block_update()

    def data_update(self, data):
        logger.debug("updating data%d", data.index)
        self.task_counter += 1
        task_id = self.index2taskid[data.index]
        inference_time = task_id // len(self.program)

        logger.debug("current block_ptr is %d", self.block_ptr)
        logger.debug("task block_ptr is %d", task_id // self.block_size)

        inst_id = task_id
        layer_id = self.new_program[inst_id].layer_id

        if self.arch.layer_start[layer_id] == -1:
            self.arch.layer_start[layer_id] = self.env.now
        self.arch.layer_end[layer_id] = max(self.arch.layer_end[layer_id], self.env.now)

        if task_id // self.block_size == self.block_ptr:
            self.block_counter += 1
            logger.debug("PE%d self.counter += 1, [%d/%d]", self.id, self.block_counter, self.block_size)

            if self.block_counter == self.block_size:
                self.task_block_update()

        self.tasks[task_id].feat.append(data)
        self.new_program[inst_id].record.exe_end_time.append((self.env.now, inference_time))

        for idx in range(len(self.new_program[inst_id].trigger_index)):
            true_trigger_index = self.new_program[inst_id].trigger_index[idx] + inference_time * INST_OFFSET

            logger.debug("data%d triggered %d", data.index, true_trigger_index)
            tri_task_id = self.index2taskid[true_trigger_index]
            tri_inst_id = tri_task_id % len(self.new_program)

            self.new_program[tri_inst_id].record.mulins.append((self.env.now, inference_time))
            if self.new_program[inst_id].data_type == DataType.FEAT:
                self.tasks[tri_task_id].feat.append(data)
            else:
                self.tasks[tri_task_id].para.append(data)
            
            logger.debug("triggered block_ptr is %d", tri_task_id // self.block_size)
            if tri_task_id // self.block_size == self.block_ptr:
                logger.debug("update triggered instruction...")
                para_len = len(self.tasks[tri_task_id].para)
                feat_len = len(self.tasks[tri_task_id].feat)
                logger.debug("para:%d/%d + feat:%d/%d", para_len, self.tasks[tri_task_id].para_num, feat_len, self.tasks[tri_task_id].feat_num)

                if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                    if self.tag[tri_task_id]:
                        logger.debug("PE%d insert %d into waiting_queue", self.id, tri_task_id)
                        self.tag[tri_task_id] = False
                        self.waiting_queue.put(tri_task_id)

            else:
                para_len = len(self.tasks[tri_task_id].para)
                feat_len = len(self.tasks[tri_task_id].feat)
                if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                    self.new_program[tri_inst_id].start_time = self.env.now

    def task_update(self, task):
        inst_index = task.index
        logger.debug("updating task%d", inst_index)
        self.task_counter += 1
        task_id = self.index2taskid[inst_index]
        inference_time = task_id // len(self.program)

        logger.debug("current block_ptr is %d", self.block_ptr)
        logger.debug("task block_ptr is %d", task_id // self.block_size)

        inst_id = task_id
        layer_id = self.new_program[task_id].layer_id

        if self.arch.layer_start[layer_id] == -1:
            self.arch.layer_start[layer_id] = self.env.now

        self.arch.layer_end[layer_id] = max(self.arch.layer_end[layer_id], self.env.now)

        self.block_counter += 1
        logger.debug("PE%d self.counter += 1, [%d/%d]", self.id, self.block_counter, self.block_size)
        if self.block_counter == self.block_size:
            self.task_block_update()

        flag = False
        if self.new_program[inst_id].inst_type == TaskType.WRITE:
            if len(self.new_program[inst_id].trigger_index) != 0:

                for id in range(len(self.new_program[inst_id].trigger_core_id)):
                    true_trigger_index = self.new_program[inst_id].trigger_index[id] + inference_time * INST_OFFSET

                    tri_core_id = self.new_program[inst_id].trigger_core_id[id]
                    pur_sche = self.cores[tri_core_id].scheduler if tri_core_id != self.id else self
                    tri_task_id = pur_sche.index2taskid[true_trigger_index]

                    logger.debug("task%d[core%d] triggered task%d[core%d]", inst_index, self.id, true_trigger_index, tri_core_id)

                    match self.new_program[inst_id].data_type:
                        case DataType.PARA:
                            pur_sche.tasks[tri_task_id].para.append(Data())
                        case DataType.FEAT:
                            pur_sche.tasks[tri_task_id].feat.append(Data())
                    
                    tri_inst_id = tri_task_id
                    if tri_task_id // pur_sche.block_size != pur_sche.block_ptr:
                        continue
                    
                    logger.debug("update triggered instruction...")
                    para_len = len(pur_sche.tasks[tri_task_id].para)
                    feat_len = len(pur_sche.tasks[tri_task_id].feat)
                    logger.debug("para:%d/%d + feat:%d/%d", para_len, pur_sche.tasks[tri_task_id].para_num, feat_len, pur_sche.tasks[tri_task_id].feat_num)

                    if feat_len == pur_sche.tasks[tri_task_id].feat_num and para_len == pur_sche.tasks[tri_task_id].para_num:
                        if pur_sche.tag[tri_task_id]:
                            logger.debug("PE%d put %d into data_in", tri_core_id, tri_task_id)
                            pur_sche.tag[tri_task_id] = False

                            if tri_core_id != self.id:
                                ins = copy.deepcopy(self.new_program[inst_id])
                                ins.index = true_trigger_index
                                if self.model == "basic":
                                    self.cores[tri_core_id].data_in.put(Message(ins=ins, data=Data(index=true_trigger_index, tensor_slice=self.new_program[inst_id].tensor_slice), dst=tri_core_id, src=self.id))
                                elif self.model == "packet":
                                    self.cores[tri_core_id].data_in.put_hop(Packet(ins=ins, data=Data(index=true_trigger_index, tensor_slice=self.new_program[inst_id].tensor_slice), dst=tri_core_id, src=self.id, end=True))
                            else:
                                ins = copy.deepcopy(self.new_program[inst_id])
                                ins.index = true_trigger_index
                                if self.model == "basic":
                                    self.data_in.put(Message(ins=ins, data=Data(index=true_trigger_index, tensor_slice=self.new_program[inst_id].tensor_slice), dst=tri_core_id, src=self.id))
                                elif self.model == "packet":
                                    self.data_in.put_hop(Packet(ins=ins, data=Data(index=true_trigger_index, tensor_slice=self.new_program[inst_id].tensor_slice), dst=tri_core_id, src=self.id, end=True))

        if self.new_program[inst_id].inst_type != TaskType.WRITE or flag == True:
            for idx in range(len(self.new_program[inst_id].trigger_index)):
                true_trigger_index = self.new_program[inst_id].trigger_index[idx] + inference_time * INST_OFFSET

                logger.debug("task%d triggered %d", inst_index, true_trigger_index)
                tri_task_id = self.index2taskid[true_trigger_index]

                tri_inst_id = tri_task_id
                match self.new_program[inst_id].data_type:
                    case DataType.PARA:
                        self.tasks[tri_task_id].para.append(Data(tensor_slice=self.new_program[inst_id].tensor_slice))
                    case DataType.FEAT:
                        self.tasks[tri_task_id].feat.append(Data(tensor_slice=self.new_program[inst_id].tensor_slice))

                logger.debug("triggered block_ptr is %d", tri_task_id // self.block_size)
                if tri_task_id // self.block_size == self.block_ptr:
                    logger.debug("update triggered instruction...")
                    para_len = len(self.tasks[tri_task_id].para)
                    feat_len = len(self.tasks[tri_task_id].feat)
                    logger.debug("para:%d/%d + feat:%d/%d", para_len, self.tasks[tri_task_id].para_num, feat_len, self.tasks[tri_task_id].feat_num)

                    if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                        if self.tag[tri_task_id]:
                            logger.debug("PE%d insert %d into waiting_queue", self.id, tri_task_id)
                            self.tag[tri_task_id] = False
                            self.waiting_queue.put(tri_task_id)

                else:
                    para_len = len(self.tasks[tri_task_id].para)
                    feat_len = len(self.tasks[tri_task_id].feat)
                    if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                        self.new_program[tri_inst_id].start_time = self.env.now
    
    def update(self, data):
        self.task_counter += 1
        task_id = self.index2taskid[data.index]
        logger.debug(f"updating {data.index}")
        logger.debug(f"{task_id} // {self.block_size} == {self.block_ptr}")

        layer_id = self.program[task_id].layer_id
        if self.arch.layer_start[layer_id] == -1:
            self.arch.layer_start[layer_id] = self.env.now

        self.arch.layer_end[layer_id] = max(self.arch.layer_end[layer_id], self.env.now)
        
        if self.program[task_id].inst_type != TaskType.RECV:
            assert task_id // self.block_size == self.block_ptr

        if task_id // self.block_size == self.block_ptr:
            logger.debug(f"PE{self.id} self.counter += 1")
            self.block_counter += 1
            logger.debug(f"PE{self.id} self.counter is {self.block_counter}/{self.block_size}")

        if self.block_counter == self.block_size:
            self.task_block_update()

        if self.program[task_id].inst_type == TaskType.WRITE:
            if len(self.program[task_id].trigger_index) != len(self.program[task_id].trigger_core_id):
                print(data.index)
                
            if len(self.program[task_id].trigger_index) != 0:
                logger.debug(f"WRITE trigger {data.index}")
                for id in range(len(self.program[task_id].trigger_index)):

                    tri_core_id = self.program[task_id].trigger_core_id[id]
                    pur_sche = self.cores[tri_core_id].scheduler if tri_core_id != self.id else self
                    tri_task_id = pur_sche.index2taskid[self.program[task_id].trigger_index[id]]
                    logger.debug(f"core_id: {tri_core_id}, task_id: {tri_task_id}")
                    
                    match self.program[task_id].data_type:
                        case DataType.PARA:
                            pur_sche.tasks[tri_task_id].para.append(data)
                        case DataType.FEAT:
                            pur_sche.tasks[tri_task_id].feat.append(data)

                    if tri_task_id // pur_sche.block_size != pur_sche.block_ptr:
                        continue

                    feat_len = len(pur_sche.tasks[tri_task_id].feat)
                    if feat_len == pur_sche.tasks[tri_task_id].feat_num:
                        if pur_sche.tag[tri_task_id]:

                            logger.debug(f"PE{tri_core_id} insert {tri_task_id} into waiting_queue")
                            pur_sche.tag[tri_task_id] = False
                            pur_sche.waiting_queue.put(tri_task_id)
            return
        
        if self.program[task_id].inst_type == TaskType.RECV:
            self.tasks[task_id].feat.append(data)
            self.program[task_id].record.ready_run_time.append(self.env.now)
            self.program[task_id].record.exe_start_time.append(self.env.now)
            self.program[task_id].record.exe_end_time.append(self.env.now)

        for idx in range(len(self.program[task_id].trigger_index)):
            tri_task_id = self.index2taskid[self.program[task_id].trigger_index[idx]]
            self.program[tri_task_id].record.mulins.append(self.env.now)

            match self.program[task_id].data_type:
                case DataType.PARA:
                    self.tasks[tri_task_id].para.append(data)
                case DataType.FEAT:
                    self.tasks[tri_task_id].feat.append(data)

            logger.debug(f"{data.index} has triggered {self.program[task_id].trigger_index[idx]}")
            logger.debug(f"{tri_task_id} // {self.block_size} == {self.block_ptr}")
            if tri_task_id // self.block_size == self.block_ptr:

                para_len = len(self.tasks[tri_task_id].para)
                feat_len = len(self.tasks[tri_task_id].feat)
                logger.debug("inside")
                logger.debug(f"para:{para_len}/{self.tasks[tri_task_id].para_num} + feat:{feat_len}/{self.tasks[tri_task_id].feat_num}")
                if self.program[tri_task_id].inst_type in self.comp_inst:
                    if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                        if self.tag[tri_task_id]:
                            if self.stage=="pre_analysis":
                                self.program[task_id].next.append(self.program[tri_task_id])

                            logger.debug(f"PE{self.id} insert {tri_task_id} into waiting_queue")
                            self.tag[tri_task_id] = False
                            self.waiting_queue.put(tri_task_id)
                else:
                    if feat_len == self.tasks[tri_task_id].feat_num:
                        if self.tag[tri_task_id]:
                            logger.debug(f"PE{self.id} insert {tri_task_id} into waiting_queue")
                            self.tag[tri_task_id] = False
                            self.waiting_queue.put(tri_task_id)
            
            else:
                para_len = len(self.tasks[tri_task_id].para)
                feat_len = len(self.tasks[tri_task_id].feat)
                if self.program[tri_task_id].inst_type in self.comp_inst:
                    if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                        self.program[tri_task_id].start_time = self.env.now
                else:
                    if feat_len == self.tasks[tri_task_id].feat_num:
                        self.program[tri_task_id].start_time = self.env.now

    def schedule(self):
        if self.waiting_queue.empty():
            return None
        else:
            logger.debug("PE%d block_counter: %d/%d", self.id, self.block_counter, self.block_size)
            if self.block_counter == self.block_size:
                self.task_block_update()

            task_ready = []
            while not self.waiting_queue.empty():
                task_id = self.waiting_queue.get()
                task_ready.append(self.tasks[task_id])

            return task_ready

def print_event_queue(env):
    print("Remaining keys() are:")
    print("="*40)
    
    for event in env._queue:
        print(event)


class Core:
    def __init__(self, env, config: CoreConfig, program: List[Instruction], id: int, arch, link1, link2, model, inference_time, probe, stage=None):
        self.env = env
        self.type = config.type
        self.program = program
        self.id = id
        self.spm_manager = SPMManager(env, self.id, config.spm)
        self.flow_out = []
        self.flow_in = []
        self.stage = stage
        self.model = model
        self.inference_time = inference_time
        self.cur_inference_time = 0
        self.config = config

        self.mu = self.config.tpu.flops
        self.sigma = self.mu * 0.1 / 1.645
        self.core_dist = CoreDist(mu=self.mu, sigma=self.sigma)

        self.bound_with_router(link2, link1)

        self.waitinglist = []

        self.recv_queue = []

        self.end_time = []

        self.scheduler = TableScheduler(self.program, self.spm_manager, config.blk_size, self.id, self.env, arch, self.data_in, model, self.inference_time, probe, stage)

        self.lsu_bandwidth = config.lsu.width
        self.tpu_flops = config.tpu.flops
        self.lsu = MonitoredResource(env=env, capacity=4)
        self.tpu = MonitoredResource(env=env, capacity=1)

        self.probe_data = {}
        
        self.arch = arch
        self.env.process(self.core_run())

    def bound_with_router(self, data_in, data_out):
        self.data_in = data_in
        self.data_out = data_out

    def lsu_fail(self, times):
        self.lsu_bandwidth /= times
    
    def lsu_recover(self, times):
        self.lsu_bandwidth *= times
    
    def tpu_fail(self, times):
        self.tpu_flops /= times
        self.core_dist = CoreDist(mu=self.tpu_flops, sigma=self.tpu_flops*0.1/1.645)

    def tpu_recover(self, times):
        self.tpu_flops *= times
        self.core_dist = CoreDist(mu=self.tpu_flops, sigma=self.tpu_flops*0.1/1.645)

    def core_bound_cores(self, cores):
        self.cores = []
        for core in cores:
            if core.id == self.id:
                self.cores.append(None)
            else:
                self.cores.append(core)

    def receive_data(self, msg):
        logger.debug("in function receive_data()")

        if self.model == "packet":
            msg = Message(ins=msg.ins, src=msg.src, dst=msg.dst, data=msg.data)

        task_id = self.scheduler.index2taskid[msg.data.index]
        if task_id in range(self.scheduler.start, self.scheduler.end):
            slice = Slice(tensor_slice=msg.data.tensor_slice)

            yield self.env.process(self.spm_manager.allocate("recv"+str(msg.data.index), slice.size()))
            self.scheduler.tasks[task_id].probe_ed.run(self, msg.data.index, msg.ins.layer_id, "Recv", dst=self.id)
            
            self.scheduler.data_update(msg.data)
        else:
            logger.debug("PE%d insert data%d into recv_queue", self.id, msg.data.index)
            
            msg.data.index = self.scheduler.index2taskid[msg.data.index]
            heapq.heappush(self.recv_queue, msg)

    def core_run(self):
        self.running_event = []
        self.running_send = []
        self.event2task = {}

        while True:
            while self.recv_queue:
                top = self.recv_queue[0]
                if top.data.index in range(self.scheduler.start, self.scheduler.end):      
                    msg = heapq.heappop(self.recv_queue)

                    msg.data.index = self.scheduler.taskid2index[msg.data.index]
                    logger.debug("PE%d pop data%d from recv_queue", self.id, msg.ins.index)

                    yield self.env.process(self.receive_data(msg))
                else:
                    break

            task_ready = self.scheduler.schedule()
            if task_ready:
                for task in task_ready:

                    instruction = self.scheduler.new_program[self.scheduler.index2taskid[task.index]]
                    task_event = None

                    if task.opcode == "Send":
                        if self.model == "basic":
                            task_event = self.env.process(task.run(self, instruction))
                        elif self.model == "packet":
                            task_event = self.env.process(task.run_hop(self, instruction))
                    else:
                        task_event = self.env.process(task.run(self, instruction))
                    
                    logger.debug("Time %.2f: PE%d add a %s task(id:%d, layer:%d) into running queue.", self.env.now, self.id, type(task), task.index, self.scheduler.new_program[self.scheduler.index2taskid[task.index]].layer_id)

                    if task_event:
                        self.running_event.append(task_event)
                        self.event2task[task_event] = task

            with self.data_in.get() as msg_arrive:
                result = yield simpy.events.AnyOf(self.env, self.running_event + [msg_arrive])

                logger.debug("Time %.2f: PE%d's result is %s", self.env.now, self.id, result)
            
                if msg_arrive.triggered:
                    msg = msg_arrive.value

                    task_id = self.scheduler.index2taskid[msg.ins.index]
                    self.scheduler.tasks[task_id].probe_st.run(self, msg.data.index, msg.ins.layer_id, "Recv", data_size=Slice(tensor_slice=msg.data.tensor_slice).size())

                    if cfg.flow and self.env.now >= cfg.simstart and self.env.now <= cfg.simend:
                        self.flow_in.append((msg.ins.index, self.scheduler.new_program[inst_id].inst_type, "recv", self.env.now))

                    logger.debug("Time %.2f: PE%d receive data%d", self.env.now, self.id, msg.ins.index)
                    logger.debug("function call: receive_data()")

                    if self.model == "basic":
                        yield self.env.process(self.receive_data(msg))
                    elif self.model == "packet":
                        if msg.end:
                            yield self.env.process(self.receive_data(msg))

                    if self.data_in.len() > 0:
                        if self.scheduler.index2taskid[self.data_in.store.items[0].data.index] not in range(self.scheduler.start, self.scheduler.end):
                            break

                    while self.data_in.len() > 0:
                        msg = yield self.data_in.get()

                        logger.debug("Time %.2f: PE%d receive data%d", self.env.now, self.id, msg.ins.index)
                        logger.debug("received data is %s", msg.ins)

                        logger.debug("function call: receive_data()")
                        if self.model == "basic":
                            yield self.env.process(self.receive_data(msg))
                        elif self.model == "packet":
                            if msg.end:
                                yield self.env.process(self.receive_data(msg))

                for event in self.running_event:
                    if event.triggered:
                        task = self.event2task[event]
                        inst_id = self.scheduler.index2taskid[task.index]

                        logger.debug("Time %.2f: PE%d finish processing %s task(id:%d).", self.env.now, self.id, type(self.event2task[event]), self.event2task[event].index)

                        self.scheduler.task_update(self.event2task[event])

                        self.running_event.remove(event)

            if self.scheduler.task_counter == len(self.scheduler.tasks):
                self.end_time.append(self.env.now)
                self.cur_inference_time += 1
                
                print(f"Time {self.env.now:.2f}: PE{self.id} finished processing all of its instructions.")