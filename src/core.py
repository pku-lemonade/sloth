import heapq
import simpy
import logging
from queue import Queue
from src.common import MonitoredResource, cfg, ind2ins
from src.arch_config import CoreConfig, ScratchpadConfig
from src.noc_new import Link, Router
from src.sim_type import *
from typing import List

logger = logging.getLogger("PE")
waitready = []

# class SPMManager:
#     def __init__(self, env,config: ScratchpadConfig):
#         self.size = config.size
#         self.delay = config.delay
#         self.env=env
#         self.capacity = self.size
#         self.data=[]

#     def allocate(self, string, size):
#         if size > self.capacity:
#             raise ValueError("Not enough space in SPM")
#         self.capacity -= size
#         #TODO:add instruction type and id and check if need record
#         self.data.append((string,self.capacity,"alloc",size,self.env.now,"B"))

#     def free(self, string,size):
#         self.capacity += size
#         #TODO：add instruction type and id and check if need record
#         self.data.append((string,self.capacity,"free",size,self.env.now,"E"))

class SPMManager:
    def __init__(self, env, id, config: ScratchpadConfig):
        self.delay = config.delay
        self.env = env
        self.id = id
        self.container = simpy.Container(self.env, init=config.size, capacity=config.size)
        self.max_buf = 0
        self.data=[]

    def allocate(self, string, size):
        if size > 0:
            logger.debug(f"Time {self.env.now:.2f}: PE{self.id} before-allocate: {size}, [{self.container.level}/{self.container.capacity}]")
            yield self.container.get(size)
            logger.debug(f"Time {self.env.now:.2f}: PE{self.id} after-allocate: {size}, [{self.container.level}/{self.container.capacity}]")
        #TODO:add instruction type and id and check if need record
        self.max_buf = max(self.max_buf, self.container.capacity-self.container.level)
        if "recv" not in string:
            self.data.append((string, self.container.level, "alloc", size, self.env.now, "B"))

    def free(self, string, size):
        if size > 0:
            logger.debug(f"Time {self.env.now:.2f}: PE{self.id} before-free: {size}, [{self.container.level}/{self.container.capacity}]")
            yield self.container.put(size)
            logger.debug(f"Time {self.env.now:.2f}: PE{self.id} after-free: {size}, [{self.container.level}/{self.container.capacity}]")
        #TODO：add instruction type and id and check if need record
        self.data.append((string, self.container.level, "free", size, self.env.now, "E"))

class Graph:
    def __init__(self, num):
        self.node_num = num 
        self.edges = [[] for _ in range(num)]
        self.degree = [0 for _ in range(num)]
        self.tag = [True for _ in range(num)]
        self.queue = Queue()
    
    def addedge(self, start, end):
        self.edges[start].append(end)
        self.degree[end] += 1

    def topo_init(self):
        for point in range(self.node_num):
            if self.degree[point] == 0 and self.tag[point]:
                self.queue.put(point)

    def update(self, node_id):
        for neighbor in self.edges[node_id]:
            self.degree[neighbor] -= 1

            if self.degree[neighbor] == 0 and self.tag[neighbor]:
                self.queue.put(neighbor)

    def topo_pop(self) -> int:
        if self.queue.empty() == False:
            top = self.queue.get()
            self.update(top)
            return top
        else:
            return None


class GraphScheduler:
    def __init__(self, program, spm):
        self.program = program
        self.spm = spm
        self.graph = None

        self.list2inst = {}
        self.index2inst = {}
        self.index2list = {}
        for ind, ins in enumerate(self.program):
            self.list2inst[ind] = ins
            self.index2inst[ins.index] = ins
            self.index2list[ins.index] = ind 
            
    def build_graph(self):
        inst_graph = Graph(len(self.program))

        last_layer = -1
        last_layer_ind = -1

        inputs = []
        outputs = []
        
        for ind, inst in enumerate(self.program):
            if inst.inst_type == "RECV":
                inst_graph.tag[ind] = False

            if inst.layer_id != last_layer:
                if last_layer != -1:
                    inst_start = last_layer_ind
                    inst_end = ind
                    inst_graph.addedge(inst_end-1, inst_end)
                else:
                    inst_start = 0
                    inst_end = ind

                inputs.clear()
                outputs.clear()
                comp_pos = -1

                for idx, layer_inst in enumerate(self.program[inst_start:inst_end], start=inst_start):
                    match layer_inst.inst_type:
                        case "READ":
                            inputs.append(idx)
                        case "RECV":
                            inputs.append(idx)
                        case "WRITE":
                            outputs.append(idx)
                        case "SEND":
                            outputs.append(idx)
                        case "COMP":
                            comp_pos = idx
                    
                for input in inputs:
                    inst_graph.addedge(input, comp_pos)

                for output in outputs:
                    inst_graph.addedge(comp_pos, output)
                    
                last_layer = inst.layer_id
                last_layer_ind = ind

        self.graph = inst_graph
        self.graph.topo_init()

    def update(self, data):
        node_id = self.index2list[data.index]
        self.graph.update(node_id)

    def schedule(self):
        if self.graph.queue.empty():
            return None
        else:
            item = self.graph.topo_pop()
            inst = self.list2inst[item]
            match inst.inst_type:
                case "READ":
                    return Read(index=inst.index, size=inst.size)
                case "WRITE":
                    return Write(index=inst.index, size=inst.size)
                case "SEND":
                    return Send(index=inst.index, size=inst.size, dst=inst.position)
                case "COMP":
                    match inst.operation:
                        case "CONV":
                            return Conv(flops=inst.size, layer_id=inst.layer_id)
                        case "POOL":
                            return Pool(flops=inst.size, layer_id=inst.layer_id)
                        case "FC":
                            return FC(flops=inst.size, layer_id=inst.layer_id)

class TableScheduler:
    def __init__(self, program, spm, block_size, id, env, arch, data_in, stage):
        self.program = program
        self.spm = spm

        self.block_size = block_size
        self.block_ptr = -1
        self.block_counter = 0

        self.block_start = 0
        self.start_first = 0
        self.env = env
        # 记录block的时间list，便于在inst_ready之后将等待的时间返回到block
        self.block_time = []
        self.block_hot = []

        self.data_in = data_in
        
        # 当前block的id区间 [start, end)
        self.start = 0
        self.end = 0

        self.id = id
        self.tag = [True for _ in range(len(self.program))]
        self.finish = False
        self.inst_counter = 0

        self.stage = stage
        self.arch = arch

        self.tasks = []

        self.index2taskid = {}
        self.taskid2index = {}

        self.waiting_queue = Queue()

        self.comp_inst = [TaskType.CONV, TaskType.POOL, TaskType.ELEM, TaskType.FC, TaskType.GCONV, TaskType.PTP, TaskType.TRANS]

        for id, inst in enumerate(self.program):
            # 因为每次只考虑一个block,这样的话可以保证每次观察一条指令的时候，它们的前驱一定有running，从而可以终止
            # 只考虑running_event
            # if self.stage == "post_analysis":
                # if inst.prev == None:
                    #inst.running = True
                # if inst.inst_type == TaskType.RECV:
                    # print("="*40)
                    # print(inst.prev)
                    # print(inst)        
            self.index2taskid[inst.index] = id
            self.taskid2index[id] = inst.index
            match inst.inst_type:
                case TaskType.STAY:
                    self.tasks.append(Stay(index=inst.index, tensor_slice=inst.tensor_slice, inst=inst))
                case TaskType.RECV:
                    self.tasks.append(Recv(index=inst.index, tensor_slice=inst.tensor_slice, inst=inst))
                case TaskType.READ:
                    self.tasks.append(Read(index=inst.index, feat_num=inst.feat_num, tensor_slice=inst.tensor_slice, inst=inst))
                case TaskType.WRITE:
                    self.tasks.append(Write(index=inst.index, tensor_slice=inst.tensor_slice, inst=inst))
                case TaskType.SEND:
                    self.tasks.append(Send(index=inst.index, tensor_slice=inst.tensor_slice, inst=inst, dst=inst.position))
                case TaskType.CONV:
                    self.tasks.append(Conv(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inst=inst, layer_id=inst.layer_id))
                case TaskType.POOL:
                    self.tasks.append(Pool(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inst=inst, layer_id=inst.layer_id))
                case TaskType.ELEM:
                    self.tasks.append(Elem(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inst=inst, layer_id=inst.layer_id))
                case TaskType.FC:
                    self.tasks.append(FC(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inst=inst, layer_id=inst.layer_id))
                case TaskType.GCONV:
                    self.tasks.append(GConv(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inst=inst, layer_id=inst.layer_id, group_num=inst.group_num))
                case TaskType.PTP:
                    self.tasks.append(PTP(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inst=inst, layer_id=inst.layer_id))
                case TaskType.TRANS:
                    self.tasks.append(Trans(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, inst=inst, layer_id=inst.layer_id))

        self.task_block_update()

    def bound_cores(self, cores):
        self.cores = []
        for core in cores:
            if core.id == self.id:
                self.cores.append(None)
            else:
                self.cores.append(core)

    # 去除前几个block中的等待时间，作为热点时间？
    def back_add(self, delta_time):
        i = 1
        while delta_time > 0:
            val = min(self.block_time[-i], delta_time)
            delta_time -= val
            self.block_hot[-i] += val
            i += 1

    def task_block_update(self):
        # 维护block的执行时间
        if self.start_first == 0:
            self.start_first = 1
        else:
            # 上一个block的执行总时间（计算+等待）
            self.block_time.append(self.env.now-self.block_start)
            # 初始化下一个block
            self.block_hot.append(0)
            self.block_start = self.env.now

        logger.debug(f"PE{self.id} is task_block_updating")
        self.block_counter = 0
        self.block_ptr += 1
        self.start = self.block_ptr * self.block_size
        self.end = min((self.block_ptr + 1) * self.block_size, len(self.program))

        for id, inst in enumerate(self.program[self.start:self.end], start=self.start):
            # 当前指令的等待时间
            if inst.start_time != -1:
                delta_time = self.env.now-inst.start_time
                self.back_add(delta_time)

            # print("-"*30)
            # print(f"inst_id is {id}, type is {inst.inst_type}")
            logger.debug("-"*30)
            logger.debug(f"inst_id is {id}, index is {inst.index}, type is {inst.inst_type}")
            match inst.inst_type:
                case TaskType.SEND:
                    if self.tasks[id].feat:
                        # print(f"insert {id} into waiting queue")
                        if self.tag[id]:
                            self.tag[id] = False
                            logger.debug(f"insert {id} into waiting queue")
                            self.waiting_queue.put(id)
                case TaskType.WRITE:
                    if self.tasks[id].feat:
                        # print(f"insert {id} into waiting queue")
                        if self.tag[id]:
                            self.tag[id] = False
                            logger.debug(f"insert {id} into waiting queue")
                            self.waiting_queue.put(id)
                # 只有当前block的RECV才会接收，应该不需要这里的逻辑
                case TaskType.RECV:
                    if self.tasks[id].feat:
                        # if self.tag[id]:
                        #     self.tag[id] = False
                        #     logger.debug(f"self.counter += 1")
                        #     self.block_counter += 1
                        logger.debug(f"data{inst.index} has already arrived")
                case TaskType.STAY:
                    if self.tag[id]:
                        self.tag[id] = False
                        self.waiting_queue.put(id)
                # READ 也在这一类，因为有不同 group layer 的触发关系
                case _:
                    # para = 1 if self.tasks[id].para else 0
                    # feat = 1 if self.tasks[id].feat else 0
                    # # print(f"para is {para}, feat is {feat}")
                    # logger.debug(f"para is {para}, feat is {feat}")
                    # if para + feat == self.tasks[id].num_operands:
                    #     # print(f"insert {id} into waiting queue")
                    #     if self.tag[id]:
                    #         self.tag[id] = False
                    #         logger.debug(f"insert {id} into waiting queue")
                    #         self.waiting_queue.put(id)
                    para = len(self.tasks[id].para)
                    feat = len(self.tasks[id].feat)
                    logger.debug(f"para is {para}/{self.tasks[id].para_num}, feat is {feat}/{self.tasks[id].feat_num}")
                    # 输入特征可能分多次，需要全部读进才能计算
                    if para == self.tasks[id].para_num and feat == self.tasks[id].feat_num:
                        if self.tag[id]:
                            self.tag[id] = False
                            logger.debug(f"insert {id} into waiting queue")
                            self.waiting_queue.put(id)

        if self.block_counter == self.block_size:
            self.task_block_update()

    # 利用接收到的数据更新（RECV指令）
    def data_update(self, data):
        logger.debug(f"updating data{data.index}")
        # 将数据的id转换成core内部task id
        self.inst_counter += 1
        task_id = self.index2taskid[data.index]

        logger.debug(f"current block_ptr is {self.block_ptr}")
        logger.debug(f"task block_ptr is {task_id//self.block_size}")

        # 记录每层开始时间
        layer_id = self.program[task_id].layer_id
        if self.arch.layer_start[layer_id] == -1:
            self.arch.layer_start[layer_id] = self.env.now
        # 记录每层结束时间
        self.arch.layer_end[layer_id] = max(self.arch.layer_end[layer_id], self.env.now)

        # 维护block_counter和block更新（会接收到的数据应该都是在当前block内的，应该可以删掉if）
        if task_id // self.block_size == self.block_ptr:
            self.block_counter += 1
            logger.debug(f"PE{self.id} self.counter += 1, [{self.block_counter}/{self.block_size}]")

            if self.block_counter == self.block_size:
                self.task_block_update()

        # 记录RECV指令的执行信息
        self.tasks[task_id].feat.append(data)
        self.program[task_id].record.exe_end_time.append(self.env.now)

        # 更新被触发的指令
        for idx in range(len(self.program[task_id].trigger_index)):
            logger.debug(f"data{data.index} triggered {self.program[task_id].trigger_index[idx]}")
            tri_task_id = self.index2taskid[self.program[task_id].trigger_index[idx]]

            # 更新被触发指令的操作数信息
            self.program[tri_task_id].record.mulins.append(self.env.now)
            if self.program[task_id].data_type == DataType.FEAT:
                self.tasks[tri_task_id].feat.append(data)
            else:
                self.tasks[tri_task_id].para.append(data)
            
            # 更新被触发的指令
            logger.debug(f"triggered block_ptr is {tri_task_id//self.block_size}")
            # block内指令更新
            if tri_task_id // self.block_size == self.block_ptr:
                logger.debug(f"update triggered instruction...")
                para_len = len(self.tasks[tri_task_id].para)
                feat_len = len(self.tasks[tri_task_id].feat)
                logger.debug(f"para:{para_len}/{self.tasks[tri_task_id].para_num} + feat:{feat_len}/{self.tasks[tri_task_id].feat_num}")
                
                # 判断操作数是否到齐
                if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                    if self.tag[tri_task_id]:
                        logger.debug(f"PE{self.id} insert {tri_task_id} into waiting_queue")
                        self.tag[tri_task_id] = False
                        self.waiting_queue.put(tri_task_id)
                        # 暂时没用到
                        if self.stage=="pre_analysis":
                            self.program[task_id].next.append(self.program[tri_task_id])
                
                # 暂时没用到
                if self.stage == "post_analysis":
                    assert self.tasks[tri_task_id].feat_num+self.tasks[tri_task_id].para_num-para_len-feat_len >= 0
                    if self.tasks[tri_task_id].feat_num+self.tasks[tri_task_id].para_num-para_len-feat_len == 1:
                        # 关键路径
                        self.program[tri_task_id].waitinglast = True

            # block间指令更新
            else:
                para_len = len(self.tasks[tri_task_id].para)
                feat_len = len(self.tasks[tri_task_id].feat)
                # block外的指令不会放入waiting_queue，但需要记录就绪时间
                if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                    self.program[tri_task_id].start_time = self.env.now

    def task_update(self, inst_index):
        logger.debug(f"updating task{inst_index}")
        # 将指令的index转换成core内部task id
        self.inst_counter += 1
        task_id = self.index2taskid[inst_index]

        logger.debug(f"current block_ptr is {self.block_ptr}")
        logger.debug(f"task block_ptr is {task_id//self.block_size}")

        # 记录每层开始时间
        layer_id = self.program[task_id].layer_id
        if self.arch.layer_start[layer_id] == -1:
            self.arch.layer_start[layer_id] = self.env.now

        # 记录每层结束时间
        self.arch.layer_end[layer_id] = max(self.arch.layer_end[layer_id], self.env.now)

        self.block_counter += 1
        logger.debug(f"PE{self.id} self.counter += 1, [{self.block_counter}/{self.block_size}]")
        if self.block_counter == self.block_size:
            self.task_block_update()

        # 处理WRITE指令
        if self.program[task_id].inst_type == TaskType.WRITE:
            # WRITE指令存在触发关系
            if len(self.program[task_id].trigger_index) != 0:
                for id in range(len(self.program[task_id].trigger_index)):

                    tri_core_id = self.program[task_id].trigger_core_id[id]
                    pur_sche = self.cores[tri_core_id].scheduler if tri_core_id != self.id else self
                    tri_task_id = pur_sche.index2taskid[self.program[task_id].trigger_index[id]]

                    logger.debug(f"task{inst_index}[core{self.id}] triggered task{self.program[task_id].trigger_index[id]}[core{tri_core_id}]")

                    # WRITE只会触发READ，操作数占位即可
                    match self.program[task_id].data_type:
                        case DataType.PARA:
                            pur_sche.tasks[tri_task_id].para.append(Data())
                        case DataType.FEAT:
                            pur_sche.tasks[tri_task_id].feat.append(Data())
                    
                    # 维护跨核心更新的record
                    pur_sche.program[tri_task_id].record.mulins.append(self.env.now)
                    
                    # 不在当前block没有影响
                    if tri_task_id // pur_sche.block_size != pur_sche.block_ptr:
                        continue
                    
                    # block内指令更新
                    logger.debug(f"update triggered instruction...")
                    para_len = len(pur_sche.tasks[tri_task_id].para)
                    feat_len = len(pur_sche.tasks[tri_task_id].feat)
                    logger.debug(f"para:{para_len}/{pur_sche.tasks[tri_task_id].para_num} + feat:{feat_len}/{pur_sche.tasks[tri_task_id].feat_num}")
                    
                    if feat_len == pur_sche.tasks[tri_task_id].feat_num and para_len == pur_sche.tasks[tri_task_id].para_num:
                        if pur_sche.tag[tri_task_id]:
                            # 在当前block以数据的形式放入data_in
                            logger.debug(f"PE{tri_core_id} put {tri_task_id} into data_in")
                            pur_sche.tag[tri_task_id] = False

                            if tri_core_id != self.id:
                                # 为了支持packet routing
                                ins = self.program[task_id]
                                ins.index = self.program[task_id].trigger_index[id]
                                self.cores[tri_core_id].data_in.put(Message(ins=self.program[task_id], data=Data(index=self.program[task_id].trigger_index[id], tensor_slice=self.program[task_id].tensor_slice), dst=tri_core_id, src=self.id))
                            else:
                                self.data_in.put(Message(ins=self.program[task_id], data=Data(index=self.program[task_id].trigger_index[id], tensor_slice=self.program[task_id].tensor_slice), dst=tri_core_id, src=self.id))
                            # logger.debug(f"PE{pur_sche.id} insert {tri_task_id} into waiting_queue")
                            # pur_sche.tag[tri_task_id] = False
                            # pur_sche.waiting_queue.put(tri_task_id)

        # 处理其他指令
        else:
            for idx in range(len(self.program[task_id].trigger_index)):
                logger.debug(f"task{inst_index} triggered {self.program[task_id].trigger_index[idx]}")
                tri_task_id = self.index2taskid[self.program[task_id].trigger_index[idx]]
                # 记录触发指令的一个触发时间
                self.program[tri_task_id].record.mulins.append(self.env.now)

                # 计算flops时需要slice
                match self.program[task_id].data_type:
                    case DataType.PARA:
                        self.tasks[tri_task_id].para.append(Data(tensor_slice=self.program[task_id].tensor_slice))
                    case DataType.FEAT:
                        self.tasks[tri_task_id].feat.append(Data(tensor_slice=self.program[task_id].tensor_slice))

                # 更新被触发的指令
                logger.debug(f"triggered block_ptr is {tri_task_id//self.block_size}")
                # block内指令更新
                if tri_task_id // self.block_size == self.block_ptr:
                    logger.debug(f"update triggered instruction...")
                    para_len = len(self.tasks[tri_task_id].para)
                    feat_len = len(self.tasks[tri_task_id].feat)
                    logger.debug(f"para:{para_len}/{self.tasks[tri_task_id].para_num} + feat:{feat_len}/{self.tasks[tri_task_id].feat_num}")
                    
                    # 判断操作数是否到齐
                    if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                        if self.tag[tri_task_id]:
                            logger.debug(f"PE{self.id} insert {tri_task_id} into waiting_queue")
                            self.tag[tri_task_id] = False
                            self.waiting_queue.put(tri_task_id)
                            # 暂时没用到
                            if self.stage=="pre_analysis":
                                self.program[task_id].next.append(self.program[tri_task_id])
                    
                    # 暂时没用到
                    if self.stage == "post_analysis":
                        assert self.tasks[tri_task_id].feat_num+self.tasks[tri_task_id].para_num-para_len-feat_len >= 0
                        if self.tasks[tri_task_id].feat_num+self.tasks[tri_task_id].para_num-para_len-feat_len == 1:
                            # 关键路径
                            self.program[tri_task_id].waitinglast = True

                # block间指令更新
                else:
                    para_len = len(self.tasks[tri_task_id].para)
                    feat_len = len(self.tasks[tri_task_id].feat)
                    # block外的指令不会放入waiting_queue，但需要记录就绪时间
                    if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                        self.program[tri_task_id].start_time = self.env.now
    
    # 原来的二合一update，已弃用
    def update(self, data):
        self.inst_counter += 1
        task_id = self.index2taskid[data.index]
        logger.debug(f"updating {data.index}")
        logger.debug(f"{task_id} // {self.block_size} == {self.block_ptr}")

        # 记录每层开始时间
        layer_id = self.program[task_id].layer_id
        if self.arch.layer_start[layer_id] == -1:
            self.arch.layer_start[layer_id] = self.env.now

        # 记录每层结束时间
        self.arch.layer_end[layer_id] = max(self.arch.layer_end[layer_id], self.env.now)
        
        if self.program[task_id].inst_type != TaskType.RECV:
            assert task_id // self.block_size == self.block_ptr

        if task_id // self.block_size == self.block_ptr:
            # print(f"PE{self.id} self.counter += 1")
            logger.debug(f"PE{self.id} self.counter += 1")
            self.block_counter += 1
            logger.debug(f"PE{self.id} self.counter is {self.block_counter}/{self.block_size}")

        if self.block_counter == self.block_size:
            self.task_block_update()

        if self.program[task_id].inst_type == TaskType.WRITE:
            # print(f"{len(self.program[task_id].trigger_index)} - {len(self.program[task_id].trigger_core_id)}")
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

        # 这里可以构建连线表示依赖关系
        for idx in range(len(self.program[task_id].trigger_index)):
            # 对WRITE触发的指令有问题，可能不在同一core，改到上面单独处理
            tri_task_id = self.index2taskid[self.program[task_id].trigger_index[idx]]
            # 记录触发指令的一个触发时间
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
                    # 非计算指令最多需要1个操作数，并且无需同步
                    if feat_len == self.tasks[tri_task_id].feat_num:
                        if self.tag[tri_task_id]:
                            logger.debug(f"PE{self.id} insert {tri_task_id} into waiting_queue")
                            self.tag[tri_task_id] = False
                            self.waiting_queue.put(tri_task_id)

                if self.stage == "post_analysis":
                    assert self.tasks[tri_task_id].feat_num+self.tasks[tri_task_id].para_num-para_len-feat_len >= 0
                    if self.tasks[tri_task_id].feat_num+self.tasks[tri_task_id].para_num-para_len-feat_len == 1:
                        # 关键路径
                        self.program[tri_task_id].waitinglast = True
            
            # 指令之间的依赖考虑的是block内的，没有考虑block间的
            else:
                # 同上
                para_len = len(self.tasks[tri_task_id].para)
                feat_len = len(self.tasks[tri_task_id].feat)
                if self.program[tri_task_id].inst_type in self.comp_inst:
                    if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                        self.program[tri_task_id].start_time = self.env.now
                else:
                    # 非计算指令最多需要1个操作数，并且无需同步
                    if feat_len == self.tasks[tri_task_id].feat_num:
                        self.program[tri_task_id].start_time = self.env.now

    def schedule(self):
        if self.waiting_queue.empty():
            # print(f"waiting queue is empty")
            return None
        else:
            logger.debug(f"PE{self.id} block_counter: {self.block_counter}/{self.block_size}")
            if self.block_counter == self.block_size:
                self.task_block_update()

            task_ready = []
            while not self.waiting_queue.empty():
                task_id = self.waiting_queue.get()
                if task_id == len(self.program) - 1:
                    self.finish = True
                task_ready.append(self.tasks[task_id])

            return task_ready

def print_event_queue(env):
    print("Remaining keys() are:")
    print("="*40)
    
    for event in env._queue:
        print(event)



class Core:
    def __init__(self, env, config: CoreConfig, program: List[Instruction], id: int, arch, link1, link2, stage=None):
        self.env = env
        self.type = config.type
        self.program = program
        self.id = id
        self.spm_manager = SPMManager(env, self.id, config.spm)
        self.flow_out = []
        self.flow_in = []
        self.stage = stage

        self.bound_with_router(link2, link1)

        # 1.in a block, 2.close to running instruction
        self.waitinglist = []

        self.recv_queue = []

        self.end_time = 0

        # self.scheduler = GraphScheduler(self.program, self.spm_manager)
        self.scheduler = TableScheduler(self.program, self.spm_manager, config.blk_size, self.id, self.env, arch, self.data_in, stage)

        self.lsu_bandwidth = config.lsu.width
        self.tpu_flops = config.tpu.flops
        self.lsu = MonitoredResource(env=env, capacity=4)
        self.tpu = MonitoredResource(env=env, capacity=1)

        self.data_ready = {}
        
        
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

    def tpu_recover(self, times):
        self.tpu_flops *= times

    def receive_data(self, msg):
        logger.debug(f"in function receive_data()")
        task_id = self.scheduler.index2taskid[msg.data.index]
        # 接收到的数据在block内才进行更新，否则放回data_in
        if task_id in range(self.scheduler.start, self.scheduler.end):
            slice = Slice(tensor_slice=msg.data.tensor_slice)

            self.program[task_id].record.exe_start_time.append(self.env.now)
            # 分配接收数据空间
            yield self.env.process(self.spm_manager.allocate("recv"+str(msg.data.index), slice.size()))
            # RECV完成
            self.program[task_id].record.exe_end_time.append(self.env.now)
            self.scheduler.data_update(msg.data)
        else:
            logger.debug(f"PE{self.id} insert data{msg.data.index} into recv_queue")
            
            msg.data.index = self.scheduler.index2taskid[msg.data.index]
            heapq.heappush(self.recv_queue, msg)
            # yield self.env.process(self.data_in.insert(msg))

    def core_run(self):
        # running 事件列表
        self.running_event = []
        self.running_send = []
        self.event2task = {}

        while True:
            while self.recv_queue:
                top = self.recv_queue[0]
                if top.data.index in range(self.scheduler.start, self.scheduler.end):      
                    msg = heapq.heappop(self.recv_queue)

                    # task_id转换回index
                    msg.data.index = self.scheduler.taskid2index[msg.ins.index]
                    logger.debug(f"PE{self.id} pop data{msg.ins.index} from recv_queue")

                    yield self.env.process(self.receive_data(msg))
                else:
                    break

            task_ready = self.scheduler.schedule()
            if task_ready:
                for task in task_ready:

                    # if task.index == 3969:
                    #     print(f"task3969 is triggered")

                    instruction = self.program[self.scheduler.index2taskid[task.index]]
                    task_event = self.env.process(task.run(self, instruction))
                    
                    logger.info(f"Time {self.env.now:.2f}: PE{self.id} add a {type(task)} task(id:{task.index}, layer:{self.scheduler.program[self.scheduler.index2taskid[task.index]].layer_id}) into running queue.")
                    self.running_event.append(task_event)
                    self.event2task[task_event] = task

            if self.id == 12:
                logger.debug(f"Before AnyOf yield")
                logger.debug(f"PE{self.id}'s running_queue is {self.running_event}")

            with self.data_in.get() as msg_arrive:
                if self.id == 12:
                    logger.debug(f"PE{self.id} is yielding AnyOf")

                # 其他core更新当前core的任务时，控制权不在当前core
                # running_event无法及时更新（可能为空），导致只yield了msg，没数据来就会卡死
                # 把WRITE触发的READ数据包装到data_in里，而非修改waiting_queue可以解决
                result = yield simpy.events.AnyOf(self.env, self.running_event + [msg_arrive])
                if self.id == 12:
                    logger.debug(f"PE{self.id} finish yielding AnyOf")

                logger.info(f"Time {self.env.now:.2f}: PE{self.id}'s result is {result}")
            
                # 从NoC接收数据
                if msg_arrive.triggered:
                    msg = msg_arrive.value

                    # if msg.ins.index == 3969:
                    #     print(f"task3969 is triggered")

                    # 记录数据到达的时间（可能未及时接收）
                    task_id = self.scheduler.index2taskid[msg.ins.index]

                    # RECV 或者包装后的 READ
                    self.program[task_id].record.ready_run_time.append(self.env.now)
                    self.program[task_id].record.pe_id = self.id

                    # 暂时没用到
                    if self.stage == "post_analysis":
                        src = msg.src
                        inst = ind2ins[src][msg.ins.index]
                        assert inst in self.arch.cores[src].running_send
                        self.arch.cores[src].running_send.remove(inst)

                    # 记录数据接收信息
                    if cfg.flow and self.env.now >= cfg.simstart and self.env.now <= cfg.simend:
                        self.flow_in.append((msg.ins.index, self.program[self.scheduler.index2taskid[msg.ins.index]].inst_type, "recv", self.env.now))

                    logger.info(f"Time {self.env.now:.2f}: PE{self.id} receive data{msg.ins.index}")
                    # logger.debug(f"received data is {msg.ins}")

                    logger.debug(f"function call: receive_data()")
                    yield self.env.process(self.receive_data(msg))

                    if self.data_in.len() > 0:
                        if self.scheduler.index2taskid[self.data_in.store.items[0].data.index] not in range(self.scheduler.start, self.scheduler.end):
                            break

                    # 处理其他已经到达且在当前block的数据
                    while self.data_in.len() > 0:
                        msg = yield self.data_in.get()

                        logger.info(f"Time {self.env.now:.2f}: PE{self.id} receive data{msg.ins.index}")
                        logger.info(f"received data is {msg.ins}")
                        
                        logger.debug(f"function call: receive_data()")
                        yield self.env.process(self.receive_data(msg))

                #注意，单纯put的msg一定是send，这里还有msg是其它的
                for event in self.running_event:
                    if event.triggered:
                        # updated = True
                        task = self.event2task[event]

                        # print(f"PE{self.id} free: {task.input_size()}, [{self.spm_manager.capacity}/{self.spm_manager.size}]")

                        logger.info(f"Time {self.env.now:.2f}: PE{self.id} finish processing {type(self.event2task[event])} task(id:{self.event2task[event].index}).")
                        # 这个也不可能是RECV，我需要找到其中的SEND
                        # print(self.program[self.scheduler.index2taskid[self.event2task[event].index]].inst_type)
                        if cfg.flow and self.env.now >= cfg.simstart and self.env.now <= cfg.simend:
                            if self.program[self.scheduler.index2taskid[self.event2task[event].index]].inst_type == TaskType.SEND:
                                self.flow_out.append((self.event2task[event].index, self.program[self.scheduler.index2taskid[self.event2task[event].index]].inst_type,"send",self.env.now))
                        # 所有更新都经过update
                        self.scheduler.task_update(self.event2task[event].index)
                        
                        # 这个是仿真时间的瓶颈,大致用这个估算规模
                        if self.stage == "post_analysis":
                            if task.inst.inst_type == TaskType.SEND:
                                self.running_send.append(task.inst)

                        self.running_event.remove(event)

            if self.scheduler.inst_counter == len(self.program):
                self.end_time = self.env.now
                print(f"Time {self.env.now:.2f}: PE{self.id} finished processing all of its instructions.")


#maybe concurrent 
    def run_task(self, task):
        self.spm_manager.allocate(task.string+str(task.index),task)
        yield self.env.process(task.run(self))
        self.spm_manager.free(task.string+str(task.index),task)