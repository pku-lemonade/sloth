import heapq
import simpy
import logging
from queue import Queue
from src.common import MonitoredResource,cfg,cores_deps
from src.arch_config import CoreConfig, ScratchpadConfig
from src.noc_new import Link, Router
from src.sim_type import *
from typing import List

logger = logging.getLogger("PE")

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
    def __init__(self, program, spm, block_size, id):
        self.program = program
        self.spm = spm

        self.block_size = block_size
        self.block_ptr = -1
        self.block_counter = 0
        
        # 当前block的id区间 [start, end)
        self.start = 0
        self.end = 0

        self.id = id
        self.tag = [True for _ in range(len(self.program))]
        self.finish = False
        self.inst_counter = 0

        self.tasks = []

        self.index2taskid = {}
        self.taskid2index = {}

        self.waiting_queue = Queue()

        self.comp_inst = [TaskType.CONV, TaskType.POOL, TaskType.ELEM, TaskType.FC, TaskType.GCONV, TaskType.PTP, TaskType.TRANS]

        for id, inst in enumerate(self.program):
            self.index2taskid[inst.index] = id
            self.taskid2index[id] = inst.index
            match inst.inst_type:
                case TaskType.STAY:
                    self.tasks.append(Stay(index=inst.index, tensor_slice=inst.tensor_slice))
                case TaskType.RECV:
                    self.tasks.append(Recv(index=inst.index, tensor_slice=inst.tensor_slice))
                case TaskType.READ:
                    self.tasks.append(Read(index=inst.index, tensor_slice=inst.tensor_slice))
                case TaskType.WRITE:
                    self.tasks.append(Write(index=inst.index, tensor_slice=inst.tensor_slice))
                case TaskType.SEND:
                    self.tasks.append(Send(index=inst.index, tensor_slice=inst.tensor_slice, dst=inst.position))
                case TaskType.CONV:
                    self.tasks.append(Conv(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, layer_id=inst.layer_id))
                case TaskType.POOL:
                    self.tasks.append(Pool(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, layer_id=inst.layer_id))
                case TaskType.ELEM:
                    self.tasks.append(Elem(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, layer_id=inst.layer_id))
                case TaskType.FC:
                    self.tasks.append(FC(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, layer_id=inst.layer_id))
                case TaskType.GCONV:
                    self.tasks.append(GConv(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, layer_id=inst.layer_id, group_num=inst.group_num))
                case TaskType.PTP:
                    self.tasks.append(PTP(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, layer_id=inst.layer_id))
                case TaskType.TRANS:
                    self.tasks.append(Trans(index=inst.index, feat_num=inst.feat_num, para_num=inst.para_num, tensor_slice=inst.tensor_slice, layer_id=inst.layer_id))

        self.task_block_update()

    def task_block_update(self):
        # print(f"PE{self.id} is task_blk_updating")
        logger.debug(f"PE{self.id} is task_block_updating")
        self.block_counter = 0
        self.block_ptr += 1
        self.start = self.block_ptr * self.block_size
        self.end = min((self.block_ptr + 1) * self.block_size, len(self.program))
        for id, inst in enumerate(self.program[self.start:self.end], start=self.start):
            # print("-"*30)
            # print(f"inst_id is {id}, type is {inst.inst_type}")
            logger.debug("-"*30)
            logger.debug(f"inst_id is {id}, index is {inst.index}, type is {inst.inst_type}")
            match inst.inst_type:
                case TaskType.READ:
                    # print(f"insert {id} into waiting queue")
                    if self.tag[id]:
                        self.tag[id] = False
                        logger.debug(f"insert {id} into waiting queue")
                        self.waiting_queue.put(id)
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
                case TaskType.RECV:
                    if self.tasks[id].feat:
                        logger.debug(f"self.counter += 1")
                        self.block_counter += 1
                case TaskType.STAY:
                    if self.tag[id]:
                        self.tag[id] = False
                        self.waiting_queue.put(id)
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
        
    def update(self, data):
        self.inst_counter += 1
        task_id = self.index2taskid[data.index]
        #print(data.index)
        #print(self.program[task_id].inst_type)
        # print(f"updating {data.index}")
        # print(f"{task_id} // {self.block_size} == {self.block_ptr}")
        logger.debug(f"updating {data.index}")
        logger.debug(f"{task_id} // {self.block_size} == {self.block_ptr}")
        if task_id // self.block_size == self.block_ptr:
            # print(f"PE{self.id} self.counter += 1")
            logger.debug(f"PE{self.id} self.counter += 1")
            self.block_counter += 1
            logger.debug(f"PE{self.id} self.counter is {self.block_counter}/{self.block_size}")

        if self.block_counter == self.block_size:
            self.task_block_update()

        if self.program[task_id].inst_type == TaskType.WRITE:
            return
        
        if self.program[task_id].inst_type == TaskType.RECV:
            self.tasks[task_id].feat.append(data)

        #这里可以构建连线表示依赖关系
        for idx in range(len(self.program[task_id].trigger_index)):
            tri_task_id = self.index2taskid[self.program[task_id].trigger_index[idx]]
            match self.program[task_id].data_type:
                case DataType.PARA:
                    self.tasks[tri_task_id].para.append(data)
                case DataType.FEAT:
                    self.tasks[tri_task_id].feat.append(data)

            # print(f"{data.index} has triggered {self.trigger[task_id][idx].index}")
            # print(f"{tri_task_id} // {self.block_size} == {self.block_ptr}")
            logger.debug(f"{data.index} has triggered {self.program[task_id].trigger_index[idx]}")
            logger.debug(f"{tri_task_id} // {self.block_size} == {self.block_ptr}")
            if tri_task_id // self.block_size == self.block_ptr:
                # logger.debug("inside")
                # para = 1 if self.program[tri_task_id].inst_type in self.comp_inst and self.tasks[tri_task_id].para else 0
                # feat = 1 if self.tasks[tri_task_id].feat else 0
                # logger.debug(f"para:{para} + feat:{feat}")
                # if para + feat == self.tasks[tri_task_id].num_operands:
                #     if self.tag[tri_task_id]:
                #         logger.debug(f"PE{self.id} insert {tri_task_id} into waiting_queue")
                #         self.tag[tri_task_id] = False
                #         self.waiting_queue.put(tri_task_id)

                para_len = len(self.tasks[tri_task_id].para)
                feat_len = len(self.tasks[tri_task_id].feat)
                logger.debug("inside")
                logger.debug(f"para:{para_len}/{self.tasks[tri_task_id].para_num} + feat:{feat_len}/{self.tasks[tri_task_id].feat_num}")
                if self.program[tri_task_id].inst_type in self.comp_inst:
                    if feat_len == self.tasks[tri_task_id].feat_num and para_len == self.tasks[tri_task_id].para_num:
                        if self.tag[tri_task_id]:
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
    def __init__(self, env, config: CoreConfig, program: List[Instruction], id: int):
        self.env = env
        self.type = config.type
        self.program = program
        self.id = id
        self.spm_manager = SPMManager(env, self.id, config.spm)
        self.flow_out=[]
        self.flow_in=[]

        self.recv_queue = []

        self.end_time = 0

        # self.scheduler = GraphScheduler(self.program, self.spm_manager)
        self.scheduler = TableScheduler(self.program, self.spm_manager, config.blk_size, self.id)

        self.lsu_bandwidth = config.lsu.width
        self.tpu_flops = config.tpu.flops
        self.lsu = MonitoredResource(env=env, capacity=4)
        self.tpu = MonitoredResource(env=env, capacity=1)

        self.data_ready = {}
        
        self.data_in, self.data_out = None, None
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

    def core_run(self):
        # running 事件列表
        self.running_event = []
        self.event2task = {}

        while True:
            while self.recv_queue:
                top = self.recv_queue[0]
                if top.index in range(self.scheduler.start, self.scheduler.end):      
                    data = heapq.heappop(self.recv_queue)
                    # task_id转换回index
                    data.index = self.scheduler.taskid2index[data.index]
                    logger.debug(f"PE{self.id} pop data{data.index} from recv_queue")
                    
                    slice = Slice(tensor_slice=data.tensor_slice)
                    yield self.env.process(self.spm_manager.allocate("recv"+str(data.index), slice.size()))
                    self.scheduler.update(data)
                else:
                    break

            task_ready = self.scheduler.schedule()
            if task_ready:
                for task in task_ready:

                    # 分配计算结果存储空间
                    # self.spm_manager.allocate(task.string+str(task.index), task.output_size())
                    # print(f"PE{self.id} allocate: {task.output_size()}, [{self.spm_manager.capacity}/{self.spm_manager.size}]")

                    task_event = self.env.process(task.run(self))
                    
                    logger.info(f"Time {self.env.now:.2f}: PE{self.id} add a {type(task)} task(id:{task.index}, layer:{self.scheduler.program[self.scheduler.index2taskid[task.index]].layer_id}) into running queue.")
                    self.running_event.append(task_event)
                    self.event2task[task_event] = task
            logger.info(f"Time {self.env.now:.2f}: Before trigger::PE{self.id} data_len is: {self.data_in.len()}")

            with self.data_in.get() as msg_arrive:
                result = yield simpy.events.AnyOf(self.env, self.running_event + [msg_arrive])
                logger.info(f"Time {self.env.now:.2f}: PE{self.id}'s result is {result}")
                # updated = False

                if msg_arrive.triggered:
                    # updated = True
                    msg = msg_arrive.value
                    #一定是RECV
                    assert self.program[self.scheduler.index2taskid[msg.data.index]].inst_type == TaskType.RECV
                    #TODO:需要加入什么？
                    if cfg.flow and self.env.now >= cfg.simstart and self.env.now <= cfg.simend:
                        self.flow_in.append((msg.data.index, self.program[self.scheduler.index2taskid[msg.data.index]].inst_type,"recv",self.env.now))
                    logger.info(f"Time {self.env.now:.2f}: triggered::PE{self.id} receive data{msg.data.index}")
                    logger.debug(f"received data is {msg.data}")
                    logger.debug(f"data_queue_len is {self.data_in.len()}")

                    # 分配接收数据空间
                    if self.scheduler.index2taskid[msg.data.index] in range(self.scheduler.start, self.scheduler.end):
                        slice = Slice(tensor_slice=msg.data.tensor_slice)
                        yield self.env.process(self.spm_manager.allocate("recv"+str(msg.data.index), slice.size()))
                        # 接收到的数据在block内才进行更新，否则放入recv_queue
                        self.scheduler.update(msg.data)
                    else:
                        logger.debug(f"PE{self.id} insert data{msg.data.index} into recv_queue")
                        # index转换成PE内id
                        msg.data.index = self.scheduler.index2taskid[msg.data.index]
                        heapq.heappush(self.recv_queue, msg.data)
                    
                    # print(f"PE{self.id} allocate: {slice.size()}, [{self.spm_manager.capacity}/{self.spm_manager.size}]")

                    logger.info(f"Time {self.env.now:.2f}: After trigger::PE{self.id} data_len is: {self.data_in.len()}")

                    while self.data_in.len() > 0:
                        # assert 0
                        msg = yield self.data_in.get()
                        logger.info(f"Time {self.env.now:.2f}: PE{self.id} receive data{msg.data.index}")
                        logger.info(f"received data is {msg.data}")
                        logger.info(f"data_queue_len is {self.data_in.len()}")
                        
                        # 分配接收数据空间
                        if self.scheduler.index2taskid[msg.data.index] in range(self.scheduler.start, self.scheduler.end):
                            slice = Slice(tensor_slice=msg.data.tensor_slice)
                            yield self.env.process(self.spm_manager.allocate("recv"+str(msg.data.index), slice.size()))
                            # 接收到的数据在block内才进行更新，否则放入recv_queue
                            self.scheduler.update(msg.data)
                        else:
                            logger.debug(f"PE{self.id} insert data{msg.data.index} into recv_queue")
                            # index转换成PE内id
                            msg.data.index = self.scheduler.index2taskid[msg.data.index]
                            heapq.heappush(self.recv_queue, msg.data)
                        
                        # print(f"PE{self.id} allocate: {slice.size()}, [{self.spm_manager.capacity}/{self.spm_manager.size}]")

                #注意，单纯put的msg一定是send，这里还有msg是其它的
                for event in self.running_event:
                    if event.triggered:
                        # updated = True
                        task=self.event2task[event]

                        # 执行完成，释放输入数据空间
                        self.env.process(self.spm_manager.free(task.string+str(task.index),task.input_size()))
                        # print(f"PE{self.id} free: {task.input_size()}, [{self.spm_manager.capacity}/{self.spm_manager.size}]")

                        logger.info(f"Time {self.env.now:.2f}: PE{self.id} finish processing {type(self.event2task[event])} task(id:{self.event2task[event].index}).")
                        #这个也不可能是RECV，我需要找到其中的SEND
                        #print(self.program[self.scheduler.index2taskid[self.event2task[event].index]].inst_type)
                        if cfg.flow and self.env.now >= cfg.simstart and self.env.now <= cfg.simend:
                            if self.program[self.scheduler.index2taskid[self.event2task[event].index]].inst_type == TaskType.SEND:
                                self.flow_out.append((self.event2task[event].index, self.program[self.scheduler.index2taskid[self.event2task[event].index]].inst_type,"send",self.env.now))
                        self.scheduler.update(Data(index=self.event2task[event].index, tensor_slice=self.event2task[event].tensor_slice))
                        self.running_event.remove(event)

            if self.scheduler.inst_counter == len(self.program):
                self.end_time = self.env.now
                print(f"Time {self.env.now:.2f}: PE{self.id} finished processing all of its instructions.")


#maybe concurrent 
    def run_task(self, task):
        self.spm_manager.allocate(task.string+str(task.index),task)
        yield self.env.process(task.run(self))
        self.spm_manager.free(task.string+str(task.index),task)