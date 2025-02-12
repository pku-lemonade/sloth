import simpy
import logging
from queue import Queue
from src.arch_config import CoreConfig, ScratchpadConfig
from src.noc import Link, Router
from src.sim_type import Instruction, Read, Write, Conv, Pool, FC, Send, Recv, Data, Stay, TaskType, OperationType, DataType
from typing import List

logger = logging.getLogger("PE")

class SPMManager:
    def __init__(self, config: ScratchpadConfig):
        self.size = config.size
        self.delay = config.delay

        self.capacity = self.size

    def allocate(self, task):
        if task.size > self.capacity:
            raise ValueError("Not enough space in SPM")
        self.capacity -= task.size

    def free(self, task):
        self.capacity += task.size

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
        self.id = id
        self.tag = [True for _ in range(len(self.program))]
        self.finish = False

        self.tasks = []

        self.index2taskid = {}
        self.taskid2index = {}

        self.waiting_queue = Queue()

        for id, inst in enumerate(self.program):
            self.index2taskid[inst.index] = id
            self.taskid2index[id] = inst.index
            match inst.inst_type:
                case TaskType.STAY:
                    self.tasks.append(Stay(index=inst.index, size=inst.size))
                case TaskType.RECV:
                    self.tasks.append(Recv(index=inst.index, size=inst.size))
                case TaskType.READ:
                    self.tasks.append(Read(index=inst.index, size=inst.size))
                case TaskType.WRITE:
                    self.tasks.append(Write(index=inst.index, size=inst.size))
                case TaskType.SEND:
                    self.tasks.append(Send(index=inst.index, size=inst.size, dst=inst.position))
                case TaskType.COMP:
                    match inst.operation:
                        case OperationType.CONV:
                            self.tasks.append(Conv(index=inst.index, flops=inst.size, layer_id=inst.layer_id))
                        case OperationType.POOL:
                            self.tasks.append(Pool(index=inst.index, flops=inst.size, layer_id=inst.layer_id))
                        case OperationType.FC:
                            self.tasks.append(FC(index=inst.index, flops=inst.size, layer_id=inst.layer_id))

        self.task_block_update()

    def task_block_update(self):
        # print(f"PE{self.id} is task_blk_updating")
        logger.debug(f"PE{self.id} is task_block_updating")
        self.block_counter = 0
        self.block_ptr += 1
        start = self.block_ptr * self.block_size
        end = min((self.block_ptr + 1) * self.block_size, len(self.program))
        for id, inst in enumerate(self.program[start:end], start=start):
            # print("-"*30)
            # print(f"inst_id is {id}, type is {inst.inst_type}")
            logger.debug("-"*30)
            logger.debug(f"inst_id is {id}, type is {inst.inst_type}")
            match inst.inst_type:
                case TaskType.READ:
                    # print(f"insert {id} into waiting queue")
                    if self.tag[id]:
                        self.tag[id] = False
                        logger.debug(f"insert {id} into waiting queue")
                        self.waiting_queue.put(id)
                case TaskType.COMP:
                    para = 1 if self.tasks[id].para else 0
                    feat = 1 if self.tasks[id].feat else 0
                    # print(f"para is {para}, feat is {feat}")
                    logger.debug(f"para is {para}, feat is {feat}")
                    if para + feat == self.tasks[id].num_operands:
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

        if self.block_counter == self.block_size:
            self.task_block_update()
        
    def update(self, data):
        task_id = self.index2taskid[data.index]

        # print(f"updating {data.index}")
        # print(f"{task_id} // {self.block_size} == {self.block_ptr}")
        logger.debug(f"updating {data.index}")
        logger.debug(f"{task_id} // {self.block_size} == {self.block_ptr}")
        if task_id // self.block_size == self.block_ptr:
            # print(f"PE{self.id} self.counter += 1")
            logger.debug(f"PE{self.id} self.counter += 1")
            self.block_counter += 1
            logger.debug(f"PE{self.id} self.counter is {self.block_counter}/10")

        if self.block_counter == self.block_size:
            self.task_block_update()

        if self.program[task_id].inst_type == TaskType.WRITE:
            return
        
        if self.program[task_id].inst_type == TaskType.RECV:
            self.tasks[task_id].feat.append(data)

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
                # print("inside")
                logger.debug("inside")
                para = 1 if self.program[tri_task_id].inst_type == TaskType.COMP and self.tasks[tri_task_id].para else 0
                feat = 1 if self.tasks[tri_task_id].feat else 0
                # print(f"para:{para} + feat:{feat}")
                logger.debug(f"para:{para} + feat:{feat}")
                if para + feat == self.tasks[tri_task_id].num_operands:
                    # print(f"PE{self.id} insert {tri_task_id} into waiting_queue")
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
    def __init__(self, env, config: CoreConfig, program: List[List[Instruction]], id: int):
        self.env = env
        self.type = config.type
        self.program = program
        self.id = id
        self.spm_manager = SPMManager(config.spm)

        # self.scheduler = GraphScheduler(self.program, self.spm_manager)
        self.scheduler = TableScheduler(self.program, self.spm_manager, config.blk_size, self.id)

        self.lsu_bandwidth = config.lsu.width
        self.tpu_flops = config.tpu.flops
        self.lsu = simpy.Resource(env, capacity=2)
        self.tpu = simpy.Resource(env, capacity=1)
        self.data_queue = simpy.Store(self.env)

        self.link = None
        self.router = None

        self.env.process(self.run())

    def data_len(self):
        return len(self.data_queue.items)

    def connect(self, link: Link, router: Router):
        self.link = link
        self.router = router

    def run(self):
        # running 事件列表
        self.running_event = []
        self.event2task = {}

        task_initial = self.scheduler.schedule()
        for task in task_initial:
            task_event = self.env.process(task.run(self))
            logger.info(f"Time {self.env.now:.2f}: PE{self.id} add a {type(task)} task(id:{task.index}) into running queue.")
            self.running_event.append(task_event)
            self.event2task[task_event] = task
        
        while True:
            msg_arrive = []
            if self.data_len() > 0:
                msg_arrive.append(self.data_queue.get())
            elif (not self.running_event) and (not self.scheduler.finish):
                msg = yield self.data_queue.get()
                self.scheduler.update(msg.data)

            msg_or_task = self.env.any_of(self.running_event + msg_arrive)
            ret = yield msg_or_task

            for event in ret.keys():
                if event in msg_arrive:
                    msg = ret[event]
                    logger.info(f"Time {self.env.now:.2f}: PE{self.id} receive data{msg.data.index}")
                    logger.debug(f"received data is {msg.data}")
                    logger.debug(f"data_queue_len is {self.data_len()}")
                    self.scheduler.update(msg.data)
                    
                if event in self.running_event:
                    logger.info(f"Time {self.env.now:.2f}: PE{self.id} finish processing {type(self.event2task[event])} task(id:{self.event2task[event].index}).")
                    self.scheduler.update(Data(index=self.event2task[event].index))
                    self.running_event.remove(event)

            task_ready = self.scheduler.schedule()
            
            if task_ready:
                for task in task_ready:
                    task_event = self.env.process(task.run(self))
                    logger.info(f"Time {self.env.now:.2f}: PE{self.id} add a {type(task)} task(id:{task.index}) into running queue.")
                    self.running_event.append(task_event)
                    self.event2task[task_event] = task
            
            if (not self.running_event) and self.data_len() == 0 and self.scheduler.finish:
                break

    def run_task(self, task):
        self.spm_manager.allocate(task)
        yield self.env.process(task.run(self))
        self.spm_manager.free(task)