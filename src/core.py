import simpy
from queue import Queue
from src.config import CoreConfig, ScratchpadConfig
from src.noc import Link, Router
from src.sim_type import Task, Instruction, Read, Write, Conv, Pool, FC, Send, Data
from typing import List

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
    def __init__(self, program, spm, block_size):
        self.program = program
        self.spm = spm
        self.block_size = block_size
        self.block_ptr = -1
        self.block_counter = 0

        self.tasks = []
        self.trigger = [[] for _ in range(len(self.program))]

        self.index2taskid = {}
        self.taskid2index = {}

        self.waiting_queue = Queue()

        for id, inst in enumerate(self.program):
            self.index2taskid[inst.index] = id
            self.taskid2index[id] = inst.index
            match inst.inst_type:
                case "RECV":
                    self.tasks.append(Task(index=-1,size=-1,flops=-1,num_operands=-1))
                case "READ":
                    self.tasks.append(Read(index=inst.index, size=inst.size))
                case "WRITE":
                    self.tasks.append(Write(index=inst.index, size=inst.size))
                case "SEND":
                    self.tasks.append(Send(index=inst.index, size=inst.size, dst=inst.position))
                case "COMP":
                    match inst.operation:
                        case "CONV":
                            self.tasks.append(Conv(flops=inst.size, layer_id=inst.layer_id))
                        case "POOL":
                            self.tasks.append(Pool(flops=inst.size, layer_id=inst.layer_id))
                        case "FC":
                            self.tasks.append(FC(flops=inst.size, layer_id=inst.layer_id))
        
        self.build_trigger()
        self.task_block_update()
        
                    
    def build_trigger(self):
        last_layer = -1
        last_layer_ind = -1

        inputs = []
        outputs = []
        comp_pos = -1
        
        for ind, inst in enumerate(self.program):
            if inst.layer_id != last_layer or ind == len(self.program)-1:
                if ind == len(self.program)-1:
                    inst_start = last_layer_ind
                    inst_end = len(self.program)
                elif last_layer != -1:
                    inst_start = last_layer_ind
                    inst_end = ind
                else:
                    inst_start = 0
                    inst_end = ind

                inputs.clear()
                outputs.clear()
                comp_pos = -1

                for idx, layer_inst in enumerate(self.program[inst_start:inst_end], start=inst_start):
                    match layer_inst.inst_type:
                        case "READ":
                            inputs.append(layer_inst)
                        case "RECV":
                            inputs.append(layer_inst)
                        case "WRITE":
                            print("???")
                            outputs.append(layer_inst)
                        case "SEND":
                            outputs.append(layer_inst)
                        case "COMP":
                            print("!!!")
                            comp_pos = layer_inst
                    
                for input in inputs:
                    self.trigger[self.index2taskid[input.index]].append(comp_pos)

                for output in outputs:
                    print("insert")
                    print(f"{self.index2taskid[comp_pos.index]} -> {self.index2taskid[output.index]}")
                    self.trigger[self.index2taskid[comp_pos.index]].append(output)
                    
                last_layer = inst.layer_id
                last_layer_ind = ind

    def task_block_update(self):
        self.block_counter = 0
        self.block_ptr += 1
        start = self.block_ptr * self.block_size
        end = min((self.block_ptr + 1) * self.block_size, len(self.program))
        for id, inst in enumerate(self.program[start:end], start=start):
            match inst.inst_type:
                case "READ":
                    self.waiting_queue.put(id)
                case "COMP":
                    para = 1 if self.tasks[id].para else 0
                    feat = 1 if self.tasks[id].feat else 0
                    if para + feat == self.tasks[id].num_operands:
                        self.waiting_queue.put(id)
                case "SEND":
                    if self.tasks[id].feat:
                        self.waiting_queue.put(id)
                case "WRITE":
                    if self.tasks[id].feat:
                        self.waiting_queue.put(id)
        
    def update(self, data):
        task_id = self.index2taskid[data.index]
        if self.program[task_id].inst_type == "WRITE":
            return
        
        self.block_counter += 1
        tri_task_id = self.index2taskid[self.trigger[task_id][0].index]
        match self.program[task_id].data_type:
            case "PARA":
                self.tasks[tri_task_id].para.append(data)
            case "FEAT":
                self.tasks[tri_task_id].feat.append(data)

        print(tri_task_id)
        if tri_task_id // self.block_size == self.block_ptr:
            print("inside")
            para = 1 if self.program[tri_task_id].inst_type == "COMP" and self.tasks[tri_task_id].para else 0
            feat = 1 if self.tasks[tri_task_id].feat else 0
            print(f"para:{para} + feat:{feat}")
            if para + feat == self.tasks[tri_task_id].num_operands:
                self.waiting_queue.put(tri_task_id)

    def schedule(self):
        if self.waiting_queue.empty():
            return None
        else:
            self.block_counter += 1
            task_id = self.waiting_queue.get()
            print(f"local task id is:{task_id}")
            self.update(Data(index=self.taskid2index[task_id],dst=-1,size=self.program[task_id].size))

            if self.block_counter == self.block_size:
                self.task_block_update()
            return self.tasks[task_id]


class Core:
    def __init__(self, env, config: CoreConfig, program: List[List[Instruction]], id: int):
        self.env = env
        self.type = config.type
        self.program = program
        self.id = id
        self.spm_manager = SPMManager(config.spm)

        # self.scheduler = GraphScheduler(self.program, self.spm_manager)
        self.scheduler = TableScheduler(self.program, self.spm_manager, config.blk_size)

        # print(f"Building PE{self.id}'s instruction dependency graph.")
        # self.scheduler.build_graph()
        # print("Finished.")

        self.lsu_bandwidth = config.lsu.width
        self.tpu_flops = config.tpu.flops
        self.lsu = simpy.Resource(env, capacity=2)
        self.tpu = simpy.Resource(env, capacity=1)
        self.data_queue = simpy.Store(self.env)

        self.link = None
        self.router = None

        self.env.process(self.run())

    def connect(self, link: Link, router: Router):
        self.link = link
        self.router = router

    def run(self):
        while True:
            if self.router.compute_queue_len > 0:
                data = yield self.data_queue.get()
                self.router.compute_queue_len -= 1
                self.scheduler.update(data)
            
            task = self.scheduler.schedule()

            if task:
                print(f"PE{self.id} is processing a {type(task)}task(id:{task.index}, size:{task.size}) at time {self.env.now:.2f}")
                yield self.env.process(self.run_task(task))
                print(f"Finished at time {self.env.now:.2f}")

    def run_task(self, task):
        self.spm_manager.allocate(task)
        yield self.env.process(task.run(self))
        self.spm_manager.free(task)