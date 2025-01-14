from enum import IntEnum
from typing import List
from pydantic import BaseModel

class Data(BaseModel):
    index: int
    dst: int
    size: int

class TaskType(IntEnum):
    CONV = 0
    READ = 1
    WRITE = 2

class Task(BaseModel):
    index: int
    size: int
    flops: int
    num_operands: int

class IOTask(Task):
    flops: int = 0
    num_operands: int = 0

class ComputeTask(Task):
    layer_id: int
    size: int = 0
    index: int = -1
    num_operands: int = 2
    para: list[Data] = []
    feat: list[Data] = []
        
class CommunicationTask(Task):
    dst: int
    src: int
    flops: int = 0
    num_operands: int = 0

class Instruction(BaseModel):
    inst_type: str
    index: int
    operation: str
    layer_id: int
    data_type: str
    position: int
    size: int

class Operation(BaseModel):
    operation: str
    layer_id: int

class PEworkload(BaseModel):
    id: int
    insts: List[Instruction]

class Workload(BaseModel):
    name: str
    pes: List[PEworkload]

class Read(IOTask):
    def run(self, core):
        with core.lsu.request() as req:
            print(f"waiting for lsu available at {core.env.now:.2f}")
            yield req
            print(f"lsu is available, start reading data{self.index} at {core.env.now:.2f}")
            yield core.env.timeout(self.size / core.lsu_bandwidth)
            print(f"lsu finish reading data{self.index} at {core.env.now:.2f}")

class Write(IOTask):
    num_operands: int = 1
    feat: list[Data] = []

    def run(self, core):
        with core.lsu.request() as req:
            print(f"waiting for lsu available at {core.env.now:.2f}")
            yield req
            print(f"lsu is available, start writing data{self.index} at {core.env.now:.2f}")
            yield core.env.timeout(self.size / core.lsu_bandwidth)
            print(f"lsu finish writing data{self.index} at {core.env.now:.2f}")

class Conv(ComputeTask):
    def calc_flops(self):
        paras = 0
        feats = 0
        for para in self.para:
            paras += para
        for feat in self.feat:
            feats += feat
        self.flops = paras * feats

    def run(self, core):
        with core.tpu.request() as req:
            yield req
            yield core.env.timeout(self.flops / core.tpu_flops)

class Pool(ComputeTask):
    num_operands: int = 1

    def calc_flops(self):
        self.flops = self.oprands[0].size

    def run(self, core):
        with core.tpu.request() as req:
            yield req
            yield core.env.timeout(self.flops / core.tpu_flops)

class FC(ComputeTask):
    def calc_flops(self):
        paras = 0
        feats = 0
        for para in self.para:
            paras += para
        for feat in self.feat:
            feats += feat
        self.flops = paras * feats

    def run(self, core):
        with core.tpu.request() as req:
            yield req
            yield core.env.timeout(self.flops / core.tpu_flops)

class Send(CommunicationTask):
    num_operands: int = 1
    feat: list[Data] = []
    src: int = -1

    def run(self, core):
        print(f"data{self.index} was put into router{core.router.id}")
        core.env.process(core.link.transmit(self.size))
        # core.router.route_queue_len += 1
        yield core.router.route_queue.put(Data(index=self.index, dst=self.dst, size=self.size))

class Recv(CommunicationTask):
    feat: list[Data] = []

class Message(BaseModel):
    task: Task
    data: Data