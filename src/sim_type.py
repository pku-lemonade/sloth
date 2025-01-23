from enum import IntEnum
from typing import List
from pydantic import BaseModel

def ceil(a: int, b: int):
    return (a + b - 1) // b

class Data(BaseModel):
    index: int
    size: int = -1

class Message(BaseModel):
    data: Data
    dst: int

class TaskType(IntEnum):
    READ = 0
    WRITE = 1
    SEND = 2
    RECV = 3
    STAY = 4
    COMP = 5

class OperationType(IntEnum):
    CONV = 0
    POOL = 1
    FC = 2

class DataType(IntEnum):
    PARA = 0
    FEAT = 1
    WGT = 2

class Task(BaseModel):
    index: int
    size: int
    flops: int
    num_operands: int

class Nop(Task):
    index: int = -1
    size: int = -1
    flops: int = -1
    num_operands: int = -1
    feat: list[Data] = []
    def run(self, core):
        yield core.env.timeout(0, value=self.index)

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
    inst_type: TaskType
    index: int
    trigger_index: List[int] = []
    operation: OperationType
    layer_id: int
    data_type: DataType
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
            # print(f"waiting for lsu available at {core.env.now:.2f}")
            yield req
            # print(f"lsu is available, start reading data{self.index} at {core.env.now:.2f}")
            yield core.env.timeout(ceil(self.size, core.lsu_bandwidth), value=self.index)
            # print(f"lsu finish reading data{self.index} at {core.env.now:.2f}")

class Write(IOTask):
    num_operands: int = 1
    feat: list[Data] = []

    def run(self, core):
        with core.lsu.request() as req:
            # print(f"waiting for lsu available at {core.env.now:.2f}")
            yield req
            # print(f"lsu is available, start writing data{self.index} at {core.env.now:.2f}")
            yield core.env.timeout(ceil(self.size, core.lsu_bandwidth), value=self.index)
            # print(f"lsu finish writing data{self.index} at {core.env.now:.2f}")

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
            # if core.id == 9:
            #     print(f"conv{self.index}::req")
            yield req
            # if core.id == 9:
            #     print(f"conv{self.index}::run")
            yield core.env.timeout(ceil(self.flops, core.tpu_flops), value=self.index)

class Pool(ComputeTask):
    num_operands: int = 1

    def calc_flops(self):
        self.flops = self.oprands[0].size

    def run(self, core):
        with core.tpu.request() as req:
            yield req
            yield core.env.timeout(ceil(self.flops, core.tpu_flops), value=self.index)

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
            yield core.env.timeout(ceil(self.flops, core.tpu_flops), value=self.index)

class Stay(Task):
    flops: int = -1
    num_operands: int = -1
    def run(self, core):
        yield core.env.process(core.link.transmit(0))

class Send(CommunicationTask):
    num_operands: int = 1
    feat: list[Data] = []
    src: int = -1

    def run(self, core):
        # print(f"data{self.index} was put into router{core.router.id}")
        core.env.process(core.link.transmit(self.size))
        # core.router.route_queue_len += 1
        yield core.router.route_queue.put(Message(data=Data(index=self.index, size=self.size), dst=self.dst))

class Recv(CommunicationTask):
    dst: int = -1
    src: int = -1
    feat: list[Data] = []
    def run(self, core):
        # wait to be compeleted
        core.spm_manager.allocate(self.size)