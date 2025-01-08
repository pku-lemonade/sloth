from enum import IntEnum
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

class Read(IOTask):
    def run(self, core):
        with core.lsu.request() as req:
            yield req
            yield core.env.timeout(self.size / core.lsu_bandwidth)

class Write(IOTask):
    def run(self, core):
        with core.lsu.request() as req:
            yield req
            yield core.env.timeout(self.size / core.lsu_bandwidth)

class Conv(ComputeTask):
    def run(self, core):
        with core.tpu.request() as req:
            yield req
            yield core.env.timeout(self.flops / core.tpu_flops)

class Pool(ComputeTask):
    num_operands: int = 1

    def run(self, core):
        with core.tpu.request() as req:
            yield req
            yield core.env.timeout(self.flops / core.tpu_flops)

class FC(ComputeTask):
    num_operands: int = 1

    def run(self, core):
        with core.tpu.request() as req:
            yield req
            yield core.env.timeout(self.flops / core.tpu_flops)

class Send(CommunicationTask):
    src: int = -1

    def run(self, core):
        core.env.process(core.link.transmit(self.size))
        core.router.route_queue_len += 1
        yield core.router.route_queue.put(Data(index=self.index, dst=self.dst, size=self.size))

class Recv(CommunicationTask):
    def run(self, core):
        ...

class Message(BaseModel):
    task: Task
    data: Data