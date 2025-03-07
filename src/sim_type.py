from enum import IntEnum
from typing import List
from pydantic import BaseModel


def ceil(a: int, b: int):
    return (a + b - 1) // b

class RouterFail(BaseModel):
    start_time: int
    end_time: int
    times: int

class LinkFail(BaseModel):
    start_time: int
    end_time: int
    router_id: int
    direction: int
    times: int

class FailSlow(BaseModel):
    router: List[RouterFail]
    link: List[LinkFail]

#this is defination os message


class TaskType(IntEnum):
    READ = 0
    WRITE = 1
    SEND = 2
    RECV = 3
    STAY = 4

    CONV = 5
    POOL = 6
    FC = 7
    ELEM = 8

class OperationType(IntEnum):
    CONV = 0
    POOL = 1
    FC = 2
    ELEM = 3

class DataType(IntEnum):
    PARA = 0
    FEAT = 1

class DimSlice(BaseModel):
    start: int
    end: int

class Slice(BaseModel):
    tensor_slice: List[DimSlice]
    def size(self) -> int:
        res = None
        for dim_slice in self.tensor_slice:
            dim_len = max(0, dim_slice.end - dim_slice.start)
            if res == None:
                res = dim_len
            else:
                res = res * dim_len
        return res
    
class Data(BaseModel):
    index: int
    tensor_slice: List[DimSlice]

class Message(BaseModel):
    data: Data
    dst: int

class Task(BaseModel):
    string: str
    index: int
    tensor_slice: List[DimSlice]
    flops: int = 0
    num_operands: int
    def size(self) -> int:
        cur_slice = Slice(tensor_slice=self.tensor_slice)
        return cur_slice.size()

class Nop(Task):
    index: int = -1
    size: int = -1
    flops: int = -1
    num_operands: int = -1
    feat: list[Data] = []
    def run(self, core):
        yield core.env.timeout(0, self.index)

class IOTask(Task):
    num_operands: int = 0

class ComputeTask(Task):
    layer_id: int
    num_operands: int = 2
    para: list[Data] = []
    feat: list[Data] = []

class CommunicationTask(Task):
    dst: int
    src: int
    num_operands: int = 0

class Instruction(BaseModel):
    inst_type: TaskType
    index: int
    trigger_index: List[int] = []
    layer_id: int
    data_type: DataType
    position: int = 0
    tensor_slice: List[DimSlice]

class Operation(BaseModel):
    operation: str
    layer_id: int

class PEworkload(BaseModel):
    id: int
    insts: List[Instruction] = []

class Workload(BaseModel):
    name: str
    pes: List[PEworkload] = []

class Read(IOTask):
    string: str = "Read"
    def run(self, core):
        if(self.index==101):
            print(f"Read101:{self.size}")
        yield core.lsu.execute("Read"+str(self.index),ceil(self.size(), core.lsu_bandwidth), self.index)

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()

class Write(IOTask):
    string: str = "Write"
    num_operands: int = 1
    feat: list[Data] = []

    def run(self, core):
        yield core.lsu.execute("Write"+str(self.index),ceil(self.size(), core.lsu_bandwidth), self.index)

    def input_size(self):
        return self.size()

    def output_size(self):
        return 0

class Conv(ComputeTask):
    string: str = "Conv"
    def calc_flops(self):
        # for CNN
        wgt_slice = Slice(tensor_slice=self.para[0].tensor_slice)
        wgt_H = wgt_slice.tensor_slice[2].end - wgt_slice.tensor_slice[2].start
        wgt_W = wgt_slice.tensor_slice[3].end - wgt_slice.tensor_slice[3].start

        self.flops = self.size() * wgt_H * wgt_W

    def run(self, core):
        self.calc_flops()
        # self.flops = 0
        yield core.tpu.execute("Conv"+str(self.index), ceil(self.flops, core.tpu_flops), self.index)

    def input_size(self):
        res = 0
        for input in self.feat:
            input_slice = Slice(tensor_slice=input.tensor_slice)
            res += input_slice.size()
        return res

    def output_size(self):
        return self.size()

class Pool(ComputeTask):
    string: str = "Pool"
    num_operands: int = 1

    def calc_flops(self):
        self.flops = self.size() * 4

    def run(self, core):
        self.calc_flops()
        # self.flops = 0
        yield core.tpu.execute("Pool"+str(self.index),ceil(self.flops, core.tpu_flops), self.index)

    def input_size(self):
        input_slice = Slice(tensor_slice=self.feat[0].tensor_slice)
        return input_slice.size()

    def output_size(self):
        return self.size()
    
class Elem(ComputeTask):
    string: str = "Elem"
    num_operands: int = 1

    def calc_flops(self):
        self.flops = self.size()

    def run(self, core):
        self.calc_flops()
        # self.flops = 0
        yield core.tpu.execute("Elem"+str(self.index),ceil(self.flops, core.tpu_flops), self.index)

    def input_size(self):
        return self.size() * 2

    def output_size(self):
        return self.size()

class FC(ComputeTask):
    string: str = "FC"
    def calc_flops(self):
        self.flops = self.input_size() * self.size()

    def run(self, core):
        self.calc_flops()
        # self.flops = 0
        yield core.tpu.execute("FC"+str(self.index),ceil(self.flops, core.tpu_flops),self.index)

    def input_size(self):
        input_slice = Slice(tensor_slice=self.feat[0].tensor_slice)
        return input_slice.size()

    def output_size(self):
        return self.size()

class Stay(Task):
    string: str = "Stay"
    flops: int = -1
    num_operands: int = -1
    def run(self, core):
        yield core.env.timeout(0)

    def input_size(self):
        return 0

    def output_size(self):
        return 0

class Send(CommunicationTask):
    string: str = "Send"
    num_operands: int = 1
    src: int = -1
    feat: List[Data] = []
    def run(self, core):
        # print(f"data{self.index} was put into router{core.router.id}")
        # yield core.env.process(core.link.transmit(self.size))
        # core.router.route_queue_len += 1
        # yield core.router.route_queue.put(Message(data=Data(index=self.index, size=self.size), dst=self.dst))
        yield core.data_out.put(Message(data=Data(index=self.index, tensor_slice=self.tensor_slice), dst=self.dst))

    def input_size(self):
        return self.size()

    def output_size(self):
        return 0

class Recv(CommunicationTask):
    string: str = "Recv"
    dst: int = -1
    src: int = -1
    feat: List[Data] = []
    def run(self, core):
        pass

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()
    