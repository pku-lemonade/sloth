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
    GCONV = 9
    PTP = 10
    TRANS = 11

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
    
    def max(self, other: "Slice") -> "Slice":
        res = []
        for i in range(len(self.tensor_slice)):
            res.append(
                DimSlice(
                    start = min(self.tensor_slice[i].start, other.tensor_slice[i].start),
                    end = max(self.tensor_slice[i].end, other.tensor_slice[i].end)
                )
            )
        return Slice(tensor_slice=res)
    
class Data(BaseModel):
    index: int
    tensor_slice: List[DimSlice]

    def __lt__(self, other: "Data") -> bool:
        return self.index < other.index

class Message(BaseModel):
    data: Data
    dst: int

class Task(BaseModel):
    string: str
    index: int
    tensor_slice: List[DimSlice]
    flops: int = 0
    num_operands: int
    feat_num: int = 0
    para_num: int = 0
    feat: List[Data] = []
    para: List[Data] = []
    def size(self) -> int:
        cur_slice = Slice(tensor_slice=self.tensor_slice)
        return cur_slice.size()

class Nop(Task):
    def run(self, core):
        yield core.env.timeout(0, self.index)

class IOTask(Task):
    num_operands: int = 0

class ComputeTask(Task):
    layer_id: int
    num_operands: int = 2

    def input_size(self):
        res = 0
        for input in self.feat:
            input_slice = Slice(tensor_slice=input.tensor_slice)
            res += input_slice.size()
            
        for wgt in self.para:
            wgt_slice = Slice(tensor_slice=wgt.tensor_slice)
            res += wgt_slice.size()
        return res + self.size()

    def output_size(self):
        return self.size()

class CommunicationTask(Task):
    dst: int
    src: int
    num_operands: int = 0

class Instruction(BaseModel):
    inst_type: TaskType
    index: int
    trigger_index: List[int] = []
    layer_id: int
    group_num: int = 1
    data_type: DataType
    position: int = 0
    tensor_slice: List[DimSlice]
    feat_num: int = 0
    para_num: int = 0

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
        # if(self.index==101):
        #     print(f"Read101:{self.size}")
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        # core.spm_manager.allocate(self.string+str(self.index), self.output_size())
        yield core.lsu.execute("Read"+str(self.index),ceil(self.size(), core.lsu_bandwidth), self.index)

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()

class Write(IOTask):
    string: str = "Write"
    feat_num: int = 1

    def run(self, core):
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        # core.spm_manager.allocate(self.string+str(self.index), self.output_size())
        yield core.lsu.execute("Write"+str(self.index),ceil(self.size(), core.lsu_bandwidth), self.index)

    def input_size(self):
        # return self.size()
        return 0

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
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        # core.spm_manager.allocate(self.string+str(self.index), self.output_size())
        yield core.tpu.execute("Conv"+str(self.index), ceil(self.flops, core.tpu_flops), self.index)

class Pool(ComputeTask):
    string: str = "Pool"

    def calc_flops(self):
        self.flops = self.size() * 4

    def run(self, core):
        self.calc_flops()
        # self.flops = 0
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        # core.spm_manager.allocate(self.string+str(self.index), self.output_size())
        yield core.tpu.execute("Pool"+str(self.index),ceil(self.flops, core.tpu_flops), self.index)
    
class Elem(ComputeTask):
    string: str = "Elem"

    def calc_flops(self):
        self.flops = self.size()

    def run(self, core):
        self.calc_flops()
        # self.flops = 0
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        # core.spm_manager.allocate(self.string+str(self.index), self.output_size())
        yield core.tpu.execute("Elem"+str(self.index),ceil(self.flops, core.tpu_flops), self.index)

class FC(ComputeTask):
    string: str = "FC"
    def calc_flops(self):
        self.flops = self.input_size() * self.size()

    def run(self, core):
        self.calc_flops()
        # self.flops = 0
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        # core.spm_manager.allocate(self.string+str(self.index), self.output_size())
        yield core.tpu.execute("FC"+str(self.index),ceil(self.flops, core.tpu_flops),self.index)

class GConv(ComputeTask):
    string: str = "GConv"
    group_num: int
    def calc_flops(self):
        wgt_slice = Slice(tensor_slice=self.para[0].tensor_slice)
        wgt_H = wgt_slice.tensor_slice[2].end - wgt_slice.tensor_slice[2].start
        wgt_W = wgt_slice.tensor_slice[3].end - wgt_slice.tensor_slice[3].start

        self.flops = self.size() * wgt_H * wgt_W
        self.flops //= self.group_num
        
    def run(self, core):
        self.calc_flops()
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        yield core.tpu.execute("GConv"+str(self.index), ceil(self.flops, core.tpu_flops), self.index)

class PTP(ComputeTask):
    string: str = "PTP"
    def calc_flops(self):
        self.flops = self.size() * 7

    def run(self, core):
        self.calc_flops()
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        yield core.tpu.execute("PTP"+str(self.index), ceil(self.flops, core.tpu_flops), self.index)

class Trans(ComputeTask):
    string: str = "Trans"
    def calc_flops(self):
        self.flops = 0

    def run(self, core):
        self.calc_flops()
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        yield core.tpu.execute("Trans"+str(self.index), ceil(self.flops, core.tpu_flops), self.index)

class Stay(Task):
    string: str = "Stay"
    flops: int = -1
    def run(self, core):
        yield core.env.timeout(0)

    def input_size(self):
        return 0

    def output_size(self):
        return 0

class Send(CommunicationTask):
    string: str = "Send"
    src: int = -1
    feat_num: int = 1
    def run(self, core):
        # print(f"data{self.index} was put into router{core.router.id}")
        # yield core.env.process(core.link.transmit(self.size))
        # core.router.route_queue_len += 1
        # yield core.router.route_queue.put(Message(data=Data(index=self.index, size=self.size), dst=self.dst))
        yield core.env.process(core.spm_manager.allocate(self.string+str(self.index), self.output_size()))
        # core.spm_manager.allocate(self.string+str(self.index), self.output_size())
        yield core.data_out.put(Message(data=Data(index=self.index, tensor_slice=self.tensor_slice), dst=self.dst))

    def input_size(self):
        # return self.size()
        return 0

    def output_size(self):
        return 0

class Recv(CommunicationTask):
    string: str = "Recv"
    dst: int = -1
    src: int = -1
    def run(self, core):
        pass

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()
    