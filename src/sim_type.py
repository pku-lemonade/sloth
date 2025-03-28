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
    #计算
    CONV = 5
    POOL = 6
    FC = 7
    #点积
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
    #张量不同维度切片
    tensor_slice: List[DimSlice]
    #尺寸
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


class Task(BaseModel):
    #定义变了
    layer_id:int=-1
    string: str
    index: int
    tensor_slice: List[DimSlice]
    flops: int = 0
    num_operands: int
    feat_num: int = 0
    para_num: int = 0
    feat: List[Data] = []
    para: List[Data] = []
    inst:"Instruction"=None
    def size(self) -> int:
        cur_slice = Slice(tensor_slice=self.tensor_slice)
        #assert len(self.feat) <= 1
        assert len(self.para) <= 1
        return cur_slice.size()

class Nop(Task):
    index: int = -1
    size: int = -1
    flops: int = -1
    num_operands: int = -1
    feat: list[Data] = []
    def run(self, core,ins):
        ins.record.exe_start_time.append(core.env.now)
        yield core.env.timeout(0, self.index)
        ins.record.exe_end_time.append(core.env.now)

class IOTask(Task):
    num_operands: int = 0

class ComputeTask(Task):
    layer_id: int
    num_operands: int = 2

    def input_size(self):
        res = 0
        #feat:List[Data]
        #print("-"*40)
        #print("input")
        for input in self.feat:
            input_slice = Slice(tensor_slice=input.tensor_slice)
            #print(input_slice.tensor_slice)
            res += input_slice.size()
        #print("wgt")
        for wgt in self.para:
            wgt_slice = Slice(tensor_slice=wgt.tensor_slice)
            #print(wgt_slice.tensor_slice)
            res += wgt_slice.size()
        return res

    def output_size(self):
        return self.size()

class CommunicationTask(Task):
    dst: int
    src: int
    num_operands: int = 0

class Record(BaseModel):
    exe_start_time: List[int] = []
    exe_end_time: List[int] = []
    ready_run_time: List[int] = []
    #ready_time: List[int] = []
    #self.layer_start=[]
    mulins: List[int]=[]#记录多个指令的唤醒

class Instruction(BaseModel):
    inst_type: TaskType
    index: int
    trigger_index: List[int] = []
    layer_id: int
    data_type: DataType
    position: int = 0
    tensor_slice: List[DimSlice]
    feat_num: int = 0
    para_num: int = 0
    start_time:int=-1#在想应该累计每个block对后面造成的影响，这样的热点或许更有效
    record:Record=Record()
    #目前我没想细化这些，ready到finish都是running
    #通过last_trigger_tree反向搜索，找到第一个running的指令,并将它作为性能的瓶颈
    #ready:bool=False
    running:bool=False
    waitinglast:bool=False#在pre_analysis中将真的只有1个来wait的置为1
    #finish:bool=False
    hot:int=0
    next:List["Instruction"]=[]
    #以及每个指令造成的影响是一样的吗？
    #tensor_slice is unused in hash
    def trig(self):
        self.ready=True
    def run(self):
        self.running=True
    def addhot(self,hot):
        self.hot+=hot

    def __eq__(self, other):
        if not isinstance(other, Instruction):
            return NotImplemented
        return (self.inst_type, self.index, self.layer_id, self.data_type, self.position) == \
               (other.inst_type, other.index, other.layer_id, other.data_type, other.position)

    def __hash__(self):
        return hash((self.inst_type, self.index, self.layer_id, self.data_type, self.position))



class Message(BaseModel):
    ins:Instruction
    src:int
    data: Data
    dst: int


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
    def run(self, core,ins):
        ins.record.ready_run_time.append(core.env.now)
        yield core.lsu.execute("Read"+str(self.index),ceil(self.size(), core.lsu_bandwidth), ins,self.index)

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()

class Write(IOTask):
    string: str = "Write"
    feat_num: int = 1

    def run(self, core,ins):
        ins.record.ready_run_time.append(core.env.now)
        yield core.lsu.execute("Write"+str(self.index),ceil(self.size(), core.lsu_bandwidth), ins,self.index)

    def input_size(self):
        return self.size()

    def output_size(self):
        return 0

#quesition：为什么不*输入通道数和batch_size(似乎=1)?0，1是batch_size和channel吗？
class Conv(ComputeTask):
    string: str = "Conv"
    def calc_flops(self):
        # for CNN
        #para_size=1
        wgt_slice = Slice(tensor_slice=self.para[0].tensor_slice)
        wgt_H = wgt_slice.tensor_slice[2].end - wgt_slice.tensor_slice[2].start
        wgt_W = wgt_slice.tensor_slice[3].end - wgt_slice.tensor_slice[3].start

        self.flops = self.size() * wgt_H * wgt_W

    def run(self, core,ins):
        self.calc_flops()
        # self.flops = 0
        ins.record.ready_run_time.append(core.env.now)
        yield core.tpu.execute("Conv"+str(self.index), ceil(self.flops, core.tpu_flops), ins,self.index)

class Pool(ComputeTask):
    string: str = "Pool"

    def calc_flops(self):
        self.flops = self.size() * 4

    def run(self, core,ins):
        self.calc_flops()
        # self.flops = 0
        ins.record.ready_run_time.append(core.env.now)
        yield core.tpu.execute("Pool"+str(self.index),ceil(self.flops, core.tpu_flops), ins,self.index)
    
class Elem(ComputeTask):
    string: str = "Elem"

    def calc_flops(self):
        self.flops = self.size()

    def run(self, core,ins):
        self.calc_flops()
        # self.flops = 0
        ins.record.ready_run_time.append(core.env.now)
        yield core.tpu.execute("Elem"+str(self.index),ceil(self.flops, core.tpu_flops), ins,self.index)

class FC(ComputeTask):
    string: str = "FC"
    def calc_flops(self):
        self.flops = self.input_size() * self.size()

    def run(self, core,ins):
        self.calc_flops()
        # self.flops = 0
        ins.record.ready_run_time.append(core.env.now)
        yield core.tpu.execute("FC"+str(self.index),ceil(self.flops, core.tpu_flops),ins,self.index)

class Stay(Task):
    string: str = "Stay"
    flops: int = -1
    def run(self, core,ins):
        ins.record.ready_run_time.append(core.env.now)
        ins.record.exe_start_time.append(core.env.now)
        yield core.env.timeout(0)
        ins.record.exe_end_time.append(core.env.now)

    def input_size(self):
        return 0

    def output_size(self):
        return 0

class Send(CommunicationTask):
    string: str = "Send"
    src: int = -1
    feat_num: int = 1
    def run(self, core,ins):
        # print(f"data{self.index} was put into router{core.router.id}")
        # yield core.env.process(core.link.transmit(self.size))
        # core.router.route_queue_len += 1
        # yield core.router.route_queue.put(Message(data=Data(index=self.index, size=self.size), dst=self.dst))
        yield core.data_out.put(Message(data=Data(index=self.index, tensor_slice=self.tensor_slice), src=core.id,dst=self.dst,ins=ins))

    def input_size(self):
        return self.size()

    def output_size(self):
        return 0

class Recv(CommunicationTask):
    string: str = "Recv"
    dst: int = -1
    src: int = -1
    def run(self, core,ins):
        #ins.record.ready_time.append(core.env.now)
        ins.record.ready_run_time.append(core.env.now)
        ins.record.exe_start_time.append(core.env.now)
        ins.record.exe_end_time.append(core.env.now)
        pass

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()
    