from enum import IntEnum
from typing import List
from pydantic import BaseModel

INST_OFFSET = 1000000

def ceil(a: int, b: int):
    return (a + b - 1) // b

class Direction(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class RouterFail(BaseModel):
    start_time: int
    end_time: int
    router_id: int
    times: int

class LinkFail(BaseModel):
    start_time: int
    end_time: int
    router_id: int
    direction: Direction
    times: int

class LsuFail(BaseModel):
    start_time: int
    end_time: int
    pe_id: int
    times: int

class TpuFail(BaseModel):
    start_time: int
    end_time: int
    pe_id: int
    times: int

class FailSlow(BaseModel):
    router: List[RouterFail]
    link: List[LinkFail]
    lsu: List[LsuFail]
    tpu: List[TpuFail]

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

opcode2type = {
    "Read": TaskType.READ,
    "Write": TaskType.WRITE,
    "Send": TaskType.SEND,
    "Recv": TaskType.RECV,
    "Stay": TaskType.STAY,
    "Conv": TaskType.CONV,
    "Pool": TaskType.POOL,
    "FC": TaskType.FC,
    "Elem": TaskType.ELEM,
    "GConv": TaskType.GCONV,
    "PTP": TaskType.PTP,
    "Trans": TaskType.TRANS
}

compute_task = [TaskType.CONV, TaskType.POOL, TaskType.FC, TaskType.ELEM, TaskType.GCONV, TaskType.PTP, TaskType.TRANS]
communication_task = [TaskType.SEND, TaskType.RECV]
io_task = [TaskType.READ, TaskType.WRITE]

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
    index: int = -1
    tensor_slice: List[DimSlice] = []

    def __lt__(self, other: "Data") -> bool:
        return self.index < other.index
    
class ProbeData(BaseModel):
    metric: dict = {}
    
class Probe(BaseModel):
    flag: int 
    metric: dict = {}

    def run(self, core, inst_index, layer_id, opcode, flops=None, data_size=None, src=None, dst=None):
        if inst_index not in core.probe_data:
            core.probe_data[inst_index] = ProbeData(metric={})
        
        # 前置 probe 记录基础信息
        if self.flag == 0:
            core.probe_data[inst_index].metric["instruction_id"] = inst_index
            core.probe_data[inst_index].metric["instruction_type"] = opcode2type[opcode]
            core.probe_data[inst_index].metric["layer_id"] = layer_id
            core.probe_data[inst_index].metric["pe_id"] = core.id
        
        # 根据用户定义的 metric 进行记录
        for key in self.metric.keys():
            # print(key)
            match key:
                case "start_time":
                    core.probe_data[inst_index].metric[key] = core.env.now
                case "end_time":
                    core.probe_data[inst_index].metric[key] = core.env.now
                case "flops":
                    core.probe_data[inst_index].metric[key] = flops
                case "data_size":
                    core.probe_data[inst_index].metric[key] = data_size
                case "src_id":
                    core.probe_data[inst_index].metric[key] = src
                case "dst_id":
                    core.probe_data[inst_index].metric[key] = dst


class Task(BaseModel):
    layer_id: int = -1
    opcode: str
    index: int
    tensor_slice: List[DimSlice]
    flops: int = 0
    num_operands: int
    feat_num: int = 0
    para_num: int = 0
    inference_end: bool = False
    feat: List[Data] = []
    para: List[Data] = []
    inst: "Instruction" = None

    probe_st: Probe = None
    probe_ed: Probe = None

    def size(self) -> int:
        cur_slice = Slice(tensor_slice=self.tensor_slice)
        return cur_slice.size()
    
class Nop(Task):
    def run(self, core, ins):
        ins.record.exe_start_time.append((core.env.now, self.inference_time))
        yield core.env.timeout(0, self.index)
        ins.record.exe_end_time.append((core.env.now, self.inference_time))

class IOTask(Task):
    num_operands: int = 0

    def input_size(self):
        raise NotImplementedError(f"{self.opcode} 类未实现 input_size 方法")

    def output_size(self):
        raise NotImplementedError(f"{self.opcode} 类未实现 output_size 方法")

    def run(self, core, ins):
        # ins.record.ready_run_time.append((core.env.now, self.inference_time))
        # ins.record.pe_id = core.id
        yield core.env.process(core.spm_manager.allocate(self.opcode+str(self.index), self.output_size()))
        yield core.lsu.execute(self.opcode+str(self.index), ceil(self.size(), core.lsu_bandwidth), ins)
        core.env.process(core.spm_manager.free(self.opcode+str(self.index), self.input_size()))

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
    
    def calc_flops(self):
        raise NotImplementedError(f"{self.opcode} 类未实现 calc_flops 方法")
    
    def run(self, core, ins):
        self.calc_flops()
        # ins.record.ready_run_time.append((core.env.now, self.inference_time))
        # ins.record.pe_id = core.id
        # ins.record.flops = self.flops
        
        yield core.env.process(core.spm_manager.allocate(self.opcode+str(self.index), self.output_size()))
        
        # 执行前置探针代码 (应该在req后执行，common里有修改)
        # self.probe_st.run(core, self.index, ins.layer_id, self.opcode, flops=self.flops)
        
        true_tpu_flops = core.core_dist.generate()
        # if true_tpu_flops < 500:
        #     print(f"yes, core{core.id} time{core.env.now}, id{ins.index}")
        yield core.tpu.execute(self.opcode+str(self.index), ceil(self.flops, true_tpu_flops), ins, v=self.probe_st, core=core, index=self.index, opcode=self.opcode, flops=self.flops)

        # 执行后置探针代码
        self.probe_ed.run(core, self.index, ins.layer_id, self.opcode)

        core.env.process(core.spm_manager.free(self.opcode+str(self.index), self.input_size()))

    
class Record(BaseModel):
    # 记录的格式是 (时间，推理次数)
    exe_start_time: List[tuple[int, int]] = []
    exe_end_time: List[tuple[int, int]] = []
    ready_run_time: List[tuple[int, int]] = []
    # 记录多个指令的唤醒
    mulins: List[tuple[int, int]] = []
    # 记录指令执行的PE
    pe_id: int = -1
    flops: int = 0

class CommunicationTask(Task):
    dst: int
    src: int
    num_operands: int = 0

class Instruction(BaseModel):
    inst_type: TaskType
    index: int
    trigger_index: List[int] = []
    # 只有WRITE指令会用到
    trigger_core_id: List[int] = []
    layer_id: int
    group_num: int = 1
    data_type: DataType
    position: int = 0
    tensor_slice: List[DimSlice]
    feat_num: int = 0
    para_num: int = 0
    inference_end: bool = False

    # 在想应该累计每个block对后面造成的影响，这样的热点或许更有效
    start_time: int = -1
    record: Record = Record()
    # 目前我没想细化这些，ready到finish都是running
    # 通过last_trigger_tree反向搜索，找到第一个running的指令,并将它作为性能的瓶颈
    # ready: bool = False
    running: bool = False
    # 在pre_analysis中将真的只有1个来wait的置为1
    waitinglast: bool = False
    # finish: bool = False
    hot: int = 0
    next: List["Instruction"] = []
    # 以及每个指令造成的影响是一样的吗？
    # tensor_slice is unused in hash
    def trig(self):
        self.ready = True

    def run(self):
        self.running = True

    def addhot(self, hot):
        self.hot += hot

    def __eq__(self, other):
        if not isinstance(other, Instruction):
            return NotImplemented
        return (self.inst_type, self.index, self.layer_id, self.data_type, self.position) == \
               (other.inst_type, other.index, other.layer_id, other.data_type, other.position)

    def __hash__(self):
        return hash((self.inst_type, self.index, self.layer_id, self.data_type, self.position))
    
class Message(BaseModel):
    ins: Instruction
    src: int
    data: Data = Data()
    dst: int

    def __lt__(self, other: "Message") -> bool:
        return self.data < other.data

# data_size 固定
class Packet(Message):
    # 首尾 packet 标记
    start: bool = False
    end: bool = False

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
    opcode: str = "Read"

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()

class Write(IOTask):
    opcode: str = "Write"
    feat_num: int = 1

    def input_size(self):
        return 0

    def output_size(self):
        return 0

class Conv(ComputeTask):
    opcode: str = "Conv"
    def calc_flops(self):
        wgt_slice = Slice(tensor_slice=self.para[0].tensor_slice)
        wgt_H = wgt_slice.tensor_slice[2].end - wgt_slice.tensor_slice[2].start
        wgt_W = wgt_slice.tensor_slice[3].end - wgt_slice.tensor_slice[3].start

        self.flops = self.size() * wgt_H * wgt_W

class Pool(ComputeTask):
    opcode: str = "Pool"
    def calc_flops(self):
        self.flops = self.size() * 4
    
class Elem(ComputeTask):
    opcode: str = "Elem"
    def calc_flops(self):
        self.flops = self.size()

class FC(ComputeTask):
    opcode: str = "FC"
    def calc_flops(self):
        self.flops = self.input_size() * self.size()

class GConv(ComputeTask):
    opcode: str = "GConv"
    group_num: int
    def calc_flops(self):
        wgt_slice = Slice(tensor_slice=self.para[0].tensor_slice)
        wgt_H = wgt_slice.tensor_slice[2].end - wgt_slice.tensor_slice[2].start
        wgt_W = wgt_slice.tensor_slice[3].end - wgt_slice.tensor_slice[3].start

        self.flops = self.size() * wgt_H * wgt_W
        self.flops //= self.group_num

class PTP(ComputeTask):
    opcode: str = "PTP"
    def calc_flops(self):
        self.flops = self.size() * 7

class Trans(ComputeTask):
    opcode: str = "Trans"
    def calc_flops(self):
        self.flops = 0

class Stay(Task):
    opcode: str = "Stay"
    flops: int = -1
    def run(self, core):
        yield core.env.timeout(0)

    def input_size(self):
        return 0

    def output_size(self):
        return 0

# 这里不用ins.record吗
class Send(CommunicationTask):
    opcode: str = "Send"
    src: int = -1
    feat_num: int = 1
    def run(self, core, ins):
        # # 分析时send/recv合并处理，因为index一致
        # ins.record.pe_id = core.id
        # ins.record.ready_run_time.append((core.env.now, self.inference_time))
        # # yield core.env.process(core.spm_manager.allocate(self.opcode+str(self.index), self.output_size()))
        # ins.record.exe_start_time.append((core.env.now, self.inference_time))

        # 执行前置探针代码
        self.probe_st.run(core, self.index, ins.layer_id, self.opcode, data_size=Slice(tensor_slice=self.tensor_slice).size(), src=core.id)

        # startup time
        startup_time = 10
        yield core.env.timeout(startup_time)

        # true_index = self.index + self.inference_time * INST_OFFSET
        true_index = self.index
        yield core.data_out.put(Message(data=Data(index=true_index, tensor_slice=self.tensor_slice), dst=self.dst, src=core.id, ins=ins))
        # ins.record.exe_end_time.append((core.env.now, self.inference_time))

        # 执行后置探针代码
        self.probe_ed.run(core, self.index, ins.layer_id, self.opcode)

    # 未进行修改
    def run_hop(self, core, ins):
        ins.record.pe_id = core.id
        ins.record.ready_run_time.append((core.env.now, self.inference_time))
        ins.record.exe_start_time.append((core.env.now, self.inference_time))
        # 计算冗余信息
        # packet 大小 16b
        startup_time = 5
        yield core.env.timeout(startup_time)
        
        # 把SEND指令解释为若干packet
        packet_num = ceil(Slice(tensor_slice=self.tensor_slice).size(), 16)
        true_index = self.index + self.inference_time * INST_OFFSET
        for index in range(packet_num):
            if index == 0:
                yield core.data_out.put_hop(Packet(ins=ins, data=Data(index=true_index, tensor_slice=self.tensor_slice), src=core.id, dst=self.dst, start=True))
            elif index == packet_num-1:
                yield core.data_out.put_hop(Packet(ins=ins, data=Data(index=true_index, tensor_slice=self.tensor_slice), src=core.id, dst=self.dst, end=True))
            else:
                yield core.data_out.put_hop(Packet(ins=ins, data=Data(index=true_index, tensor_slice=self.tensor_slice), src=core.id, dst=self.dst))

        ins.record.exe_end_time.append((core.env.now, self.inference_time))

    def input_size(self):
        return 0

    def output_size(self):
        return 0

class Recv(CommunicationTask):
    opcode: str = "Recv"
    dst: int = -1
    src: int = -1
    def run(self, core, ins):
        ins.record.pe_id = core.id
        # ins.record.exe_end_time.append((core.env.now, self.inference_time))

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()
    