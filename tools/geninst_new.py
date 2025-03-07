import os
import sys
import json
from typing import List
from enum import IntEnum
from pydantic import BaseModel, ValidationError

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.sim_type import DimSlice, Slice, Instruction, PEworkload, Workload, TaskType, DataType

class Core(BaseModel):
    x: int
    y: int

class Block(BaseModel):
    cores: List[Core]
    tensor_slice: List[DimSlice]

class IFeature(BaseModel):
    source: str
    source_layer_id: int
    blocks: List[Block]

class WFeature(BaseModel):
    source: str
    blocks: List[Block]

class OFeature(BaseModel):
    dest: str
    blocks: List[Block]

class Partition(BaseModel):
    dims: List[int]
    def num(self) -> int:
        res = None
        for dim_part in self.dims:
            if res == None:
                res = dim_part
            else:
                res = res * dim_part
        return res

class Layer(BaseModel):
    type: str
    layer_id: int
    layer_batch_size: int
    output_partition: Partition
    input_fetch: Partition
    input_feature: List[IFeature]
    wgt_feature: List[WFeature]
    output_feature: List[OFeature]

class Network(BaseModel):
    name: str
    batch_size: int
    layers: List[Layer]

def json_analyzer(filename: str) -> Network:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            net = Network.model_validate(data)
            return net
        except ValidationError as e:
            print(e.json())

def get_list_id(x: int, y: int) -> int:
    return x * 4 + y

def intersect(a: Slice, b: Slice) -> Slice:
    new_slice = []
    for id in range(len(a.tensor_slice)):
        new_slice.append(
            DimSlice(
                start = max(a.tensor_slice[id].start, b.tensor_slice[id].start), 
                end = min(a.tensor_slice[id].end, b.tensor_slice[id].end)
            )
        )
    res = Slice(tensor_slice=new_slice)
    return res

def time_range(a: Slice, time: int) -> Slice:
    batch_size = a.tensor_slice[0].end - a.tensor_slice[0].start
    batch_start = batch_size * time
    batch_end = batch_start + batch_size
    a.tensor_slice[0] = DimSlice(start=batch_start, end=batch_end)
    return a

def fetch_range(a: Slice, fetch_sch: Partition, step: int) -> Slice:
    if fetch_sch.num() == 1:
        return a
    
    dim_sizes, dim_lens, dim_poses = [], [], []

    for (i, dim_slice) in enumerate(a.tensor_slice):
        dim_size = 1
        for j in range(i+1, len(fetch_sch.dims)):
            dim_size = dim_size * fetch_sch.dims[j]
        dim_sizes.append(dim_size)
        dim_lens.append(dim_slice.end-dim_slice.start)

        if i == 0:
            dim_poses.append((step%a.size())//dim_sizes[i])
        else:
            dim_poses.append((step%dim_sizes[i-1])//dim_sizes[i])

    new_slice = []
    for id in range(len(a.tensor_slice)):
        new_slice.append(
            DimSlice(
                start = dim_lens[id]*dim_poses[id], 
                end = dim_lens[id]*dim_poses[id]+dim_lens[id]
            )
        )

    return Slice(tensor_slice=new_slice)

# 对权重而言，只有输出的C维度的切分有影响
def wgt_fetch_range(a: Slice, fetch_sch: Partition, step: int) -> Slice:
    if fetch_sch.dims[1] == 1:
        return a
    
    size_0, size_1 = 1, 1
    for dim_part in fetch_sch.dims:
        size_0 = size_0 * dim_part
        size_1 = size_1 * dim_part

    size_0 = size_0 // fetch_sch.dims[0]
    size_1 = size_1 // fetch_sch.dims[0] // fetch_sch.dims[1]

    len_2 = a.tensor_slice[1].end - a.tensor_slice[1].start
    pos_2 = (step % size_0) // size_1
    a.tensor_slice[1] = DimSlice(start=len_2*pos_2, end=len_2*pos_2+len_2)
    return a
 
if __name__ == "__main__":
    net = json_analyzer("tools/mapping2.json")

    global_inst_id = 0
    pewls = [PEworkload(id=id) for id in range(16)]
    
    
    # 枚举层
    for (lid, layer) in enumerate(net.layers):
        # 当前层是否分多次计算
        for time in range(net.batch_size//layer.layer_batch_size):
            # 当前层是否进行fetch
            fetch = layer.input_fetch.num()
            for step in range(fetch):
                # 输入指令
                for input in layer.input_feature:
                    print(f"lid{lid} time{time} step{step}")
                    # 从DRAM读取数据
                    if input.source == "dram":
                        for block in input.blocks:
                            for core in block.cores:
                                global_inst_id += 1

                                cur_range = Slice(tensor_slice=block.tensor_slice)
                                cur_range = time_range(cur_range, time)
                                cur_range = fetch_range(cur_range, layer.input_fetch, step)

                                list_core_id = get_list_id(core.x, core.y)
                                pewls[list_core_id].insts.append(
                                    Instruction(
                                        inst_type = TaskType.READ,
                                        index = global_inst_id,
                                        layer_id = lid,
                                        data_type = DataType.FEAT,
                                        tensor_slice = cur_range.tensor_slice
                                    )
                                )
                    # 从前一层的core读取数据
                    else:
                        # 枚举当前层输入块
                        for block in input.blocks:
                            for core in block.cores:
                                # 当前块需要的输入块range
                                cur_range = Slice(tensor_slice=block.tensor_slice)
                                cur_range = time_range(cur_range, time)
                                cur_range = fetch_range(cur_range, layer.input_fetch, step)
                                # 来源层是否进行多次计算
                                for pre_time in range(net.batch_size//net.layers[input.source_layer_id].layer_batch_size):
                                    # 来源层是否进行fetch
                                    pre_fetch = net.layers[input.source_layer_id].input_fetch.num()
                                    for pre_step in range(pre_fetch):

                                        for pre_block in net.layers[input.source_layer_id].output_feature[0].blocks:
                                            for pre_core in pre_block.cores:
                                                # 来源层的输出块range
                                                pre_range = Slice(tensor_slice=pre_block.tensor_slice)
                                                pre_range = time_range(pre_range, pre_time)
                                                pre_range = fetch_range(pre_range, net.layers[input.source_layer_id].input_fetch, pre_step)
                                                # 计算交集
                                                intersection = intersect(cur_range, pre_range)
                                                if intersection.size() != 0:
                                                    # 一对send/recv共用id
                                                    global_inst_id += 1
                                                    list_core_id = get_list_id(core.x, core.y)
                                                    pre_list_core_id = get_list_id(pre_core.x, pre_core.y)

                                                    pewls[pre_list_core_id].insts.append(
                                                        Instruction(
                                                            inst_type = TaskType.SEND,
                                                            index = global_inst_id,
                                                            layer_id = lid,
                                                            data_type = DataType.FEAT,
                                                            position = list_core_id,
                                                            tensor_slice = intersection.tensor_slice
                                                        )
                                                    )
                                                    pewls[list_core_id].insts.append(
                                                        Instruction(
                                                            inst_type = TaskType.RECV,
                                                            index = global_inst_id,
                                                            layer_id = lid,
                                                            data_type = DataType.FEAT,
                                                            tensor_slice = intersection.tensor_slice
                                                        )
                                                    )
                # 权重指令
                for wgt in layer.wgt_feature:
                    # 只会从DRAM读取数据
                    for block in wgt.blocks:
                        for core in block.cores:
                            global_inst_id += 1

                            cur_range = Slice(tensor_slice=block.tensor_slice)
                            # 多次计算使用同样的权重
                            cur_range = wgt_fetch_range(cur_range, layer.input_fetch, step)

                            list_core_id = get_list_id(core.x, core.y)
                            pewls[list_core_id].insts.append(
                                Instruction(
                                    inst_type = TaskType.READ,
                                    index = global_inst_id,
                                    layer_id = lid,
                                    data_type = DataType.PARA,
                                    tensor_slice = cur_range.tensor_slice
                                )
                            )
                # 计算指令
                for output in layer.output_feature:
                    for block in wgt.blocks:
                        for core in block.cores:
                            global_inst_id += 1

                            cur_range = Slice(tensor_slice=block.tensor_slice)
                            cur_range = time_range(cur_range, time)
                            cur_range = fetch_range(cur_range, layer.input_fetch, step)
                            
                            list_core_id = get_list_id(core.x, core.y)
                            match layer.type:
                                case "conv":
                                    pewls[list_core_id].insts.append(
                                        Instruction(
                                            inst_type = TaskType.CONV,
                                            index = global_inst_id,
                                            layer_id = lid,
                                            data_type = DataType.FEAT,
                                            tensor_slice = cur_range.tensor_slice
                                        )
                                    )
                                case "pool":
                                    pewls[list_core_id].insts.append(
                                        Instruction(
                                            inst_type = TaskType.POOL,
                                            index = global_inst_id,
                                            layer_id = lid,
                                            data_type = DataType.FEAT,
                                            tensor_slice = cur_range.tensor_slice
                                        )
                                    )
                                case "elewise":
                                    pewls[list_core_id].insts.append(
                                        Instruction(
                                            inst_type = TaskType.ELEM,
                                            index = global_inst_id,
                                            layer_id = lid,
                                            data_type = DataType.FEAT,
                                            tensor_slice = cur_range.tensor_slice
                                        )
                                    )
                                case "fc":
                                    pewls[list_core_id].insts.append(
                                        Instruction(
                                            inst_type = TaskType.FC,
                                            index = global_inst_id,
                                            layer_id = lid,
                                            data_type = DataType.FEAT,
                                            tensor_slice = cur_range.tensor_slice
                                        )
                                    )
                
                # 输出指令
                for output in layer.output_feature:
                    # 如果不是dram则等待后续层添加指令
                    if output.dest == "dram":
                        for block in output.blocks:
                            for core in block.cores:
                                global_inst_id += 1

                                cur_range = Slice(tensor_slice=block.tensor_slice)
                                cur_range = time_range(cur_range, time)
                                cur_range = fetch_range(cur_range, layer.input_fetch, step)

                                list_core_id = get_list_id(core.x, core.y)
                                pewls[list_core_id].insts.append(
                                    Instruction(
                                        inst_type = TaskType.WRITE,
                                        index = global_inst_id,
                                        layer_id = lid,
                                        data_type = DataType.PARA,
                                        tensor_slice = cur_range.tensor_slice
                                    )
                                )

    # 构建指令trigger关系
    input_inst = [TaskType.READ, TaskType.RECV]
    comp_inst = [TaskType.CONV, TaskType.POOL, TaskType.ELEM, TaskType.FC]
    output_inst = [TaskType.WRITE, TaskType.SEND]

    for pe_id in range(16):
        last_seg_id = 0

        for (id, inst) in enumerate(pewls[pe_id].insts):
            # 输入/权重trigger计算
            if inst.inst_type in comp_inst:
                while last_seg_id != id:
                    pewls[pe_id].insts[last_seg_id].trigger_index.append(pewls[pe_id].insts[id].index)
                    last_seg_id += 1
            # 计算trigger输出
            if inst.inst_type in output_inst:
                pewls[pe_id].insts[last_seg_id].trigger_index.append(pewls[pe_id].insts[id].index)
            # 处理新的一段计算
            if inst.inst_type in input_inst and pewls[pe_id].insts[last_seg_id].inst_type not in input_inst:
                last_seg_id = id

    wl = Workload(name=net.name, pes=pewls)
    workload_json = wl.model_dump_json(indent=4)

    with open("tools/workload.json", "w") as file:
        print(workload_json, file=file)