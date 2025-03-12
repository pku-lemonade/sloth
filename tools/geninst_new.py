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
    next: List[int]
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

def time_range(a: Slice, lb_size: int, time: int) -> Slice:
    batch_start = lb_size * time + a.tensor_slice[0].start
    batch_end = lb_size * time + a.tensor_slice[0].end
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
    net = json_analyzer("tools/mapping.json")

    max_inst = 10000
    global_inst_id = 0
    pewls = [PEworkload(id=id) for id in range(16)]
    
    send_map = {}
    
    # 枚举层
    for (lid, layer) in enumerate(net.layers):
        # if lid > 37:
        #     continue
        print(f"Generating inst of layer{lid}.")
        # 当前层是否分多次计算
        for time in range(net.batch_size//layer.layer_batch_size):
            # 当前层是否进行fetch
            fetch = layer.input_fetch.num()
            for step in range(fetch):
                # 输入指令
                for input in layer.input_feature:
                    # print(f"lid{lid} time{time} step{step}")
                    # 从DRAM读取数据
                    if input.source == "dram":
                        for block in input.blocks:
                            for core in block.cores:
                                global_inst_id += 1

                                cur_range = Slice(tensor_slice=block.tensor_slice)
                                cur_range = time_range(cur_range, layer.layer_batch_size, time)
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
                                cur_range = time_range(cur_range, layer.layer_batch_size, time)
                                cur_range = fetch_range(cur_range, layer.input_fetch, step)

                                # if lid == 17 and time == 7 and step == 0:
                                #     print("cur_range")
                                #     print(cur_range)

                                # 来源层是否进行多次计算
                                for pre_time in range(net.batch_size//net.layers[input.source_layer_id].layer_batch_size):
                                    # 来源层是否进行fetch
                                    pre_fetch = net.layers[input.source_layer_id].input_fetch.num()
                                    for pre_step in range(pre_fetch):

                                        for pre_block in net.layers[input.source_layer_id].output_feature[0].blocks:
                                            for pre_core in pre_block.cores:
                                                # 来源层的输出块range
                                                pre_range = Slice(tensor_slice=pre_block.tensor_slice)
                                                pre_range = time_range(pre_range, net.layers[input.source_layer_id].layer_batch_size, pre_time)
                                                pre_range = fetch_range(pre_range, net.layers[input.source_layer_id].input_fetch, pre_step)
                                                # 计算交集
                                                intersection = intersect(cur_range, pre_range)

                                                # if lid == 17 and time == 7 and step == 0:
                                                #     print(f"pre_range: pre_time->{pre_time} pre_fetch->{pre_fetch}")
                                                #     print(pre_range)

                                                if intersection.size() != 0:
                                                    # 一对send/recv共用id
                                                    list_core_id = get_list_id(core.x, core.y)
                                                    pre_list_core_id = get_list_id(pre_core.x, pre_core.y)
                                                    
                                                    if list_core_id != pre_list_core_id:
                                                        info = (pre_time, pre_step, time, step, pre_list_core_id, list_core_id, lid, pre_range.tensor_slice[1].end, pre_range.tensor_slice[2].end, pre_range.tensor_slice[3].end, cur_range.tensor_slice[1].end, cur_range.tensor_slice[2].end, cur_range.tensor_slice[3].end)
                                                        recv_id = send_map[info]

                                                        # if recv_id == 51877:
                                                        #     print(f"RECV: PE{pre_list_core_id} -> PE{list_core_id}")

                                                        pewls[list_core_id].insts.append(
                                                            Instruction(
                                                                inst_type = TaskType.RECV,
                                                                index = recv_id,
                                                                layer_id = lid,
                                                                data_type = DataType.FEAT,
                                                                tensor_slice = intersection.tensor_slice
                                                            )
                                                        )
                wgt_inst_num = [0 for id in range(16)]
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
                    for block in output.blocks:
                        for core in block.cores:
                            global_inst_id += 1

                            cur_range = Slice(tensor_slice=block.tensor_slice)
                            cur_range = time_range(cur_range, layer.layer_batch_size, time)
                            cur_range = fetch_range(cur_range, layer.input_fetch, step)

                            # if global_inst_id == 25083:
                            #     print("cur_range")
                            #     print(cur_range)
                            
                            list_core_id = get_list_id(core.x, core.y)
                            # print(layer.type)
                            # print(list_core_id)
                            match layer.type:
                                case "conv":
                                    pewls[list_core_id].insts.append(
                                        Instruction(
                                            inst_type = TaskType.CONV,
                                            index = global_inst_id,
                                            layer_id = lid,
                                            data_type = DataType.FEAT,
                                            tensor_slice = cur_range.tensor_slice,
                                            # feat_num = input_inst_num[list_core_id],
                                            # para_num = wgt_inst_num[list_core_id]
                                        )
                                    )
                                case "pool":
                                    # print(f"add pool inst into {list_core_id}")
                                    pewls[list_core_id].insts.append(
                                        Instruction(
                                            inst_type = TaskType.POOL,
                                            index = global_inst_id,
                                            layer_id = lid,
                                            data_type = DataType.FEAT,
                                            tensor_slice = cur_range.tensor_slice,
                                            # feat_num = input_inst_num[list_core_id],
                                            # para_num = wgt_inst_num[list_core_id]
                                        )
                                    )
                                case "elewise":
                                    pewls[list_core_id].insts.append(
                                        Instruction(
                                            inst_type = TaskType.ELEM,
                                            index = global_inst_id,
                                            layer_id = lid,
                                            data_type = DataType.FEAT,
                                            tensor_slice = cur_range.tensor_slice,
                                            # feat_num = input_inst_num[list_core_id],
                                            # para_num = wgt_inst_num[list_core_id]
                                        )
                                    )
                                case "fc":
                                    pewls[list_core_id].insts.append(
                                        Instruction(
                                            inst_type = TaskType.FC,
                                            index = global_inst_id,
                                            layer_id = lid,
                                            data_type = DataType.FEAT,
                                            tensor_slice = cur_range.tensor_slice,
                                            # feat_num = input_inst_num[list_core_id],
                                            # para_num = wgt_inst_num[list_core_id]
                                        )
                                    )
                
                # 输出指令
                for output in layer.output_feature:
                    # 需要放回dram
                    if output.dest == "dram":
                        for block in output.blocks:
                            for core in block.cores:
                                global_inst_id += 1

                                cur_range = Slice(tensor_slice=block.tensor_slice)
                                cur_range = time_range(cur_range, layer.layer_batch_size, time)
                                cur_range = fetch_range(cur_range, layer.input_fetch, step)

                                list_core_id = get_list_id(core.x, core.y)
                                pewls[list_core_id].insts.append(
                                    Instruction(
                                        inst_type = TaskType.WRITE,
                                        index = global_inst_id,
                                        layer_id = lid,
                                        data_type = DataType.FEAT,
                                        tensor_slice = cur_range.tensor_slice
                                    )
                                )
                    # 需要发送
                    for block in output.blocks:
                        for core in block.cores:

                            cur_range = Slice(tensor_slice=block.tensor_slice)
                            cur_range = time_range(cur_range, layer.layer_batch_size, time)
                            cur_range = fetch_range(cur_range, layer.input_fetch, step)
                            
                            # if lid == 14:
                            #     print(f"cur_range: time->{time} step->{step}")
                            #     print(cur_range)

                            for next_layer_id in output.next:
                                for next_time in range(net.batch_size//net.layers[next_layer_id].layer_batch_size):
                                    # 触发层是否进行fetch
                                    next_fetch = net.layers[next_layer_id].input_fetch.num()
                                    for next_step in range(next_fetch):
                                        
                                        for next_input in net.layers[next_layer_id].input_feature:
                                            # if lid == 36:
                                            #     print(next_input.source)

                                            if next_input.source == "dram" or next_input.source_layer_id != lid:
                                                continue
                                            # loop_time = 0
                                            for next_block in next_input.blocks:
                                                for next_core in next_block.cores:
                                                    # loop_time += 1
                                                    # 触发层的输出块range
                                                    next_range = Slice(tensor_slice=next_block.tensor_slice)
                                                    next_range = time_range(next_range, net.layers[next_layer_id].layer_batch_size, next_time)
                                                    next_range = fetch_range(next_range, net.layers[next_layer_id].input_fetch, next_step)
                                                    # 计算交集
                                                    intersection = intersect(cur_range, next_range)

                                                    # if lid == 14:
                                                    #     print(f"next_range: time->{next_time} step->{next_step}")
                                                    #     print(next_range)

                                                    if intersection.size() != 0:
                                                        # 一对send/recv共用id
                                                        global_inst_id += 1
                                                        list_core_id = get_list_id(core.x, core.y)
                                                        next_list_core_id = get_list_id(next_core.x, next_core.y)
                                                        
                                                        # if global_inst_id == 51877:
                                                        #     print(f"cur_range: time->{time} step->{step}")
                                                        #     print(cur_range)
                                                        #     print(list_core_id)
                                                        #     print(f"next_range: time->{next_time} step->{next_step} next_lid->{next_layer_id}")
                                                        #     print(next_range)
                                                        #     print(next_list_core_id)
                                                        
                                                        if list_core_id != next_list_core_id:
                                                            pewls[list_core_id].insts.append(
                                                                Instruction(
                                                                    inst_type = TaskType.SEND,
                                                                    index = global_inst_id,
                                                                    layer_id = lid+1,
                                                                    data_type = DataType.FEAT,
                                                                    position = next_list_core_id,
                                                                    tensor_slice = intersection.tensor_slice
                                                                )
                                                            )

                                                            # if global_inst_id == 51877:
                                                            #     print(f"SEND: PE{list_core_id} -> PE{next_list_core_id}")

                                                            info = (time, step, next_time, next_step, list_core_id, next_list_core_id, next_layer_id, cur_range.tensor_slice[1].end, cur_range.tensor_slice[2].end, cur_range.tensor_slice[3].end, next_range.tensor_slice[1].end, next_range.tensor_slice[2].end, next_range.tensor_slice[3].end)
                                                            if info in send_map:
                                                                print("Error!")
                                                            # print(info)
                                                            send_map[info] = global_inst_id

                                            # print(f"loop_time is {loop_time}")

    # 构建指令trigger关系
    input_inst = [TaskType.READ, TaskType.RECV]
    comp_inst = [TaskType.CONV, TaskType.POOL, TaskType.ELEM, TaskType.FC]
    output_inst = [TaskType.WRITE, TaskType.SEND]

    for pe_id in range(16):
        last_seg_id = 0

        for (id, inst) in enumerate(pewls[pe_id].insts):
            # if pe_id == 0:
            #     print(f"layer:{inst.layer_id} inst_index:{inst.index}")
            #     print(inst.inst_type)
            #     print(inst.tensor_slice)
            # input指令指向comp指令，结束后last_seg_id指向comp指令
            if inst.inst_type in comp_inst:
                while last_seg_id != id:
                    if pewls[pe_id].insts[last_seg_id].data_type == DataType.FEAT:
                        inst.feat_num += 1
                    if pewls[pe_id].insts[last_seg_id].data_type == DataType.PARA:
                        inst.para_num += 1

                    pewls[pe_id].insts[last_seg_id].trigger_index.append(inst.index)
                    last_seg_id += 1
            # comp指令指向output指令
            if inst.inst_type in output_inst:
                pewls[pe_id].insts[last_seg_id].trigger_index.append(inst.index)
            # 当前指令是input，并且last_seg_id指令是comp，说明进入新的一段
            if inst.inst_type in input_inst and pewls[pe_id].insts[last_seg_id].inst_type in comp_inst:
                last_seg_id = id

    wl = Workload(name=net.name, pes=pewls)
    workload_json = wl.model_dump_json(indent=4)

    with open("tools/workload.json", "w") as file:
        print(workload_json, file=file)