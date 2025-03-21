import os
import sys
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import json
from typing import List
from src import ArchConfig
from pydantic import BaseModel, ValidationError
from src.sim_type import DimSlice, Slice, Instruction, PEworkload, Workload, TaskType, DataType

def config_analyzer(filename: str) -> ArchConfig:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            config = ArchConfig.model_validate(data)
            return config
        except ValidationError as e:
            print(e.json())

arch_configs = config_analyzer("arch/gemini4_4.json")
arch_configs.core.spm.size /= 4

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
    # input_fetch实际上以输出为视角fetch，C维度的fetch不影响输入
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
    return x * arch_configs.core.y + y

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

# 对输出而言，四个维度都可能随fetch改变
def output_fetch_range(a: Slice, fetch_sch: Partition, step: int) -> Slice:
    if fetch_sch.num() == 1:
        return a
    
    dim_sizes, dim_lens, dim_poses = [], [], []

    for (i, dim_slice) in enumerate(a.tensor_slice):
        dim_size = 1
        for j in range(i+1, len(fetch_sch.dims)):
            dim_size = dim_size * fetch_sch.dims[j]
        dim_sizes.append(dim_size)
        dim_lens.append((dim_slice.end-dim_slice.start)//fetch_sch.dims[i])

        if i == 0:
            dim_poses.append((step%fetch_sch.num())//dim_sizes[i])
        else:
            dim_poses.append((step%dim_sizes[i-1])//dim_sizes[i])

    new_slice = []
    for id in range(len(a.tensor_slice)):
        remain = (a.tensor_slice[id].end - a.tensor_slice[id].start) % dim_lens[id]
        if remain == 0:
            start = dim_poses[id] * dim_lens[id]
            end = min(dim_poses[id]*dim_lens[id]+dim_lens[id], a.tensor_slice[id].end)
            new_slice.append(DimSlice(start=start, end=end))
        else:
            start = dim_poses[id] * dim_lens[id] + (dim_poses[id] if dim_poses[id] < remain else remain)
            end = min(dim_poses[id]*dim_lens[id]+dim_lens[id]+(dim_poses[id]+1 if dim_poses[id]+1<remain else remain), a.tensor_slice[id].end)
            new_slice.append(DimSlice(start=start, end=end))

    return Slice(tensor_slice=new_slice)

# 对输入而言，C1维度的大小不随fetch的C改变
def input_fetch_range(a: Slice, fetch_sch: Partition, step: int) -> Slice:
    if fetch_sch.num() == 1:
        return a
    
    dim_sizes, dim_lens, dim_poses = [], [], []

    for (i, dim_slice) in enumerate(a.tensor_slice):
        dim_size = 1
        for j in range(i+1, len(fetch_sch.dims)):
            dim_size = dim_size * fetch_sch.dims[j]
        dim_sizes.append(dim_size)
        dim_lens.append((dim_slice.end-dim_slice.start)//fetch_sch.dims[i])

        if i == 0:
            dim_poses.append((step%fetch_sch.num())//dim_sizes[i])
        else:
            dim_poses.append((step%dim_sizes[i-1])//dim_sizes[i])
    
    new_slice = []
    for id in range(len(a.tensor_slice)):
        if id != 1:
            remain = (a.tensor_slice[id].end - a.tensor_slice[id].start) % dim_lens[id]
            if remain == 0:
                start = dim_poses[id] * dim_lens[id]
                end = min(dim_poses[id]*dim_lens[id]+dim_lens[id], a.tensor_slice[id].end)
                new_slice.append(DimSlice(start=start, end=end))
            else:
                start = dim_poses[id] * dim_lens[id] + (dim_poses[id] if dim_poses[id] < remain else remain)
                end = min(dim_poses[id]*dim_lens[id]+dim_lens[id]+(dim_poses[id]+1 if dim_poses[id]+1<remain else remain), a.tensor_slice[id].end)
                new_slice.append(DimSlice(start=start, end=end))
        else:
            # C维度不变
            new_slice.append(a.tensor_slice[id])

    return Slice(tensor_slice=new_slice)

# 对权重而言，只有fetch的C维度有影响（影响第二维）
def wgt_fetch_range(a: Slice, fetch_sch: Partition, step: int) -> Slice:
    if fetch_sch.dims[1] == 1:
        return a
    
    # fetch后C维度的大小
    len_c = (a.tensor_slice[1].end - a.tensor_slice[1].start) // fetch_sch.dims[1]
    # 当前step对应的是C维度的第几块
    size_c = fetch_sch.num() // fetch_sch.dims[0]
    pos_c = (step % size_c) // (fetch_sch.dims[2] * fetch_sch.dims[3])
    # print(f"step:{step}, size_c:{size_c}, pos_c:{pos_c}")
    # 计算该块的C维度范围（最后一块需要均分给前面的块）
    remain = (a.tensor_slice[1].end - a.tensor_slice[1].start) % len_c
    if remain == 0:
        start = pos_c * len_c
        end = min(pos_c*len_c+len_c, a.tensor_slice[1].end)
        a.tensor_slice[1] = DimSlice(start=start, end=end)
    else:
        start = pos_c * len_c + (pos_c if pos_c < remain else remain)
        end = min(pos_c*len_c+len_c+(pos_c+1 if pos_c+1<remain else remain), a.tensor_slice[1].end)
        a.tensor_slice[1] = DimSlice(start=start, end=end)

    return a

def get_input_size(layer: Layer):
    layer_input_part = layer.input_feature[0].blocks[0].tensor_slice
    layer_input_part = Slice(tensor_slice=layer_input_part)
    # print("input-before")
    # print(layer_input_part)
    # print(layer.input_fetch)
    layer_input_part = input_fetch_range(layer_input_part, layer.input_fetch, 0)
    # print("input-after")
    # print(layer_input_part)
    return layer_input_part.size()

def get_spm_size(layer: Layer):
    spm_size = 0
    input_size = get_input_size(layer)
    spm_size += input_size
    
    if layer.wgt_feature[0].blocks:
        layer_wgt_part = layer.wgt_feature[0].blocks[0].tensor_slice
        layer_wgt_part = Slice(tensor_slice=layer_wgt_part)
        # print("wgt-before")
        # print(layer_wgt_part)
        layer_wgt_part = wgt_fetch_range(layer_wgt_part, layer.input_fetch, 0)
        # print("wgt-agter")
        # print(layer_wgt_part)
        spm_size += layer_wgt_part.size()

    # 计算当前分块下的output_slice
    layer_output_part = layer.output_feature[0].blocks[0].tensor_slice
    layer_output_part = Slice(tensor_slice=layer_output_part)
    layer_output_part = output_fetch_range(layer_output_part, layer.input_fetch, 0)
    spm_size += layer_output_part.size()

    return spm_size
 
if __name__ == "__main__":
    net = json_analyzer("tools/mapping.json")
    core_num = arch_configs.core.x * arch_configs.core.y

    global_inst_id = 0
    pewls = [PEworkload(id=id) for id in range(core_num)]
    
    send_map = {}

    # 先统一调整fetch，再生成指令
    for (lid, layer) in enumerate(net.layers):
        print(f"Adjusting fetch of layer{lid}.")

        # 判断spm是否足够
        spm_size = get_spm_size(layer)
        # 若不够，则继续在batch维度切分，需要修改每块slice的batch维度（输入、输出、权重）
        if spm_size > arch_configs.core.spm.size:
            print(f"Change layer_batch_size to {layer.input_fetch.dims[0]}.")
            # layer_batch_size需要是fetch_batch的整数倍 
            layer.layer_batch_size = layer.input_fetch.dims[0]
            # 调整输入batch，四维是[B,C1,H_in,W_in]
            for input in layer.input_feature:
                for block in input.blocks:
                    block.tensor_slice[0] = DimSlice(start=0, end=layer.layer_batch_size)
            # 调整输出batch，四维是[B,C2,H_out,W_out]
            for output in layer.output_feature:
                for block in output.blocks:
                    block.tensor_slice[0] = DimSlice(start=0, end=layer.layer_batch_size)
            # 权重batch，四维是[B=1,C2,C1,R*S]，无需调整

        spm_size = get_spm_size(layer)
        # print(layer.input_fetch)
        # print(f"current_spm_size:{spm_size}, current_input_size:{get_input_size(layer)}")
        c2_len = layer.output_feature[0].blocks[0].tensor_slice[1].end - layer.output_feature[0].blocks[0].tensor_slice[1].start
        # 若还不够，则继续在C2维度切分（影响权重和输出，不影响输入）
        if spm_size > arch_configs.core.spm.size:
            # 计算当前spm容量下一次能计算的最大通道数
            max_channel_num = (arch_configs.core.spm.size-get_input_size(layer)) / ((spm_size-get_input_size(layer))//c2_len)
            # 计算最小fetch块数
            layer.input_fetch.dims[1] = math.ceil(c2_len/max_channel_num)
            print(f"Change input_fetch's Dim-C to {layer.input_fetch.dims[1]}.")
    
    # 枚举层
    for (lid, layer) in enumerate(net.layers):
        print(f"Generating inst of layer{lid}.")
        # 当前层是否分多次计算
        for time in range(net.batch_size//layer.layer_batch_size):
            # 当前层是否进行fetch
            fetch = layer.input_fetch.num()
            for step in range(fetch):
                # 输入指令
                for input in layer.input_feature:
                    # 从DRAM读取数据
                    if input.source == "dram":
                        for block in input.blocks:
                            for core in block.cores:
                                global_inst_id += 1

                                cur_range = Slice(tensor_slice=block.tensor_slice)
                                cur_range = time_range(cur_range, layer.layer_batch_size, time)
                                cur_range = input_fetch_range(cur_range, layer.input_fetch, step)

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
                                cur_range = input_fetch_range(cur_range, layer.input_fetch, step)

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
                                                pre_range = output_fetch_range(pre_range, net.layers[input.source_layer_id].input_fetch, pre_step)
                                                # 计算交集
                                                intersection = intersect(cur_range, pre_range)

                                                if intersection.size() != 0:
                                                    # 一对send/recv共用id
                                                    list_core_id = get_list_id(core.x, core.y)
                                                    pre_list_core_id = get_list_id(pre_core.x, pre_core.y)
                                                    
                                                    if list_core_id != pre_list_core_id:
                                                        info = (pre_time, pre_step, time, step, pre_list_core_id, list_core_id, input.source_layer_id, lid, pre_range.tensor_slice[0].end, pre_range.tensor_slice[1].end, pre_range.tensor_slice[2].end, pre_range.tensor_slice[3].end, cur_range.tensor_slice[0].end, cur_range.tensor_slice[1].end, cur_range.tensor_slice[2].end, cur_range.tensor_slice[3].end)
                                                        recv_id = send_map[info]

                                                        pewls[list_core_id].insts.append(
                                                            Instruction(
                                                                inst_type = TaskType.RECV,
                                                                index = recv_id,
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
                    for block in output.blocks:
                        for core in block.cores:
                            global_inst_id += 1

                            cur_range = Slice(tensor_slice=block.tensor_slice)
                            cur_range = time_range(cur_range, layer.layer_batch_size, time)
                            cur_range = output_fetch_range(cur_range, layer.input_fetch, step)
                            
                            list_core_id = get_list_id(core.x, core.y)
                            match layer.type:
                                case "conv":
                                    pewls[list_core_id].insts.append(
                                        Instruction(
                                            inst_type = TaskType.CONV,
                                            index = global_inst_id,
                                            layer_id = lid,
                                            data_type = DataType.FEAT,
                                            tensor_slice = cur_range.tensor_slice,
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
                                cur_range = output_fetch_range(cur_range, layer.input_fetch, step)

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
                            cur_range = output_fetch_range(cur_range, layer.input_fetch, step)
                            
                            for next_layer_id in output.next:
                                for next_time in range(net.batch_size//net.layers[next_layer_id].layer_batch_size):
                                    # 触发层是否进行fetch
                                    next_fetch = net.layers[next_layer_id].input_fetch.num()
                                    for next_step in range(next_fetch):
                                        
                                        for next_input in net.layers[next_layer_id].input_feature:

                                            if next_input.source == "dram" or next_input.source_layer_id != lid:
                                                continue
                                            for next_block in next_input.blocks:
                                                for next_core in next_block.cores:
                                                    # 触发层的输出块range
                                                    next_range = Slice(tensor_slice=next_block.tensor_slice)
                                                    next_range = time_range(next_range, net.layers[next_layer_id].layer_batch_size, next_time)
                                                    next_range = input_fetch_range(next_range, net.layers[next_layer_id].input_fetch, next_step)
                                                    # 计算交集
                                                    intersection = intersect(cur_range, next_range)

                                                    if intersection.size() != 0:
                                                        # 一对send/recv共用id
                                                        global_inst_id += 1
                                                        list_core_id = get_list_id(core.x, core.y)
                                                        next_list_core_id = get_list_id(next_core.x, next_core.y)
                                                        
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

                                                            info = (time, step, next_time, next_step, list_core_id, next_list_core_id, lid, next_layer_id, cur_range.tensor_slice[0].end, cur_range.tensor_slice[1].end, cur_range.tensor_slice[2].end, cur_range.tensor_slice[3].end, next_range.tensor_slice[0].end, next_range.tensor_slice[1].end, next_range.tensor_slice[2].end, next_range.tensor_slice[3].end)
                                                            if info in send_map:
                                                                print("Error!")
                                                                print(info)
                                                            send_map[info] = global_inst_id

                                            # print(f"loop_time is {loop_time}")

    # 构建指令trigger关系
    input_inst = [TaskType.READ, TaskType.RECV]
    comp_inst = [TaskType.CONV, TaskType.POOL, TaskType.ELEM, TaskType.FC]
    output_inst = [TaskType.WRITE, TaskType.SEND]

    for pe_id in range(core_num):
        last_seg_id = 0

        for (id, inst) in enumerate(pewls[pe_id].insts):
            # if inst.index == 19729:
            #     for i in range(id-5, id+5):
            #         print(pewls[pe_id].insts[i])
                # print(pe_id)   15
                # print(inst.layer_id)   9
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