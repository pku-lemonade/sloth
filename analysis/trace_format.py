from typing import List
from pydantic import BaseModel

# 合计98bit，13B or 54bit，7B
# 当前负载下每个core约4k条指令，需要28kB空间
# 计算指令trace无需instruction_id和instruction_type可以再少20bit，即34bit，约4B
class InstTrace(BaseModel):
    # 16bit
    instruction_id: int
    # 4bit
    instruction_type: int
    # 8bit
    layer_id: int
    # 6bit
    pe_id: int
    # 32bit
    start_time: int
    # 32bit
    end_time: int
    # 如果存执行时间只需20bit
    inference_time: int

class CompInst(InstTrace):
    flops: int

class CompTrace(BaseModel):
    trace: List[CompInst]

# PE依赖关系建图
# 通信指令trace无需instruction_id可以少16bit
class CommInst(InstTrace):
    # 归一化传输时间
    data_size: int
    src_id: int = -1
    dst_id: int = -1

class CommTrace(BaseModel):
    trace: List[CommInst]

class LinkData(BaseModel):
    src_id: int
    dst_id: int
    layer_id: int
    data_size: int

class LinksData(BaseModel):
    data: List[LinkData]

class LayerGroupInfo(BaseModel):
    start: int
    end: int

class LayerGroupsInfo(BaseModel):
    info: List[LayerGroupInfo]