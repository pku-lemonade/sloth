from typing import List
from pydantic import BaseModel

class InstTrace(BaseModel):
    instruction_id: int
    instruction_type: int
    layer_id: int
    pe_id: int
    start_time: int
    end_time: int
    inference_time: int

class CompInst(InstTrace):
    flops: int

class CompTrace(BaseModel):
    trace: List[CompInst]

class CommInst(InstTrace):
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