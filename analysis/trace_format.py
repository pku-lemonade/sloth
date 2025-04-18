from typing import List
from pydantic import BaseModel

class InstTrace(BaseModel):
    instruction_id: int
    instruction_type: int
    layer_id: int
    pe_id: int
    start_time: int
    end_time: int

class CompTrace(BaseModel):
    trace: List[InstTrace]

# PE依赖关系建图
class CommInst(InstTrace):
    src_id: int = -1
    dst_id: int = -1

class CommTrace(BaseModel):
    trace: List[CommInst]