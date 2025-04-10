from typing import List
from pydantic import BaseModel

class InstTrace(BaseModel):
    instruction_id: int
    instruction_type: int
    layer_id: int
    pe_id: int
    ready_time: int
    end_time: int

class CompTrace(BaseModel):
    trace: List[InstTrace]

# PE依赖关系建图
class CommInst(InstTrace):
    dependency_id: int
    operands_time: List[int]

class CommTrace(BaseModel):
    trace: List[CommInst]