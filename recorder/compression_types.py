from typing import List

from pydantic import BaseModel


class CompressedTrace(BaseModel):
    layer_id: int
    pe_id: int
    start_time: int
    end_time: int
    inference_time: int
    flops: int = -1
    data_size: int = -1
    src_id: int = -1
    dst_id: int = -1
    duration: int = 0

    def merge(self, other: "CompressedTrace") -> None:
        if self.flops != -1:
            duration = other.end_time - other.start_time
            if duration > 0:
                self.flops += 1.0 * other.flops / duration
        if self.data_size != -1:
            self.duration += other.end_time - other.start_time


class CompressedSummary(BaseModel):
    pe_id: int
    layer_id: int
    start_time: int
    end_time: int
    inference_time: int
    count: int = 1


class CompressedComp(CompressedSummary):
    flops: float


class CompressedComm(CompressedSummary):
    avg_time: float
    data_size: int
    src_id: int
    dst_id: int


class CompSummary(BaseModel):
    trace: List[CompressedComp]


class CommSummary(BaseModel):
    trace: List[CompressedComm]


class CompressionResult(BaseModel):
    comm: CommSummary
    comp: CompSummary
