from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LinkCandidate:
    period: int
    src_id: int
    dst_id: int
    score: float


def normalize_link(src_id: int, dst_id: int) -> tuple[int, int]:
    if src_id > dst_id:
        return dst_id, src_id
    return src_id, dst_id


def group_paths_by_inference(summary) -> dict[int, list[tuple[int, float, int, int]]]:
    grouped = {}
    for inst_trace in summary.trace:
        exe_time = inst_trace.end_time - inst_trace.start_time
        grouped.setdefault(inst_trace.inference_time, []).append(
            (inst_trace.data_size, exe_time, inst_trace.src_id, inst_trace.dst_id)
        )
    return grouped


class LinkCandidateDetector(ABC):
    name: str

    @abstractmethod
    def detect(self, normal_summary, detect_summary, context) -> list[LinkCandidate]:
        raise NotImplementedError
