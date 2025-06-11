from pydantic import BaseModel
from typing import Tuple
from collections import defaultdict
from analysis.trace_format import InstTrace, CommInst, CompInst
from src.sim_type import TaskType
import hashlib

class InstTrace(BaseModel):
    instruction_id: int
    instruction_type: int 
    layer_id: int
    pe_id: int
    start_time: int
    end_time: int
    inference_time: int
    flops: int = 0
    data_size: int = 0
    src_id: int = -1
    dst_id: int = -1

compute_inst = [TaskType.CONV, TaskType.POOL, TaskType.FC, TaskType.ELEM, TaskType.GCONV, TaskType.PTP, TaskType.TRANS]
communication_inst = [TaskType.SEND, TaskType.RECV]
io_inst = [TaskType.READ, TaskType.WRITE]

def trace_to_key_attr(trace: InstTrace) -> Tuple[str, dict]:
    if trace.instruction_type in compute_inst:
        # 计算指令
        key = f"compute:{trace.pe_id}"
        attr = {
            "duration": trace.end_time - trace.start_time,
            "inference_time": trace.inference_time,
            "flops": trace.flops
        }
    elif trace.instruction_type in communication_inst:
        # 通信指令
        key = f"comm:{trace.src_id}->{trace.dst_id}"
        attr = {
            "duration": trace.end_time - trace.start_time,
            "inference_time": trace.inference_time,
            "data_size": trace.data_size
        }
    return key, attr

# Stage 1
class Stage1Bucket:
    def __init__(self):
        self.key = None
        self.count = 0

class RunningTrack:
    def __init__(self, num_hashes=3, num_buckets=128, threshold=10):
        self.d = num_hashes
        self.m = num_buckets
        self.H = threshold
        self.tables = [[Stage1Bucket() for _ in range(self.m)] for _ in range(self.d)]

    def _hashes(self, key: str):
        return [int(hashlib.md5((key + str(i)).encode()).hexdigest(), 16) % self.m for i in range(self.d)]

    def insert(self, key: str) -> bool:
        promoted = False
        for i, idx in enumerate(self._hashes(key)):
            bucket = self.tables[i][idx]
            if bucket.key == key:
                bucket.count += 1
                if bucket.count >= self.H:
                    promoted = True
            elif bucket.key is None:
                bucket.key = key
                bucket.count = 1
            else:
                bucket.count -= 1
                if bucket.count <= 0:
                    bucket.key = None
                    bucket.count = 0
        return promoted

# Stage 2
class BurstPattern:
    def __init__(self, key: str, start_time: int, attr: dict):
        self.key = key
        self.start_time = start_time
        self.end_time = start_time
        self.count = 1
        self.attr_sum = defaultdict(float)
        for k, v in attr.items():
            self.attr_sum[k] += v

    def update(self, timestamp: int, attr: dict):
        self.end_time = timestamp
        self.count += 1
        for k, v in attr.items():
            self.attr_sum[k] += v

    def summary(self) -> dict:
        return {
            "key": self.key,
            "start_time": self.start_time,
            "duration": self.end_time - self.start_time + 1,
            "count": self.count,
            "avg_attrs": {k: v / self.count for k, v in self.attr_sum.items()}
        }

class SnapshotTable:
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.table = dict()

    def insert(self, key: str, timestamp: int, attr: dict):
        if key not in self.table:
            if len(self.table) >= self.max_size:
                self._evict()
            self.table[key] = BurstPattern(key, timestamp, attr)
        else:
            self.table[key].update(timestamp, attr)

    def _evict(self):
        oldest_key = min(self.table.keys(), key=lambda k: self.table[k].start_time)
        del self.table[oldest_key]

    def get_summaries(self):
        return [pattern.summary() for pattern in self.table.values()]

class BurstSketchLikeCompressor:
    def __init__(self, num_hashes=3, num_buckets=128, stage2_size=128, threshold=10):
        self.stage1 = RunningTrack(num_hashes, num_buckets, threshold)
        self.stage2 = SnapshotTable(stage2_size)

    def insert(self, key: str, timestamp: int, attr: dict):
        if self.stage1.insert(key):
            self.stage2.insert(key, timestamp, attr)

    def summaries(self):
        return self.stage2.get_summaries()

