import os
import sys
from typing import List
from pydantic import BaseModel
from typing import Tuple
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from analysis.comm_fail import comm_analyzer
from analysis.comp_fail import comp_analyzer
from analysis.trace_format import InstTrace, CommInst, CompInst
from src.sim_type import TaskType
import hashlib

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

    def merge(self, other: "CompressedTrace") -> "CompressedTrace":
        if self.flops != -1:
            my_duration = other.end_time - other.start_time
            self.flops += 1.0 * other.flops / my_duration
        if self.data_size != -1:
            self.duration += other.end_time - other.start_time

class CompressedSummary(BaseModel):
    pe_id: int
    layer_id: int
    start_time: int
    end_time: int
    inference_time: int

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

compute_inst = [TaskType.CONV, TaskType.POOL, TaskType.FC, TaskType.ELEM, TaskType.GCONV, TaskType.PTP, TaskType.TRANS]
communication_inst = [TaskType.SEND, TaskType.RECV]
io_inst = [TaskType.READ, TaskType.WRITE]

# 从原始指令数据中提取 key，并去除冗余数据
def trace_to_key_attr(trace):
    key, attr = None, None
    if trace.instruction_type in compute_inst:
        # 计算指令
        key = f"pe{trace.pe_id}-flops{trace.flops}-layer{trace.layer_id}-inf{trace.inference_time}"
        attr = CompressedTrace(
            layer_id = trace.layer_id,
            pe_id = trace.pe_id,
            start_time = trace.start_time,
            end_time = trace.end_time,
            inference_time = trace.inference_time,
            flops = trace.flops
        )
    elif trace.instruction_type in communication_inst:
        # 通信指令
        key = f"src{trace.src_id}-dst{trace.dst_id}-ds{trace.data_size}-inf{trace.inference_time}"
        attr = CompressedTrace(
            layer_id = trace.layer_id,
            pe_id = trace.pe_id,
            start_time = trace.start_time,
            end_time = trace.end_time,
            inference_time = trace.inference_time,
            data_size = trace.data_size,
            src_id = trace.src_id,
            dst_id = trace.dst_id
        )
    return key, attr

# Stage 1: 参考 BurstSketch 的过滤
class Stage1Bucket:
    def __init__(self):
        self.key = None
        self.count = 0

class RunningTrack:
    def __init__(self, num_hashes=3, num_buckets=128, threshold=10):
        # 哈希函数数量
        self.d = num_hashes
        # 每个哈希表的桶数
        self.m = num_buckets
        self.H = threshold
        # 初始化 d 个哈希表，每个表有 m 个桶
        self.tables = [[Stage1Bucket() for _ in range(self.m)] for _ in range(self.d)]

    def _hashes(self, key: str):
        # 为一个 key 生成 d 个哈希索引
        return [int(hashlib.md5((key + str(i)).encode()).hexdigest(), 16) % self.m for i in range(self.d)]

    def insert(self, key: str) -> bool:
        # 插入 key，并判断是否进入 Stage 2
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
    
# Stage 2: Snapshotting（聚合高负载模式）
# 具有相似失速模式的指令集合
class BurstPattern:
    def __init__(self, key: str, start_time: int, end_time: int, attr: CompressedTrace):
        self.key = key
        self.start_time = start_time
        self.end_time = end_time
        self.count = 1

        # 记录总属性
        self.merged_attr = attr
        self.merged_attr.duration = attr.end_time - attr.start_time
        if attr.flops != -1:
            self.merged_attr.flops = attr.flops / self.merged_attr.duration

    def update(self, start_time: int, end_time: int, attr: dict):
        # 更新失速指令集合
        self.count += 1
        self.start_time = min(self.start_time, start_time)
        self.end_time = max(self.end_time, end_time)
        # 维护各属性变化
        # if self.merged_attr.flops != -1:
        #     print(f"{self.merged_attr.flops} -- {self.count}")
        self.merged_attr.merge(attr)

    # 以 Summary 格式返回
    def summary(self):
        # 返回压缩信息（平均后）
        if self.merged_attr.flops != -1:
            # print(f"{self.merged_attr.flops} / {self.count}")
            return CompressedComp(
                pe_id = self.merged_attr.pe_id,
                layer_id = self.merged_attr.layer_id,
                start_time = self.start_time,
                end_time = self.end_time,
                inference_time = self.merged_attr.inference_time,
                # 某pe执行的 某指令集和的 平均flops
                flops = self.merged_attr.flops / self.count
            )
        else:
            return CompressedComm(
                pe_id = self.merged_attr.pe_id,
                layer_id = self.merged_attr.layer_id,
                start_time = self.start_time,
                end_time = self.end_time,
                inference_time = self.merged_attr.inference_time,
                # 平均通信时间
                avg_time = self.merged_attr.duration / self.count,
                data_size = self.merged_attr.data_size,
                src_id = self.merged_attr.src_id,
                dst_id = self.merged_attr.dst_id
            )

class SnapshotTable:
    def __init__(self, max_size=128):
        # 最大指令模式数量
        self.max_size = max_size
        # key -> BurstPattern
        self.table = dict()

    def insert(self, key: str, start_time: int, end_time: int, attr: CompressedTrace):
        if key not in self.table:
            if len(self.table) >= self.max_size:
                self._evict()
            self.table[key] = BurstPattern(key, start_time, end_time, attr)
        else:
            self.table[key].update(start_time, end_time, attr)

    def _evict(self):
        # 删除最早的指令模式（其他移除方式？）
        oldest_key = min(self.table.keys(), key=lambda k: self.table[k].start_time)
        del self.table[oldest_key]

    # 计算/通信分类
    def get_summaries(self):
        # 返回所有疑似失速片段的列表
        comm_compressed_trace = [pattern.summary() for pattern in self.table.values() if pattern.key[0] == 's']
        comp_compressed_trace = [pattern.summary() for pattern in self.table.values() if pattern.key[0] == 'p']
        return comm_compressed_trace, comp_compressed_trace

class BurstSketchLikeCompressor:
    def __init__(self, num_hashes=3, num_buckets=128, stage2_size=128, threshold=10):
        self.stage1 = RunningTrack(num_hashes, num_buckets, threshold)
        self.stage2 = SnapshotTable(stage2_size)

    def insert(self, key: str, start_time: int, end_time: int, attr: CompressedTrace):
        # 先尝试 insert 到 Stage 1 进行过滤
        if self.stage1.insert(key):
            # insert 到 Stage 2
            self.stage2.insert(key, start_time, end_time, attr)

    def summaries(self, file_path):
        # 返回所有压缩后的 burst 模式摘要
        comm, comp = self.stage2.get_summaries()
        comm_model = CommSummary(trace=comm)
        comp_model = CompSummary(trace=comp)

        comm_json = comm_model.model_dump_json(indent=4)
        comp_json = comp_model.model_dump_json(indent=4)

        comp_json_file = os.path.join(file_path, "comp_trace_compress.json")
        comm_json_file = os.path.join(file_path, "comm_trace_compress.json")

        with open(comp_json_file, "w") as file:
            print(comp_json, file=file)

        with open(comm_json_file, "w") as file:
            print(comm_json, file=file)

ds = BurstSketchLikeCompressor(num_hashes=5, num_buckets=1024, stage2_size=512)
comm_file = "data/darknet19/link/comm_trace.json"
comp_file = "data/darknet19/link/comp_trace.json"

comm_data = comm_analyzer(comm_file)
comp_data = comp_analyzer(comp_file)

for trace in comm_data.trace:
    if trace.instruction_type not in io_inst:
        key, attr = trace_to_key_attr(trace)
        ds.insert(key, attr.start_time, attr.end_time, attr)

for trace in comp_data.trace:
    if trace.instruction_type not in io_inst:
        key, attr = trace_to_key_attr(trace)
        ds.insert(key, attr.start_time, attr.end_time, attr)

ds.summaries(file_path="analysis")