import os
import sys
import json
from typing import List
from pydantic import ValidationError, BaseModel
from typing import Tuple
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from recorder.trace_format import InstTrace, CommInst, CompInst, CommTrace, CompTrace
from evaluater.sim_type import TaskType
import hashlib

def comp_analyzer(filename: str) -> CompTrace:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = CompTrace.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())

def comm_analyzer(filename: str) -> CommTrace:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = CommTrace.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())

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

def trace_to_key_attr(trace):
    key, attr = None, None
    if trace.instruction_type in compute_inst:
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
        self.d_max = _size_of_int(self.d)
        self.m_max = _size_of_int(self.m)
        self.H_max = _size_of_int(self.H)
        self.tables_key_max = 0
        self.tables_count_max = 1

    def _hashes(self, key: str):
        return [int(hashlib.md5((key + str(i)).encode()).hexdigest(), 16) % self.m for i in range(self.d)]

    def insert(self, key: str) -> bool:
        promoted = False
        for i, idx in enumerate(self._hashes(key)):
            bucket = self.tables[i][idx]
            if bucket.key == key:
                bucket.count += 1
                self.tables_count_max = max(self.tables_count_max,_size_of_int(bucket.count))
                if bucket.count >= self.H:
                    promoted = True
            elif bucket.key is None:
                bucket.key = key
                bucket.count = 1
                self.tables_count_max = max(self.tables_count_max,_size_of_int(bucket.count))
                self.tables_key_max = max(self.tables_key_max,_size_of_str(bucket.key))
            else:
                bucket.count -= 1
                if bucket.count <= 0:
                    bucket.key = None
                    bucket.count = 0
        return promoted
    
class FailSlowPattern:
    def __init__(self, key: str, start_time: int, end_time: int, attr: CompressedTrace):
        self.key = key
        self.start_time = start_time
        self.end_time = end_time
        self.count = 1

        self.merged_attr = attr
        self.merged_attr.duration = attr.end_time - attr.start_time
        if attr.flops != -1:
            self.merged_attr.flops = attr.flops / self.merged_attr.duration

    def update(self, start_time: int, end_time: int, attr: dict):
        self.count += 1
        self.start_time = min(self.start_time, start_time)
        self.end_time = max(self.end_time, end_time)
        self.merged_attr.merge(attr)
        if self.merged_attr.flops != -1:
            if self.merged_attr.pe_id == 10:
                return True , self.count, self.end_time, self.merged_attr
        return False , self.count, self.end_time, self.merged_attr

    def summary(self):
        if self.merged_attr.flops != -1:
            return CompressedComp(
                pe_id = self.merged_attr.pe_id,
                layer_id = self.merged_attr.layer_id,
                start_time = self.start_time,
                end_time = self.end_time,
                inference_time = self.merged_attr.inference_time,
                flops = self.merged_attr.flops / self.count
            )
        else:
            return CompressedComm(
                pe_id = self.merged_attr.pe_id,
                layer_id = self.merged_attr.layer_id,
                start_time = self.start_time,
                end_time = self.end_time,
                inference_time = self.merged_attr.inference_time,
                avg_time = self.merged_attr.duration / self.count,
                data_size = self.merged_attr.data_size,
                src_id = self.merged_attr.src_id,
                dst_id = self.merged_attr.dst_id
            )

class SnapshotTable:
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.table = dict()
        
        self.max_size_max = _size_of_int(max_size)
        self.table_key_max = 0
        self.table_value_s_time_max = 1
        self.table_value_e_time_max = 1
        self.table_value_count_max = 1
        self.table_value_attr_max = {
                                    "layer_id" : 1,
                                    "pe_id" : 1,
                                    "start_time" : 1,
                                    "end_time" : 1,
                                    "inference_time" : 1,
                                    "flops" : 1,
                                    "data_size" : 1,
                                    "src_id" : 1,
                                    "dst_id" : 1,
                                    "duration" : 1}
                                

    def insert(self, key: str, start_time: int, end_time: int, attr: CompressedTrace):
        if key not in self.table:
            if len(self.table) >= self.max_size:
                self._evict()
            self.table[key] = FailSlowPattern(key, start_time, end_time, attr)
            
            self.table_key_max = max(self.table_key_max,_size_of_str(key))
            self.table_value_s_time_max = max(self.table_value_s_time_max,_size_of_int(start_time))
            self.table_value_e_time_max = max(self.table_value_e_time_max,_size_of_int(end_time))
            self.update_attr_max(attr)
        else:
            result, new_count, new_e_time, new_attr = self.table[key].update(start_time, end_time, attr)
            
            self.table_value_e_time_max = max(self.table_value_e_time_max,_size_of_int(new_e_time))
            self.table_value_count_max = max(self.table_value_count_max,_size_of_int(new_count))
            self.update_attr_max(new_attr)

    def _evict(self):
        oldest_key = min(self.table.keys(), key=lambda k: self.table[k].start_time)
        del self.table[oldest_key]

    def get_summaries(self):
        comm_compressed_trace = [pattern.summary() for pattern in self.table.values() if pattern.key[0] == 's']
        comp_compressed_trace = [pattern.summary() for pattern in self.table.values() if pattern.key[0] == 'p']
        return comm_compressed_trace, comp_compressed_trace
    
    def update_attr_max(self, attr):
        self.table_value_attr_max["layer_id"] =max(self.table_value_attr_max["layer_id"],_size_of_int(attr.layer_id)) 
        self.table_value_attr_max["pe_id"] =max(self.table_value_attr_max["pe_id"],_size_of_int(attr.pe_id)) 
        self.table_value_attr_max["start_time"] =max(self.table_value_attr_max["start_time"],_size_of_int(attr.start_time)) 
        self.table_value_attr_max["end_time"] =max(self.table_value_attr_max["end_time"],_size_of_int(attr.end_time)) 
        self.table_value_attr_max["inference_time"] =max(self.table_value_attr_max["inference_time"],_size_of_int(attr.inference_time)) 
        self.table_value_attr_max["flops"] =max(self.table_value_attr_max["flops"],_size_of_int(attr.flops)) 
        self.table_value_attr_max["data_size"] =max(self.table_value_attr_max["data_size"],_size_of_int(attr.data_size)) 
        self.table_value_attr_max["src_id"] =max(self.table_value_attr_max["src_id"],_size_of_int(attr.src_id)) 
        self.table_value_attr_max["dst_id"] =max(self.table_value_attr_max["dst_id"],_size_of_int(attr.dst_id)) 
        self.table_value_attr_max["duration"] =max(self.table_value_attr_max["duration"],_size_of_int(attr.duration)) 
    
    
class FailSlowCompressor:
    def __init__(self, num_hashes=3, num_buckets=128, stage2_size=128, threshold=10):
        self.stage1 = RunningTrack(num_hashes, num_buckets, threshold)
        self.stage2 = SnapshotTable(stage2_size)
    
    def insert(self, key: str, start_time: int, end_time: int, attr: CompressedTrace):
        if self.stage1.insert(key):
            self.stage2.insert(key, start_time, end_time, attr)

    def summaries(self):
        comm, comp = self.stage2.get_summaries()
        comm_model = CommSummary(trace=comm)
        comp_model = CompSummary(trace=comp)

        comp_ds = len(comp_model.trace)*12/1024
        comm_ds = len(comm_model.trace)*18/1024
        
        stage1_overhead={ "d":self.stage1.d_max, 
                          "m":self.stage1.m_max, 
                          "H":self.stage1.H_max, 
                          "tables_key":self.stage1.tables_key_max, 
                          "tables_count":self.stage1.tables_count_max
                        }
        stage2_overhead=[
                            { "max_size":self.stage2.max_size_max, 
                              "table_key":self.stage2.table_key_max, 
                              "table_value_s_time":self.stage2.table_value_s_time_max, 
                              "table_value_e_time":self.stage2.table_value_e_time_max,
                              "table_value_count":self.stage2.table_value_count_max
                            },
                        
                        self.stage2.table_value_attr_max]
        return comm_model, comp_model, stage1_overhead, stage2_overhead
    
def _size_of_int(value):
    if value < 256:
        return 1 
    elif value < 65536:
        return 2  
    elif value < 4294967296:
        return 4  
    else:
        return 8  
    
def _size_of_str(value):
    if value==None:
        return 16
    else:
        return len (value.encode('utf-8'))