import os
import sys
import json
import math
import numpy as np
from typing import List
from trace_format import CommTrace, CommInst
from pydantic import ValidationError
from comp_fail import get_id
from src.sim_type import TaskType

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from tools.geninst_new import json_analyzer, config_analyzer

layer2group = {}
layer_group_divide = []

def comm_analyzer(filename: str) -> CommTrace:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = CommTrace.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())

class Node:
    def __init__(self, id: int):
        self.id = id
        self.in_edges = []
        self.out_edges = []

class PENode(Node):
    def __init__(self, node_id: int, layer_group_id: int):
        super().__init__(node_id)
        self.layer_group_id = layer_group_id

class DRAMNode(Node):
    def __init__(self, node_id: int, interval_id: int):
        super().__init__(node_id)
        self.interval = interval_id

def calc_window(windows):
    stats = []
    for start_time, event in windows:
        if not event:
            # stats.append((start_time, 0, 0))
            continue
        else:
            avg = sum(event) / len(event)
            std = (sum((exe_time - avg) ** 2 for exe_time in event) / len(event)) ** 0.5
            stats.append((start_time, avg, std))
    return stats

class DepEdge:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        # (start_time, exe_time)
        self.events = []

    def insert(self, start_time: int, exe_time: int):
        # print(f"edge start:{start_time}")
        self.events.append((start_time, exe_time))

    def start_time_range(self):
        if not self.events:
            return None
        
        times = [event[0] for event in self.events]
        # print(f"{min(times)}, {max(times)}")
        return min(times), max(times)
    
    # 生成时间窗口
    def get_time_window(self, window_size: int, step: int):
        if not self.events:
            return []

        start_time, end_time = self.start_time_range()

        # (window_start_time, [window_events_exe_time])
        windows = []
        curr_time = start_time

        while curr_time < end_time:
            window_start = curr_time
            window_end = curr_time + window_size
            window_events = [
                exe_time for (start_time, exe_time) in self.events
                if window_start <= start_time < window_end
            ]
            windows.append((window_start, window_events))
            curr_time += step

        return windows
    
    def failslow_detect(self, window_size, step, threshold):
        windows = self.get_time_window(window_size=window_size, step=step)
        # print(windows)
        stats = calc_window(windows)
        # print(stats)
        # 失速区间
        failslow = []
        fail_start = -1
        # 窗口均值检测
        for i in range(1, len(stats)):
            t0, avg0, _ = stats[i - 1]
            t1, avg1, _ = stats[i]

            if avg0 > 0 and avg1 / avg0 > threshold:
                if fail_start == -1:
                    fail_start = t0

            if avg1 > 0 and avg0 / avg1 > threshold:
                if fail_start != -1:
                    failslow.append((fail_start, t1))
                    fail_start = -1

        if fail_start != -1:
            failslow.append((fail_start, windows[-1][0]+step))
        return failslow

class CommGraph:
    def __init__(self, trace):
        self.node_num = 0
        # (group_id, pe_id) -> Node
        self.nodes = {}
        # (group_id, pe_id) -> node_id
        self.node_id = {}
        # node_id -> (group_id, pe_id)
        self.node_info = {}
        # (src_id, dst_id) -> edge
        self.edges = {}

        self.build_graph(trace)

    # SEND/RECV
    def pe2pe(self, src: tuple[int, int], dst: tuple[int, int], start_time: int, exe_time: int):
        if src not in self.nodes:
            self.node_num += 1
            self.node_id[src] = self.node_num
            self.node_info[self.node_num] = src
            self.nodes[src] = PENode(node_id=self.node_num, layer_group_id=src[0])

        if dst not in self.nodes:
            self.node_num += 1
            self.node_id[dst] = self.node_num
            self.node_info[self.node_num] = src
            self.nodes[dst] = PENode(node_id=self.node_num, layer_group_id=dst[0])

        if (self.node_id[src], self.node_id[dst]) not in self.edges:
            edge = DepEdge(src=self.nodes[src], dst=self.nodes[dst])
            edge.insert(start_time=start_time, exe_time=exe_time)
            self.edges[(self.node_id[src], self.node_id[dst])] = edge
            self.nodes[src].out_edges.append(edge)
            self.nodes[dst].in_edges.append(edge)
        else:
            self.edges[(self.node_id[src], self.node_id[dst])].insert(start_time=start_time, exe_time=exe_time)

    # WRITE
    def pe2dram(self, src: tuple[int, int], dram_id: int, start_time: int, exe_time: int):
        if src not in self.nodes:
            self.node_num += 1
            self.node_id[src] = self.node_num
            self.node_info[self.node_num] = src
            self.nodes[src] = PENode(node_id=self.node_num, layer_group_id=src[0])
        
        if (dram_id, -1) not in self.nodes:
            self.node_num += 1
            self.node_id[(dram_id, -1)] = self.node_num
            self.node_info[self.node_num] = (dram_id, -1)
            self.nodes[(dram_id, -1)] = DRAMNode(node_id=self.node_num, interval_id=dram_id)

        if (self.node_id[src], self.node_id[(dram_id, -1)]) not in self.edges:
            edge = DepEdge(src=self.nodes[src], dst=self.nodes[(dram_id, -1)])
            edge.insert(start_time=start_time, exe_time=exe_time)
            self.edges[(self.node_id[src], self.node_id[(dram_id, -1)])] = edge
            self.nodes[src].out_edges.append(edge)
            self.nodes[(dram_id, -1)].in_edges.append(edge)
        else:
            self.edges[(self.node_id[src], self.node_id[(dram_id, -1)])].insert(start_time=start_time, exe_time=exe_time)

    # READ
    def dram2pe(self, dram_id: int, dst: tuple[int, int], start_time: int, exe_time: int):
        if (dram_id, -1) not in self.nodes:
            self.node_num += 1
            self.node_id[(dram_id, -1)] = self.node_num
            self.node_info[self.node_num] = (dram_id, -1)
            self.nodes[(dram_id, -1)] = DRAMNode(node_id=self.node_num, interval_id=dram_id)

        if dst not in self.nodes:
            self.node_num += 1
            self.node_id[dst] = self.node_num
            self.node_info[self.node_num] = dst
            self.nodes[dst] = PENode(node_id=self.node_num, layer_group_id=dst[0])
        
        if (self.node_id[(dram_id, -1)], self.node_id[dst]) not in self.edges:
            edge = DepEdge(src=self.nodes[(dram_id, -1)], dst=self.nodes[dst])
            edge.insert(start_time=start_time, exe_time=exe_time)
            self.edges[(self.node_id[(dram_id, -1)], self.node_id[dst])] = edge
            self.nodes[(dram_id, -1)].out_edges.append(edge)
            self.nodes[dst].in_edges.append(edge)
        else:
            self.edges[(self.node_id[(dram_id, -1)], self.node_id[dst])].insert(start_time=start_time, exe_time=exe_time)

    def build_graph(self, trace: List[CommInst]):
        for inst_trace in trace:
            if inst_trace.instruction_type == TaskType.RECV:
                src = (layer2group[inst_trace.layer_id], inst_trace.src_id)
                dst = (layer2group[inst_trace.layer_id], inst_trace.dst_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.pe2pe(src=src, dst=dst, start_time=inst_trace.start_time, exe_time=exe_time)

            elif inst_trace.instruction_type == TaskType.READ:
                dst = (layer2group[inst_trace.layer_id], inst_trace.pe_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.dram2pe(dram_id=layer2group[inst_trace.layer_id]-1, dst=dst, start_time=inst_trace.start_time, exe_time=exe_time)
            
            elif inst_trace.instruction_type == TaskType.WRITE:
                src = (layer2group[inst_trace.layer_id], inst_trace.pe_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.pe2dram(src=src, dram_id=layer2group[inst_trace.layer_id], start_time=inst_trace.start_time, exe_time=exe_time)

    def DFS(self, curNode, window_size, step, threshold):
        # print(f"curNode: {self.node_info[curNode.id][0]} {self.node_info[curNode.id][1]}")
        for edge in curNode.out_edges:
            failslow = edge.failslow_detect(window_size, step, threshold)
            if failslow:
                print(f"Path pe({self.node_info[edge.src.id][1]}) -> pe({self.node_info[edge.dst.id][1]}) failslow at time {failslow}.")
            
            next_node = edge.dst
            self.DFS(next_node, window_size, step, threshold)

    def LinkAnalyze(self, window_size: int = 10000, step: int = 5000, threshold = 2):
        for node in self.nodes.values():
            if len(node.in_edges) == 0:
                self.DFS(node, window_size, step, threshold)

        # todo: failslow spread

if __name__ == '__main__':
    net = json_analyzer("tools/mapping.json")
    arch_configs = config_analyzer("arch/gemini4_4.json")
    comm_trace = comm_analyzer("data/darknet19/link/comm_trace.json")

    core_num = arch_configs.core.x * arch_configs.core.y
    layer_mapping = [[] for _ in range(len(net.layers))]

    # 获取层组划分情况
    cur_layer_id = 0
    for id, layer in enumerate(net.layers):
        if layer.layer_group_id != net.layers[cur_layer_id].layer_group_id:
            group_layers = []
            for ind in range(cur_layer_id, id):
                layer2group[ind] = net.layers[cur_layer_id].layer_group_id
                group_layers.append(ind)

            cur_layer_id = id
            layer_group_divide.append(group_layers)

        for output in layer.output_feature:
            for block in output.blocks:
                for pe in block.cores:
                    layer_mapping[id].append(get_id(pe.x, pe.y))

    # 处理最后一个层组
    group_layers = []
    for ind in range(cur_layer_id, len(net.layers)):
        layer2group[ind] = net.layers[cur_layer_id].layer_group_id
        group_layers.append(ind)

    cur_layer_id = id
    layer_group_divide.append(group_layers)

    comm_graph = CommGraph(comm_trace.trace)
    print(f"Finish building comm_graph, there are {len(comm_graph.nodes)} nodes and {len(comm_graph.edges)} edges in the graph.")
    
    # for lygp, pe_id in comm_graph.nodes.keys():
    #     print(f"Node: {lygp, pe_id} in_edge_num: {len(comm_graph.nodes[(lygp, pe_id)].in_edges)}")

    for edge in comm_graph.edges.values():
        print(edge.events)

    # comm_graph.LinkAnalyze()
    