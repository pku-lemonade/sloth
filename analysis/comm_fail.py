import os
import sys
import json
import math
import numpy as np
from typing import List
from trace_format import CommTrace, CommInst
from pydantic import ValidationError
from comm_fail import get_id
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
    def __init__(self, node_id:int, layer_group_id: int):
        super().__init__(node_id)
        self.layer_group_id = layer_group_id

class DRAMNode(Node):
    def __init__(self, node_id:int, interval_id: int):
        super().__init__(node_id)
        self.interval = interval_id

class DepEdge:
    def __init__(self, src, dst, waiting_time: int):
        self.src = src
        self.dst = dst
        self.waiting_time = [waiting_time]

    def insert(self, waiting_time: int):
        self.waiting_time.append(waiting_time)

class CommGraph:
    def __init__(self):
        self.node_num = 0
        # (group_id, pe_id) -> Node
        self.nodes = {}
        # (group_id, pe_id) -> node_id
        self.node_id = {}
        # (src_id, dst_id) -> edge
        self.edges = {}

    # SEND/RECV
    def pe2pe(self, src: tuple[int, int], dst: tuple[int, int], waiting_time: int):
        if src not in self.nodes:
            self.node_num += 1
            self.node_id[src] = self.node_num
            self.nodes[src] = PENode(node_id=self.node_num, layer_group_id=src[0])

        if dst not in self.nodes:
            self.node_num += 1
            self.node_id[dst] = self.node_num
            self.nodes[dst] = PENode(node_id=self.node_num, layer_group_id=dst[0])

        if (self.node_id[src], self.node_id[dst]) not in self.edges:
            edge = DepEdge(src=self.nodes[src], dst=self.nodes[dst], waiting_time=waiting_time)
            self.edges[(self.node_id[src], self.node_id[dst])] = edge
            self.nodes[src].out_edges.append(edge)
            self.nodes[dst].in_edges.append(edge)
        else:
            self.edges[(self.node_id[src], self.node_id[dst])].insert(waiting_time)

    # WRITE
    def pe2dram(self, src: tuple[int, int], dram_id: int, waiting_time: int):
        if src not in self.nodes:
            self.node_num += 1
            self.node_id[src] = self.node_num
            self.nodes[src] = PENode(node_id=self.node_num, layer_group_id=src[0])
        
        if (dram_id, -1) not in self.nodes:
            self.node_num += 1
            self.node_id[(dram_id, -1)] = self.node_num
            self.nodes[(dram_id, -1)] = DRAMNode(node_id=self.node_num, interval_id=src[0]+1)

        if (self.node_id[src], self.node_id[(dram_id, -1)]) not in self.edges:
            edge = DepEdge(src=self.nodes[src], dst=self.nodes[(dram_id, -1)], waiting_time=waiting_time)
            self.edges[(self.node_id[src], self.node_id[(dram_id, -1)])] = edge
            self.nodes[src].out_edges.append(edge)
            self.nodes[(dram_id, -1)].in_edges.append(edge)
        else:
            self.edges[(self.node_id[src], self.node_id[(dram_id, -1)])].insert(waiting_time)

    # READ
    def dram2pe(self, dram_id: int, dst: tuple[int, int], waiting_time: int):
        if (dram_id, -1) not in self.nodes:
            self.node_num += 1
            self.node_id[(dram_id, -1)] = self.node_num
            self.nodes[(dram_id, -1)] = DRAMNode(node_id=self.node_num, interval_id=dst[0])

        if dst not in self.nodes:
            self.node_num += 1
            self.node_id[dst] = self.node_num
            self.nodes[dst] = PENode(node_id=self.node_num, layer_group_id=dst[0])
        
        if (self.node_id[(dram_id, -1)], self.node_id[dst]) not in self.edges:
            edge = DepEdge(src=self.nodes[(dram_id, -1)], dst=self.nodes[dst], waiting_time=waiting_time)
            self.edges[(self.node_id[(dram_id, -1)], self.node_id[dst])] = edge
            self.nodes[(dram_id, -1)].out_edges.append(edge)
            self.nodes[dst].in_edges.append(edge)
        else:
            self.edges[(self.node_id[(dram_id, -1)], self.node_id[dst])].insert(waiting_time)

    def build_graph(self, trace: List[CommInst]):
        for inst_trace in trace:
            if inst_trace.instruction_type == TaskType.RECV:
                src = (layer2group[inst_trace.layer_id], inst_trace.src_id)
                dst = (layer2group[inst_trace.layer_id], inst_trace.dst_id)
                waiting_time = inst_trace.end_time - inst_trace.start_time
                self.pe2pe(src=src, dst=dst, waiting_time=waiting_time)

            elif inst_trace.instruction_type == TaskType.READ:
                dst = (layer2group[inst_trace.layer_id], inst_trace.dst_id)
                waiting_time = inst_trace.end_time - inst_trace.start_time
                self.dram2pe(dram_id=layer2group[inst_trace.layer_id], dst=dst, waiting_time=waiting_time)
            
            elif inst_trace.instruction_type == TaskType.WRITE:
                src = (layer2group[inst_trace.layer_id], inst_trace.src_id)
                waiting_time = inst_trace.end_time - inst_trace.start_time
                self.pe2dram(src=src, dram_id=layer2group[inst_trace.layer_id], waiting_time=waiting_time)

    def LinkPressureGraph(self):
        # todo
        pass

if __name__ == '__main__':
    net = json_analyzer("tools/mapping/json")
    arch_configs = config_analyzer("arch/gemini4_4.json")
    comm_trace = comm_analyzer("tools/trace/comm_trace.json")

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