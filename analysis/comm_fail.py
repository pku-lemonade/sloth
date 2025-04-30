import os
import sys
import json
import math
import numpy as np
from typing import List
from trace_format import CommTrace, CommInst
from pydantic import ValidationError, BaseModel
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

def interval_merge(failslow):
    interval_begin = 0
    merged_failslow = []

    for id in range(1, len(failslow)):
        if failslow[id-1][1]+1 < failslow[id][0]:
            merged_failslow.append((failslow[interval_begin][0], failslow[id-1][1]))
            interval_begin = id
    merged_failslow.append((failslow[interval_begin][0], failslow[-1][1]))
    
    return merged_failslow

class DepEdge:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        # (start_time, exe_time)
        self.events = []
        self.failslow = []

    # exe_time是归一化时间
    def insert(self, start_time: int, exe_time: float):
        # print(f"edge start:{start_time}")
        self.events.append((start_time, exe_time))

    def start_time_range(self):
        if not self.events:
            return None
        
        times = [event[0] for event in self.events]
        # print(f"{min(times)}, {max(times)}")
        return min(times), max(times)
    
    # 生成时间窗口（数据包数量）
    def get_time_window(self, window_size: int, step: int):
        if not self.events:
            return []

        self.events.sort()
        start_time, end_time = self.start_time_range()

        # (window_start_time, [window_events_exe_time])
        windows = []
        cur_pos = 0
        
        while cur_pos < len(self.events):
            # print("loop1")
            event_num = 0
            window_events = []

            while event_num < window_size:
                # print("loop2")
                if cur_pos + event_num >= len(self.events):
                    break
                window_events.append(self.events[cur_pos+event_num][1])
                event_num += 1

            windows.append((self.events[cur_pos][0], window_events))
            cur_pos += step

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
                # print(f"{avg0} vs {avg1}")
                if fail_start == -1:
                    fail_start = t0

            if avg1 > 0 and avg0 / avg1 > threshold:
                if fail_start != -1:
                    failslow.append((fail_start, t1))
                    fail_start = -1

        if fail_start != -1:
            failslow.append((fail_start, windows[-1][0]))

        if not failslow:
            return []

        # 重叠区间合并
        merged_failslow = interval_merge(failslow)
        return merged_failslow
    
class LinkFailslow(BaseModel):
    src_layer: int
    src_node: int
    dst_layer: int
    dst_node: int

# 物理链路回溯，建反向边
class Mesh:
    def __init__(self, group_num: int, x: int, y: int):
        self.group_num = group_num
        self.x = x
        self.y = y
        self.num = self.x * self.y

        # print(f"{self.group_num}-{self.x}-{self.y}")
        
        # [(node_id, [failslow list])]
        self.edges = [[] for _ in range(group_num*x*y)]
        # {node_id, edge_index}
        self.mp = [{} for _ in range(group_num*x*y)]
        # {next_node_id, count}
        self.count = [{} for _ in range(group_num*x*y)]

    # 传入的节点信息为 (group_id, pe_id)
    # 传入的边为正向边
    def mapping(self, src, dst):
        # 忽略READ和WRITE
        if src[1] == -1 or dst[1] == -1:
            return
        
        # print(f"{src} and {dst}")
        # 获取src和dst的二维坐标
        src_x = src[1] // self.y
        src_y = src[1] % self.y
        dst_x = dst[1] // self.y
        dst_y = dst[1] % self.y

        # X first
        while src_x != dst_x:
            if dst_x > src_x:
                src_x += 1
                # 路由后的结点
                curNode = src[0]*self.x*self.y+src_x*self.y+src_y
                # 路由前结点
                nextNode = src[0]*self.x*self.y+(src_x-1)*self.y+src_y
                if nextNode not in self.mp[curNode]:
                    # print(f"{curNode} --> {nextNode}")
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
            else:
                src_x -= 1
                curNode = src[0]*self.x*self.y+src_x*self.y+src_y
                nextNode = src[0]*self.x*self.y+(src_x+1)*self.y+src_y
                if nextNode not in self.mp[curNode]:
                    # print(f"{curNode} --> {nextNode}")
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
        
        # then Y
        while src_y != dst_y:
            if dst_y > src_y:
                src_y += 1
                curNode = src[0]*self.x*self.y+src_x*self.y+src_y
                nextNode = src[0]*self.x*self.y+src_x*self.y+(src_y-1)
                if nextNode not in self.mp[curNode]:
                    # print(f"{curNode} --> {nextNode}")
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
            else:
                src_y -= 1
                curNode = src[0]*self.x*self.y+src_x*self.y+src_y
                nextNode = src[0]*self.x*self.y+src_x*self.y+(src_y+1)
                if nextNode not in self.mp[curNode]:
                    # print(f"{curNode} --> {nextNode}")
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1

        # print("-"*40)

    def backtracking(self, father, curNode, failslow, step, limit):
        print(f"searching: {curNode//self.num} {curNode%self.num}")
        if step == limit:
            return
        
        for id, nextNode in enumerate(self.edges[curNode]):
            if nextNode[0] == father:
                continue
            # 失速区间传递
            nextNode[1].extend(failslow)
            # 记录链路被重复使用的次数
            if nextNode[0] not in self.count[curNode]:
                self.count[curNode][nextNode[0]] = 1
            else:
                self.count[curNode][nextNode[0]] += 1
            self.backtracking(curNode=nextNode[0], father=curNode, failslow=failslow, step=step+1, limit=limit)

    def summary(self):
        merged_count = {}
        failslow_interval = {}
        for id, count in enumerate(self.count):
            # id：起始节点id
            # count：{key：目标结点id，次数}
            for nextNode, cnt in count.items():
                # 多层合并
                src = id % self.num
                dst = nextNode % self.num
                # print(f"{src}-{dst}-{cnt}")

                if src > dst: 
                    src, dst = dst, src
                if (src, dst) not in merged_count:
                    merged_count[(src, dst)] = cnt
                    failslow_interval[(src, dst)] = self.edges[id][self.mp[id][nextNode]][1]
                else:
                    merged_count[(src, dst)] += cnt
                    failslow_interval[(src, dst)].extend(self.edges[id][self.mp[id][nextNode]][1])

        max_count = 0
        max_count_links = []
        for link, count in merged_count.items():
            src = link[0]
            dst = link[1]
            if cnt > max_count:
                max_count = cnt
                max_count_links = [(src, dst, failslow_interval[(src, dst)])]
            elif cnt == max_count:
                max_count_links.append((src, dst, failslow_interval[(src, dst)]))

        # print(f"{max_count} {len(max_count_links)}")

        for src, dst, failslow in max_count_links:
            failslow.sort()
            # print(len(failslow))
            failslow = interval_merge(failslow)
            # if len(failslow) <= 1:
            #     continue
            print(f"Link pe({src}) - pe({dst}) failslow at time {failslow}.")

# 多层级通信图
class CommGraph:
    def __init__(self, trace, mesh):
        self.node_num = 0
        self.mesh = mesh
        self.failslow_edge = []
        # (group_id, pe_id) -> Node
        self.nodes = {}
        # (group_id, pe_id) -> node_id
        self.node_id = {}
        # node_id -> (group_id, pe_id)
        self.node_info = {}
        # (src_id, dst_id) -> edge
        self.edges = {}

        self.vis = {}

        self.build_graph(trace)

    def debug(self):
        for id, edge in enumerate(self.edges.values()):
            print(f"edge {self.node_info[edge.src.id]} -> {self.node_info[edge.dst.id]}")
            print(edge.events)

    # SEND/RECV
    def pe2pe(self, src: tuple[int, int], dst: tuple[int, int], start_time: int, exe_time: float):
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
    def pe2dram(self, src: tuple[int, int], dram_id: int, start_time: int, exe_time: float):
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
    def dram2pe(self, dram_id: int, dst: tuple[int, int], start_time: int, exe_time: float):
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
                self.pe2pe(src=src, dst=dst, start_time=inst_trace.start_time, exe_time=exe_time/inst_trace.data_size)

            elif inst_trace.instruction_type == TaskType.READ:
                dst = (layer2group[inst_trace.layer_id], inst_trace.pe_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.dram2pe(dram_id=layer2group[inst_trace.layer_id]-1, dst=dst, start_time=inst_trace.start_time, exe_time=0.25)
            
            elif inst_trace.instruction_type == TaskType.WRITE:
                src = (layer2group[inst_trace.layer_id], inst_trace.pe_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.pe2dram(src=src, dram_id=layer2group[inst_trace.layer_id], start_time=inst_trace.start_time, exe_time=0.25)

    def DFS(self, curNode, threshold):
        if (self.node_info[curNode.id][0], self.node_info[curNode.id][1]) in self.vis:
            return
        
        self.vis[(self.node_info[curNode.id][0], self.node_info[curNode.id][1])] = 1
        for edge in curNode.out_edges:
            window_size = len(edge.events) // 10
            step = max(window_size // 2, 1)
            failslow = edge.failslow_detect(window_size, step, threshold)

            # 建立回溯反向边
            # print(f"{edge.src.id} -> {edge.dst.id}")
            # 传入的节点信息为 (group_id, pe_id)
            self.mesh.mapping(self.node_info[edge.src.id], self.node_info[edge.dst.id])
            if failslow:
                edge.failslow.extend(failslow)
                self.failslow_edge.append(edge)
                print(f"Layer {self.node_info[edge.src.id][0]}:: Path pe({self.node_info[edge.src.id][1]}) -> pe({self.node_info[edge.dst.id][1]}) failslow at time {failslow}.")
            
            next_node = edge.dst
            self.DFS(next_node, threshold)

    def LinkAnalyze(self, threshold = 2):
        for node in self.nodes.values():
            if len(node.in_edges) == 0:
                self.DFS(node, threshold)

        # todo: failslow spread

if __name__ == '__main__':
    net = json_analyzer("tools/mapping.json")
    arch_configs = config_analyzer("arch/gemini4_4.json")
    comm_trace = comm_analyzer("data/darknet19/router/comm_trace.json")

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

    # print(layer2group)

    cur_layer_id = id
    layer_group_divide.append(group_layers)
    mesh = Mesh(group_num=len(layer_group_divide)+1, x=arch_configs.core.x, y=arch_configs.core.y)

    comm_graph = CommGraph(comm_trace.trace, mesh)
    print(f"Finish building comm_graph, there are {len(comm_graph.nodes)} nodes and {len(comm_graph.edges)} edges in the graph.")
    
    # comm_graph.debug()

    # 失速路径分析
    comm_graph.LinkAnalyze(threshold=2)

    # 物理链路回溯
    for failslow_edge in comm_graph.failslow_edge:
        # 链路图中dst_id
        dst_id = comm_graph.node_info[failslow_edge.dst.id][0]*core_num + comm_graph.node_info[failslow_edge.dst.id][1]
        print(f"backtracking start from {comm_graph.node_info[failslow_edge.dst.id]}")
        mesh.backtracking(curNode=dst_id, father=-1, failslow=failslow_edge.failslow, step=0, limit=1)

    mesh.summary()