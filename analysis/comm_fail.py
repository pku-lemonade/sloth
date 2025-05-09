import os
import sys
import json
import math
import numpy as np
from typing import List
from scipy.stats import norm
# from scipy.special import softmax
from trace_format import CommTrace, CommInst, CompInst
from pydantic import ValidationError, BaseModel
from comp_fail import get_id, comp_analyzer
from src.sim_type import TaskType
from distribution import CoreDist
from tomography import NoCTomography

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
        # (start_time, exe_time, pkg_size)
        self.events = []
        self.failslow = []

    # exe_time是归一化时间
    def insert(self, start_time: int, exe_time: float, size: int):
        # print(f"edge start:{start_time}")
        self.events.append((start_time, exe_time, size))

    def sum(self):
        if not self.events:
            return 0
        # 获取传输数据量总和
        res = 0
        for event in self.events:
            res += event[2]
        return res

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

class FailSlowDetector:
    def __init__(self, threshold=0.01):
        self.mean = None
        self.std = None
        self.threshold = threshold
        self.fitted = False

    def fit(self, performance_data):
        self.mean = np.mean(performance_data)
        self.std = np.std(performance_data)
        self.fitted = True

    def failslow_probability(self, x):
        if not self.fitted:
            raise ValueError("Model not fitted. Call `fit` with training data first.")
        
        z = abs(x - self.mean) / self.std
        p = 2 * norm.sf(z)
        return p

    def is_failslow(self, x):
        return self.failslow_probability(x) < self.threshold
    
def softmax(x, beta = 1.0):
    x = np.array(x)
    e_x = np.exp(beta * (x - np.max(x)))
    return e_x / e_x.sum()

# 物理链路回溯，建反向边
class Mesh:
    def __init__(self, group_num: int, x: int, y: int):
        self.group_num = group_num
        self.x = x
        self.y = y
        self.num = self.x * self.y
        # DRAM用一个结点表示
        self.N = self.group_num * self.num + 1
        self.time_range = [(0, 0) for _ in range(self.N)]

        self.core_dist = CoreDist()

        # print(f"{self.group_num}-{self.x}-{self.y}")
        
        # [(node_id, [failslow list])]
        self.edges = [[] for _ in range(self.N)]
        # {node_id, edge_index}
        self.mp = [{} for _ in range(self.N)]
        # {next_node_id, count}
        self.count = [{} for _ in range(self.N)]
        # 记录每条链路被失速路径覆盖次数
        self.link_count = np.zeros((self.N, self.N))
        # 记录每条链路通信数据量
        self.link_size_count = np.zeros((self.N, self.N))

        # 目前还未引入时序信息
        # 基于概率分布的失速检测
        self.detector = FailSlowDetector(threshold=0.1)
        # 核心故障概率
        self.core_failslow_prob = np.zeros(self.N)
        # 链路故障概率
        self.link_failslow_prob = np.zeros((self.N, self.N))
        # 链路权重矩阵
        self.transition_matrix = np.zeros((self.N, self.N))

    # 传入的节点信息为 (group_id, pe_id)
    # 传入的边为正向边
    def mapping(self, src, dst, size, failslow):
        # READ
        if src[1] == -1:
            dst_x = dst[1] // self.y
            dst_y = dst[1] % self.y
            dst_id = (dst[0]-1)*self.num + dst_x*self.y + dst_y
            self.link_size_count[self.N-1][dst_id] += size
            self.link_size_count[dst_id][self.N-1] += size
            if failslow:
                self.link_count[self.N-1][dst_id] += 1
                self.link_count[dst_id][self.N-1] += 1
            return
        
        # WRITE
        if dst[1] == -1:
            src_x = src[1] // self.y
            src_y = src[1] % self.y
            src_id = (src[0]-1)*self.num + src_x*self.y + src_y
            self.link_size_count[src_id][self.N-1] += size
            self.link_size_count[self.N-1][src_id] += size
            if failslow:
                self.link_count[src_id][self.N-1] += 1
                self.link_count[self.N-1][src_id] += 1
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
                curNode = (src[0]-1)*self.x*self.y+src_x*self.y+src_y
                # 路由前结点
                nextNode = (src[0]-1)*self.x*self.y+(src_x-1)*self.y+src_y
                if nextNode not in self.mp[curNode]:
                    # print(f"{curNode} --> {nextNode}")
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
                    self.link_size_count[curNode][nextNode] += size
                    self.link_size_count[nextNode][curNode] += size
                    if failslow:
                        self.link_count[curNode][nextNode] += 1
                        self.link_count[nextNode][curNode] += 1
            else:
                src_x -= 1
                curNode = (src[0]-1)*self.x*self.y+src_x*self.y+src_y
                nextNode = (src[0]-1)*self.x*self.y+(src_x+1)*self.y+src_y
                if nextNode not in self.mp[curNode]:
                    # print(f"{curNode} --> {nextNode}")
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
                    self.link_size_count[curNode][nextNode] += size
                    self.link_size_count[nextNode][curNode] += size
                    if failslow:
                        self.link_count[curNode][nextNode] += 1
                        self.link_count[nextNode][curNode] += 1
        
        # then Y
        while src_y != dst_y:
            if dst_y > src_y:
                src_y += 1
                curNode = (src[0]-1)*self.x*self.y+src_x*self.y+src_y
                nextNode = (src[0]-1)*self.x*self.y+src_x*self.y+(src_y-1)
                if nextNode not in self.mp[curNode]:
                    # print(f"{curNode} --> {nextNode}")
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
                    self.link_size_count[curNode][nextNode] += size
                    self.link_size_count[nextNode][curNode] += size
                    if failslow:
                        self.link_count[curNode][nextNode] += 1
                        self.link_count[nextNode][curNode] += 1
            else:
                src_y -= 1
                curNode = (src[0]-1)*self.x*self.y+src_x*self.y+src_y
                nextNode = (src[0]-1)*self.x*self.y+src_x*self.y+(src_y+1)
                if nextNode not in self.mp[curNode]:
                    # print(f"{curNode} --> {nextNode}")
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
                    self.link_size_count[curNode][nextNode] += size
                    self.link_size_count[nextNode][curNode] += size
                    if failslow:
                        self.link_count[curNode][nextNode] += 1
                        self.link_count[nextNode][curNode] += 1

        # print("-"*40)

    def link_prob_init(self):
        # 初始化链路边权、故障概率
        max_fail_count = self.link_count.max()
        for i in range(self.N):
            path_num = self.link_size_count[i].sum()
            if path_num == 0:
                continue
            # else:
            #     print(f"{i}: {path_num}")
            # 根据传输数据量计算权重
            for j in range(self.N):
                # print(f"{self.link_size_count[i][j]} / {path_num} = {self.link_size_count[i][j] / path_num}")
                # print(1.0/self.N)
                self.transition_matrix[i][j] = (self.link_size_count[i][j] / path_num)
                self.link_failslow_prob[i][j] = self.link_count[i][j] / max_fail_count

    def core_prob_init(self, layer_group, pe_id, flops, start_time, end_time):
        core_id = (layer_group-1) * self.num + pe_id
        # 初始化结点故障概率
        self.core_failslow_prob[core_id] = self.core_dist.failslow_prob(flops)# + (1 / (self.N))
        self.time_range[core_id] = (start_time, end_time)

    # 参考GrootRank进行根因定位
    def pagerank(self, alpha = 0.85, tol = 1e-2, max_iter = 100):
        jump_prob = self.core_failslow_prob

        for _ in range(max_iter):
            new_prob = alpha * self.transition_matrix.T @ self.core_failslow_prob
            # 根据聚类结果设置随机跳转策略
            for i in range(self.N):
                new_prob[i] += (1-alpha) * jump_prob[i] / jump_prob.sum()
            if np.linalg.norm(new_prob-self.core_failslow_prob, ord=1) < tol:
                break
            self.core_failslow_prob = new_prob

        return self.core_failslow_prob

    def backtracking(self, father, curNode, failslow, step, limit):
        print(f"searching: {curNode//self.num} {curNode%self.num}")
        if step == limit:
            return
        
        for id, nextNode in enumerate(self.edges[curNode]):
            if nextNode[0] == father:
                continue
            # 失速区间传递
            nextNode[1].extend(failslow)
            
            # 失速概率更新
            # self.failslow_prob[nextNode[0]] = ...

            # 记录链路被重复使用的次数
            if nextNode[0] not in self.count[curNode]:
                self.count[curNode][nextNode[0]] = 1
            else:
                self.count[curNode][nextNode[0]] += 1
            self.backtracking(curNode=nextNode[0], father=curNode, failslow=failslow, step=step+1, limit=limit)

    def pagerank_summary(self, k=30, threshold=0.2):
        print("="*20)
        output_file = 'output.csv'
        np.savetxt(output_file, self.transition_matrix, delimiter=',', fmt='%f')

        # print(self.core_failslow_prob[0:self.N-1])
        # failslow_prob = softmax(self.core_failslow_prob[0:self.N-1], k=30, beta=100)
        # print(failslow_prob)
        # sorted_prob_id = np.argsort(-failslow_prob)
        # sorted_prob = failslow_prob[sorted_prob_id]

        # 分层计算失速概率
        for group_id in range(self.group_num):
            start_id = group_id * self.num
            end_id = start_id + self.num

            # print("="*40)
            # print(self.core_failslow_prob[start_id:end_id])
            group_prob = softmax(self.core_failslow_prob[start_id:end_id] ** 2, beta=15000)
            # print(group_prob)

            for id in range(self.num):
                if group_prob[id] < threshold:
                    continue
                node_id = start_id + id
                print(f"FailSlow: PE{node_id%self.num} during cycle [{self.time_range[node_id][0]},{self.time_range[node_id][1]}], Prob {group_prob[id]*100:.2f}%.")

        # sorted_PR_id = np.argsort(-self.core_failslow_prob[0:self.N-1])
        # sorted_PR = self.core_failslow_prob[sorted_PR_id]
        # sorted_PR = sorted_PR ** 2
        # # print(sorted_PR)
        # topk_prob = softmax(sorted_PR[0:k], beta=10000)
        # # print(topk_prob)

        # for id in range(k):
        #     if topk_prob[id] < threshold:
        #         break
        #     node_id = sorted_PR_id[id]
        #     print(f"FailSlow: PE{node_id%self.num} during cycle [{self.time_range[node_id][0]},{self.time_range[node_id][1]}], Prob {topk_prob[id]*100:.2f}%.")

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
    def pe2pe(self, src: tuple[int, int], dst: tuple[int, int], start_time: int, exe_time: float, size: int):
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
            edge.insert(start_time=start_time, exe_time=exe_time, size=size)
            self.edges[(self.node_id[src], self.node_id[dst])] = edge
            self.nodes[src].out_edges.append(edge)
            self.nodes[dst].in_edges.append(edge)
        else:
            self.edges[(self.node_id[src], self.node_id[dst])].insert(start_time=start_time, exe_time=exe_time, size=size)

    # WRITE
    def pe2dram(self, src: tuple[int, int], dram_id: int, start_time: int, exe_time: float, size: int):
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
            edge.insert(start_time=start_time, exe_time=exe_time, size=size)
            self.edges[(self.node_id[src], self.node_id[(dram_id, -1)])] = edge
            self.nodes[src].out_edges.append(edge)
            self.nodes[(dram_id, -1)].in_edges.append(edge)
        else:
            self.edges[(self.node_id[src], self.node_id[(dram_id, -1)])].insert(start_time=start_time, exe_time=exe_time, size=size)

    # READ
    def dram2pe(self, dram_id: int, dst: tuple[int, int], start_time: int, exe_time: float, size: int):
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
            edge.insert(start_time=start_time, exe_time=exe_time, size=size)
            self.edges[(self.node_id[(dram_id, -1)], self.node_id[dst])] = edge
            self.nodes[(dram_id, -1)].out_edges.append(edge)
            self.nodes[dst].in_edges.append(edge)
        else:
            self.edges[(self.node_id[(dram_id, -1)], self.node_id[dst])].insert(start_time=start_time, exe_time=exe_time, size=size)

    def build_graph(self, trace: List[CommInst]):
        for inst_trace in trace:
            if inst_trace.instruction_type == TaskType.RECV:
                src = (layer2group[inst_trace.layer_id], inst_trace.src_id)
                dst = (layer2group[inst_trace.layer_id], inst_trace.dst_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.pe2pe(src=src, dst=dst, start_time=inst_trace.start_time, exe_time=exe_time/inst_trace.data_size, size=inst_trace.data_size)

            elif inst_trace.instruction_type == TaskType.READ:
                dst = (layer2group[inst_trace.layer_id], inst_trace.pe_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.dram2pe(dram_id=layer2group[inst_trace.layer_id]-1, dst=dst, start_time=inst_trace.start_time, exe_time=0.25, size=inst_trace.data_size)
            
            elif inst_trace.instruction_type == TaskType.WRITE:
                src = (layer2group[inst_trace.layer_id], inst_trace.pe_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.pe2dram(src=src, dram_id=layer2group[inst_trace.layer_id], start_time=inst_trace.start_time, exe_time=0.25, size=inst_trace.data_size)

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
            if failslow:
                self.mesh.mapping(self.node_info[edge.src.id], self.node_info[edge.dst.id], edge.sum(), True)
                edge.failslow.extend(failslow)
                self.failslow_edge.append(edge)
                # print(f"Layer {self.node_info[edge.src.id][0]}:: Path pe({self.node_info[edge.src.id][1]}) -> pe({self.node_info[edge.dst.id][1]}) failslow at time {failslow}.")
            else:
                self.mesh.mapping(self.node_info[edge.src.id], self.node_info[edge.dst.id], edge.sum(), False)

            next_node = edge.dst
            self.DFS(next_node, threshold)

    def LinkAnalyze(self, threshold = 2):
        for node in self.nodes.values():
            if len(node.in_edges) == 0:
                self.DFS(node, threshold)

        # todo: failslow spread

def calc_pe_flops(trace: List[CompInst], layer_mapping: List[int]):
    tot_flops = {}
    inst_num = {}
    start_time, end_time = {}, {}
    average_flops = []

    for id in layer_mapping:
        tot_flops[id] = 0
        inst_num[id] = 0

    for id, inst_trace in enumerate(trace):
        # 指令flops需求 / 执行时间 得到 单周期flops
        tot_flops[inst_trace.pe_id] += inst_trace.flops / (inst_trace.end_time - inst_trace.start_time)
        inst_num[inst_trace.pe_id] += 1
        if inst_trace.pe_id in start_time:
            start_time[inst_trace.pe_id] = min(inst_trace.start_time, start_time[inst_trace.pe_id])
        else:
            start_time[inst_trace.pe_id] = inst_trace.start_time
        if inst_trace.pe_id in end_time:
            end_time[inst_trace.pe_id] = max(inst_trace.end_time, end_time[inst_trace.pe_id])
        else:
            end_time[inst_trace.pe_id] = inst_trace.end_time

    for id in layer_mapping:
        average_flops.append(tot_flops[id] / inst_num[id])

    # print(f"average flops: {average_flops}")
    return average_flops, start_time, end_time

def link_failslow_detection():
    factor = 10000
    arch_configs = config_analyzer("arch/gemini4_4.json")
    baseline_trace = comm_analyzer("data/darknet19/normal/comm_trace.json")
    detection_trace = comm_analyzer("data/darknet19/link/comm_trace.json")
    
    # 建立基准
    baseline = NoCTomography(arch_configs.core.x, arch_configs.core.y, factor)
    baseline_paths = []
    for inst_trace in baseline_trace.trace:
        if inst_trace.instruction_type == TaskType.RECV:
            exe_time = inst_trace.end_time - inst_trace.start_time
            baseline_paths.append((inst_trace.src_id, inst_trace.dst_id, exe_time/inst_trace.data_size*factor))
    baseline_bandwidth = baseline.train(baseline_paths)
    
    detection = NoCTomography(arch_configs.core.x, arch_configs.core.y, factor)
    detection_paths = []
    for inst_trace in detection_trace.trace:
        if inst_trace.instruction_type == TaskType.RECV:
            exe_time = inst_trace.end_time - inst_trace.start_time
            detection_paths.append((inst_trace.src_id, inst_trace.dst_id, exe_time/inst_trace.data_size*factor))
    detection_bandwidth = detection.train(detection_paths)

    for link in baseline_bandwidth.keys():
        print(f"Link_{link[0]}_{link[1]}: baseline: {baseline_bandwidth[link]} B/cycle, detection: {detection_bandwidth[link]} B/cycle.")

if __name__ == '__main__':
    net = json_analyzer("tests/darknet19/mapping.json")
    arch_configs = config_analyzer("arch/gemini4_4.json")
    comm_trace = comm_analyzer("data/darknet19/router/comm_trace.json")
    comp_trace = comp_analyzer("data/darknet19/tpu/comp_trace.json")

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

    # 处理comp指令数据
    comp_trace_layer = [[] for _ in range(len(net.layers))]

    for inst_trace in comp_trace.trace:
        comp_trace_layer[inst_trace.layer_id].append(inst_trace)

    mesh = Mesh(group_num=len(layer_group_divide), x=arch_configs.core.x, y=arch_configs.core.y)

    # 初始化结点故障概率
    for layer_trace in comp_trace_layer:
        layer_id = layer_trace[0].layer_id
        average_flops, start_time, end_time = calc_pe_flops(layer_trace, layer_mapping[layer_id])
        # 为当前层的每个pe进行初始化
        for id, pe_id in enumerate(layer_mapping[layer_id]):
            mesh.core_prob_init(layer_group=layer2group[layer_id], pe_id=pe_id, flops=average_flops[id],
                                start_time=start_time[pe_id], end_time=end_time[pe_id])

    comm_graph = CommGraph(comm_trace.trace, mesh)
    print(f"Finish building comm_graph, there are {len(comm_graph.nodes)} nodes and {len(comm_graph.edges)} edges in the graph.")
    
    link_failslow_detection()

    # comm_graph.debug()

    # 失速路径分析
    # comm_graph.LinkAnalyze(threshold=2)

    # # 物理链路回溯
    # for failslow_edge in comm_graph.failslow_edge:
    #     # 链路图中dst_id
    #     dst_id = comm_graph.node_info[failslow_edge.dst.id][0]*core_num + comm_graph.node_info[failslow_edge.dst.id][1]
    #     print(f"backtracking start from {comm_graph.node_info[failslow_edge.dst.id]}")
    #     mesh.backtracking(curNode=dst_id, father=-1, failslow=failslow_edge.failslow, step=0, limit=1)

    # mesh.summary()
    
    # pagerank RCA
    # 利用检测的失速路径初始化链路故障概率
    # print("before pagerank")
    # mesh.pagerank_summary()
    # mesh.link_prob_init()
    # mesh.pagerank()
    # print("after pagerank")
    # mesh.pagerank_summary()