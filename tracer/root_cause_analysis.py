import os
import sys
import json
import math
import argparse
import numpy as np
from typing import List
from scipy.stats import norm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from recorder.data_structure import *
from recorder.trace_format import *
from pydantic import ValidationError, BaseModel
from evaluater.sim_type import TaskType
from common.distribution import CoreDist
from tracer.tomography import NoCTomography
from common.prediction import EM_Model

from compiler.instruction_generator import json_analyzer, config_analyzer

parser = argparse.ArgumentParser()

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

def comp_analyzer(filename: str) -> CompTrace:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = CompTrace.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())

def get_id(x: int, y: int):
    return x * arch_configs.core.y + y

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
        self.events = []
        self.failslow = []

    def insert(self, start_time: int, exe_time: float, size: int):
        self.events.append((start_time, exe_time, size))

    def range(self):
        if not self.events:
            return (0,0)
        res = (10000000000, 0)
        for event in self.events:
            res = (min(res[0], event[0]), max(res[1], event[0]+event[1]))
        return res

    def sum(self):
        if not self.events:
            return 0
        res = 0
        for event in self.events:
            res += event[2]
        return res

    def start_time_range(self):
        if not self.events:
            return None
        
        times = [event[0] for event in self.events]
        return min(times), max(times)
    
    def get_time_window(self, window_size: int, step: int):
        if not self.events:
            return []

        self.events.sort()
        start_time, end_time = self.start_time_range()

        windows = []
        cur_pos = 0
        
        while cur_pos < len(self.events):
            event_num = 0
            window_events = []

            while event_num < window_size:
                if cur_pos + event_num >= len(self.events):
                    break
                window_events.append(self.events[cur_pos+event_num][1])
                event_num += 1

            windows.append((self.events[cur_pos][0], window_events))
            cur_pos += step

        return windows
    
    def failslow_detect(self, window_size, step, threshold):
        windows = self.get_time_window(window_size=window_size, step=step)
        stats = calc_window(windows)
        failslow = []
        fail_start = -1
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
            failslow.append((fail_start, windows[-1][0]))

        if not failslow:
            return []

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

def sigmoid(x, offset = 0.7, beta = 50):
    e_x = math.exp(-beta * (x - offset))
    return 1 / (1 + e_x)

class FailSlow(BaseModel):
    kind: str
    id: int
    dst_id: int = -1
    start_time: int = 0
    end_time: int = 0

class FailSlows(BaseModel):
    data: List[FailSlow] = []

    def insert(self, new_failure: FailSlow):
        repeat = False
        match new_failure.kind:
            case "pe":
                for failure in self.data:
                    if failure.id == new_failure.id:
                        failure.start_time = min(failure.start_time, new_failure.start_time)
                        failure.end_time = max(failure.end_time, new_failure.end_time)
                        repeat = True
                        break
            case "link":
                if new_failure.id > new_failure.dst_id:
                    new_failure.id, new_failure.dst_id = new_failure.dst_id, new_failure.id
                
                for failure in self.data:
                    if failure.id == new_failure.id and failure.dst_id == new_failure.dst_id:
                        failure.start_time = min(failure.start_time, new_failure.start_time)
                        failure.end_time = max(failure.end_time, new_failure.end_time)
                        repeat = True
                        break

        if not repeat:
            self.data.append(new_failure)

global_link_count = np.zeros((16, 16))

class Mesh:
    def __init__(self, group_num: int, x: int, y: int):
        self.group_num = group_num
        self.x = x
        self.y = y
        self.num = self.x * self.y
        self.N = self.group_num * self.num + 1
        self.time_range = [(0, 0) for _ in range(self.N)]
        self.link_time_range = {}

        self.core_dist = CoreDist(mu=1024, sigma=100)
        
        self.edges = [[] for _ in range(self.N)]
        self.mp = [{} for _ in range(self.N)]
        self.count = [{} for _ in range(self.N)]
        self.link_count = np.zeros((self.N, self.N))
        self.link_size_count = np.zeros((self.N, self.N))

        self.detector = FailSlowDetector(threshold=0.1)
        self.core_failslow_prob = np.zeros(self.N)
        self.link_failslow_prob = np.zeros((self.N, self.N))
        self.transition_matrix = np.zeros((self.N, self.N))
        self.link_variance = np.zeros((self.num, self.num))

    def mapping(self, src, dst, size, range, failslow):
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
        
        src_x = src[1] // self.y
        src_y = src[1] % self.y
        dst_x = dst[1] // self.y
        dst_y = dst[1] % self.y

        while src_x != dst_x:
            if dst_x > src_x:
                src_x += 1
                curNode = (src[0]-1)*self.x*self.y+src_x*self.y+src_y
                nextNode = (src[0]-1)*self.x*self.y+(src_x-1)*self.y+src_y

                if failslow:
                    global_link_count[curNode%self.num][nextNode%self.num] += 1
                    global_link_count[nextNode%self.num][curNode%self.num] += 1

                if nextNode not in self.mp[curNode]:
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
                    self.link_size_count[curNode][nextNode] += size
                    self.link_size_count[nextNode][curNode] += size
                    if failslow:
                        self.link_count[curNode][nextNode] += 1
                        self.link_count[nextNode][curNode] += 1
                if curNode > nextNode:
                    curNode, nextNode = nextNode, curNode
                if (curNode,nextNode) not in self.link_time_range:
                    self.link_time_range[(curNode,nextNode)] = range
                else:
                    self.link_time_range[(curNode,nextNode)] = (
                        min(self.link_time_range[(curNode,nextNode)][0], range[0]),
                        max(self.link_time_range[(curNode,nextNode)][1], range[1])
                    )
            else:
                src_x -= 1
                curNode = (src[0]-1)*self.x*self.y+src_x*self.y+src_y
                nextNode = (src[0]-1)*self.x*self.y+(src_x+1)*self.y+src_y

                if failslow:
                    global_link_count[curNode%self.num][nextNode%self.num] += 1
                    global_link_count[nextNode%self.num][curNode%self.num] += 1

                if nextNode not in self.mp[curNode]:
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
                    self.link_size_count[curNode][nextNode] += size
                    self.link_size_count[nextNode][curNode] += size
                    if failslow:
                        self.link_count[curNode][nextNode] += 1
                        self.link_count[nextNode][curNode] += 1
                if curNode > nextNode:
                    curNode, nextNode = nextNode, curNode
                if (curNode,nextNode) not in self.link_time_range:
                    self.link_time_range[(curNode,nextNode)] = range
                else:
                    self.link_time_range[(curNode,nextNode)] = (
                        min(self.link_time_range[(curNode,nextNode)][0], range[0]),
                        max(self.link_time_range[(curNode,nextNode)][1], range[1])
                    )
        
        while src_y != dst_y:
            if dst_y > src_y:
                src_y += 1
                curNode = (src[0]-1)*self.x*self.y+src_x*self.y+src_y
                nextNode = (src[0]-1)*self.x*self.y+src_x*self.y+(src_y-1)

                if failslow:
                    global_link_count[curNode%self.num][nextNode%self.num] += 1
                    global_link_count[nextNode%self.num][curNode%self.num] += 1

                if nextNode not in self.mp[curNode]:
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
                    self.link_size_count[curNode][nextNode] += size
                    self.link_size_count[nextNode][curNode] += size
                    if failslow:
                        self.link_count[curNode][nextNode] += 1
                        self.link_count[nextNode][curNode] += 1
                if curNode > nextNode:
                    curNode, nextNode = nextNode, curNode
                if (curNode,nextNode) not in self.link_time_range:
                    self.link_time_range[(curNode,nextNode)] = range
                else:
                    self.link_time_range[(curNode,nextNode)] = (
                        min(self.link_time_range[(curNode,nextNode)][0], range[0]),
                        max(self.link_time_range[(curNode,nextNode)][1], range[1])
                    )
            else:
                src_y -= 1
                curNode = (src[0]-1)*self.x*self.y+src_x*self.y+src_y
                nextNode = (src[0]-1)*self.x*self.y+src_x*self.y+(src_y+1)

                if failslow:
                    global_link_count[curNode%self.num][nextNode%self.num] += 1
                    global_link_count[nextNode%self.num][curNode%self.num] += 1
                    
                if nextNode not in self.mp[curNode]:
                    self.edges[curNode].append((nextNode, []))
                    self.mp[curNode][nextNode] = len(self.edges[curNode])-1
                    self.link_size_count[curNode][nextNode] += size
                    self.link_size_count[nextNode][curNode] += size
                    if failslow:
                        self.link_count[curNode][nextNode] += 1
                        self.link_count[nextNode][curNode] += 1
                if curNode > nextNode:
                    curNode, nextNode = nextNode, curNode
                if (curNode,nextNode) not in self.link_time_range:
                    self.link_time_range[(curNode,nextNode)] = range
                else:
                    self.link_time_range[(curNode,nextNode)] = (
                        min(self.link_time_range[(curNode,nextNode)][0], range[0]),
                        max(self.link_time_range[(curNode,nextNode)][1], range[1])
                    )


    def link_variance_init(self, variance):
        for item in variance:
            self.link_variance[item[1]][item[2]] = item[3]
            self.link_variance[item[2]][item[1]] = item[3]

    def link_prob_init(self):
        max_fail_count = self.link_count.max()
        for i in range(self.N):
            path_num = self.link_size_count[i].sum()
            if path_num == 0:
                continue
            for j in range(self.N):
                self.transition_matrix[i][j] = (self.link_size_count[i][j] / path_num)
                if max_fail_count != 0:
                    self.link_failslow_prob[i][j] = self.link_count[i][j] / max_fail_count
                else:
                    self.link_failslow_prob[i][j] = 0 

    def core_prob_init(self, layer_group, pe_id, flops, start_time, end_time):
        core_id = (layer_group-1) * self.num + pe_id
        self.core_failslow_prob[core_id] = self.core_dist.failslow_prob(flops)
        self.time_range[core_id] = (start_time, end_time)

    def failrank(self, alpha = 0.6, tol = 1e-4, max_iter = 1000):
        jump_prob = self.core_failslow_prob

        for _ in range(max_iter):
            new_prob = alpha * self.transition_matrix.T @ self.core_failslow_prob 
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
            nextNode[1].extend(failslow)
            
            if nextNode[0] not in self.count[curNode]:
                self.count[curNode][nextNode[0]] = 1
            else:
                self.count[curNode][nextNode[0]] += 1
            self.backtracking(curNode=nextNode[0], father=curNode, failslow=failslow, step=step+1, limit=limit)
    
    def check(self, x: int, y: int):
        return y-x == self.y or y-x == 1

    def link_summary(self, threshold=0.8):
        failslow = FailSlows()
        links = []
        values = []

        for src in range(self.num):
            for dst in range(src+1, self.num):
                if not self.check(src, dst):
                    continue
                values.append(global_link_count[src][dst])
                links.append((src, dst))                    

        softmax_values = softmax(values, beta=1)

        for id in range(len(softmax_values)):
            if softmax_values[id] > threshold:
                failslow.insert(FailSlow(kind="link", id=links[id][0], dst_id=links[id][1]))

        return failslow

    def failrank_summary(self, k=30, threshold=0.65):
        failslow = FailSlows()
        for group_id in range(self.group_num):
            start_id = group_id * self.num
            end_id = start_id + self.num

            group_prob = softmax(self.core_failslow_prob[start_id:end_id], beta=200)

            for id in range(self.num):
                if group_prob[id] < threshold:
                    continue
                node_id = start_id + id
                print(f"[FailSlow-PE] Id: {node_id%self.num} Duration: [{self.time_range[node_id][0]},{self.time_range[node_id][1]}] Prob: {group_prob[id]*100:.2f}%.")
                failslow.insert(FailSlow(kind="pe", id=node_id%self.num, start_time=self.time_range[node_id][0], end_time=self.time_range[node_id][1]))

            self.link_failslow_prob[:] = 0
            for j in range(start_id, end_id):
                for i in range(self.N):
                    if self.link_count.max() != 0:
                        self.link_failslow_prob[i][j] = self.link_count[i][j] / self.link_count.max() * 10
                    else:
                        self.link_failslow_prob[i][j] = 0

                    src_core_id = i % self.num
                    dst_core_id = j % self.num
                    if (src_core_id > dst_core_id):
                        src_core_id, dst_core_id = dst_core_id, src_core_id

                    self.link_failslow_prob[i][j] += self.link_variance[src_core_id][dst_core_id] * 60
                    self.link_failslow_prob[i][j] += group_prob[j%self.num] * 30

            flat = self.link_failslow_prob.flatten()
            group_prob = softmax(flat, beta=1)
            group_prob = group_prob.reshape(self.link_failslow_prob.shape)

            maxprob = np.max(group_prob)
            failslow_link = np.unravel_index(np.argmax(group_prob), self.link_failslow_prob.shape)

            if maxprob < threshold:
                continue

            if failslow_link[0]%self.num > failslow_link[1]%self.num:
                failslow_link = (failslow_link[1], failslow_link[0])

            if failslow_link not in self.link_time_range:
                continue
            fail_range = self.link_time_range[failslow_link]
            print(f"[FailSlow-Link] Id: {failslow_link[0]%self.num}-{failslow_link[1]%self.num} Duration: [{fail_range[0]},{fail_range[1]}]")
            failslow.insert(FailSlow(kind="link", id=failslow_link[0]%self.num, dst_id=failslow_link[1]%self.num))

        return failslow

    def summary(self):
        merged_count = {}
        failslow_interval = {}
        for id, count in enumerate(self.count):
            for nextNode, cnt in count.items():
                src = id % self.num
                dst = nextNode % self.num

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

        for src, dst, failslow in max_count_links:
            failslow.sort()
            failslow = interval_merge(failslow)
            print(f"Link pe({src}) - pe({dst}) failslow at time {failslow}.")

class CommGraph:
    def __init__(self, trace, mesh):
        self.node_num = 0
        self.mesh = mesh
        self.failslow_edge = []
        self.nodes = {}
        self.node_id = {}
        self.node_info = {}
        self.edges = {}

        self.vis = {}

        self.build_graph(trace)

    def debug(self):
        for id, edge in enumerate(self.edges.values()):
            print(f"edge {self.node_info[edge.src.id]} -> {self.node_info[edge.dst.id]}")
            print(edge.events)

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

    def build_graph(self, trace):
        if isinstance(trace[0], CommInst):
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
        else:
            for inst_trace in trace:
                src = (layer2group[inst_trace.layer_id], inst_trace.src_id)
                dst = (layer2group[inst_trace.layer_id], inst_trace.dst_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.pe2pe(src=src, dst=dst, start_time=inst_trace.start_time, exe_time=exe_time/inst_trace.data_size, size=inst_trace.data_size)
    
    
    def DFS(self, curNode, threshold):
        if (self.node_info[curNode.id][0], self.node_info[curNode.id][1]) in self.vis:
            return
        
        self.vis[(self.node_info[curNode.id][0], self.node_info[curNode.id][1])] = 1
        for edge in curNode.out_edges:
            window_size = len(edge.events) // 10
            step = max(window_size // 2, 1)
            failslow = edge.failslow_detect(window_size, step, threshold)

            if failslow:
                self.mesh.mapping(self.node_info[edge.src.id], self.node_info[edge.dst.id], edge.sum(), edge.range(), True)
                edge.failslow.extend(failslow)
                self.failslow_edge.append(edge)
            else:
                self.mesh.mapping(self.node_info[edge.src.id], self.node_info[edge.dst.id], edge.sum(), edge.range(), False)

            next_node = edge.dst
            self.DFS(next_node, threshold)

    def ConstructMesh(self, threshold = 2):
        for node in self.nodes.values():
            if len(node.in_edges) == 0:
                self.DFS(node, threshold)


def core_level_detection(trace, layer_mapping: List[int]):
    tot_flops = {}
    inst_num = {}
    start_time, end_time = {}, {}
    average_flops = []

    for id in layer_mapping:
        tot_flops[id] = 0
        inst_num[id] = 0

    for id, inst_trace in enumerate(trace):
        
        if isinstance(inst_trace, CompInst):
            tot_flops[inst_trace.pe_id] += inst_trace.flops / (inst_trace.end_time - inst_trace.start_time)
        else:
            tot_flops[inst_trace.pe_id] += inst_trace.flops

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

    return average_flops, start_time, end_time

def link_analyzer(filename: str) -> LinksData:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = LinksData.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())

def layer_group_analyzer(filename: str) -> LayerGroupsInfo:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = LayerGroupsInfo.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())
bw = {}
def link_level_detection(inference_time, ground_truth = False, file_path = None, data_compress = None):
    detection_trace = None

    if file_path is not None:
        detection_trace = comm_analyzer(file_path)
    else:
        detection_trace = data_compress

    inference_paths = [[] for _ in range(inference_time)]

    for inst_trace in detection_trace.trace:
        if file_path is not None:
            if inst_trace.instruction_type == TaskType.RECV:
                exe_time = inst_trace.end_time - inst_trace.start_time
                inference_paths[inst_trace.inference_time].append(
                    (inst_trace.data_size, exe_time, inst_trace.src_id, inst_trace.dst_id)
                )
        else:
            exe_time = inst_trace.end_time - inst_trace.start_time
            inference_paths[inst_trace.inference_time].append(
                (inst_trace.data_size, exe_time, inst_trace.src_id, inst_trace.dst_id)
            )

    failslow_links = []

    my_EM_Model = None

    for time in range(inference_time):
        model = NoCTomography(arch_configs.core.x, arch_configs.core.y, factor=1)

        X, y = model.build_feature_matrix_new(inference_paths[time])

        bandwidth, c2rbandwidth, node_latency, startup_time = model.solve(X, y)

        for link in bandwidth.keys():
            if bandwidth[link] == 0:
                continue

            true_bandwidth = 1 / bandwidth[link]
            if ground_truth is True:
                bw[(link[0], link[1])] = true_bandwidth
            else:
                if (link[0], link[1]) in bw:
                    variance = bw[(link[0], link[1])] / true_bandwidth
                    if variance > 5:
                        failslow_links.append((time, link[0], link[1], variance))
                        print(f"[Inference {time}] Link_{link[0]}_{link[1]}: {true_bandwidth:.4f} [{bw[(link[0], link[1])]:.4f}] B/cycle.")
                        print(f"[FailSlow] Bandwidth variance is {variance*100:.2f}%.")

    return failslow_links

def compress(comm_trace, comp_trace, num_hashes=5, num_buckets=1024, stage2_size=8192, threshold=10):
    ds = FailSlowCompressor(num_hashes=num_hashes, num_buckets=num_buckets, stage2_size=stage2_size, threshold=threshold)
    
    for trace in comm_trace.trace:
        if trace.instruction_type not in io_inst:
            key, attr = trace_to_key_attr(trace)
            ds.insert(key, attr.start_time, attr.end_time, attr)
    
    for trace in comp_trace.trace:
        if trace.instruction_type not in io_inst:
            key, attr = trace_to_key_attr(trace)
            ds.insert(key, attr.start_time, attr.end_time, attr)
    return ds.summaries()

class record_overhead(BaseModel):
    stage1 : dict = {"d" : 1,
                    "m" : 1,
                    "H" : 1,
                    "tables_key" : 0,
                    "tables_count" : 1}
    stage2 : dict = {
                        "max_size" : 1,
                        "table_key" : 0,
                        "table_value_s_time" : 1,
                        "table_value_e_time" : 1,
                        "table_value_count" : 1,
                        "table_value_attr" : {"layer_id" : 1,
                                                    "pe_id" : 1,
                                                    "start_time" : 1,
                                                    "end_time" : 1,
                                                    "inference_time" : 1,
                                                    "flops" : 1,
                                                    "data_size" : 1,
                                                    "src_id" : 1,
                                                    "dst_id" : 1,
                                                    "duration" : 1}
                    }
    
    def to_dict(self):
        return {
            "stage1": self.stage1,
            "stage2": self.stage2
        }

def update_overhead(old_record,stage1_overhead,stage2_overhead)-> record_overhead:
    record=record_overhead()
    record.stage1={key: max(old_record.stage1[key], stage1_overhead[key]) for key in old_record.stage1}
    record.stage2={key: max(old_record.stage2[key], stage2_overhead[0][key]) for key in stage2_overhead[0]}
    old_attr = old_record.stage2["table_value_attr"]
    new_attr = {key: max(old_attr[key], stage2_overhead[1][key]) for key in stage2_overhead[1]}
    record.stage2.update({"table_value_attr" : new_attr})
    return record

def update_compressed_summary(compressed_summary,detect_compress):
    for item in detect_compress.trace:
        compressed_summary["pe_id"]=max(compressed_summary["pe_id"],_size_of_int(item.pe_id))
        compressed_summary["layer_id"]=max(compressed_summary["layer_id"],_size_of_int(item.layer_id))
        compressed_summary["start_time"]=max(compressed_summary["start_time"],_size_of_int(item.start_time))
        compressed_summary["end_time"]=max(compressed_summary["end_time"],_size_of_int(item.end_time))
        compressed_summary["inference_time"]=max(compressed_summary["inference_time"],_size_of_int(item.inference_time))

def update_compressed_comm(compressed_comm,detect_comm_compress):
    update_compressed_summary(compressed_comm,detect_comm_compress)
    for item in detect_comm_compress.trace:
        compressed_comm["data_size"]=max(compressed_comm["data_size"],_size_of_int(item.data_size))
        compressed_comm["src_id"]=max(compressed_comm["src_id"],_size_of_int(item.src_id))
        compressed_comm["dst_id"]=max(compressed_comm["dst_id"],_size_of_int(item.dst_id))
    compressed_comm["trace_sum"]=max(compressed_comm["trace_sum"],len(detect_comm_compress.trace))
    
def update_compressed_comp(compressed_comp,detect_comp_compress):
    update_compressed_summary(compressed_comp,detect_comp_compress)
    compressed_comp["trace_sum"]=max(compressed_comp["trace_sum"],len(detect_comp_compress.trace))
    
def calc_overhead(record,hash,bucket,size):
    res = record.stage1["d"] + record.stage1["m"] + record.stage1["H"]
    res += (record.stage1["tables_key"] + record.stage1["tables_count"]) * hash * bucket
    res += record.stage2["max_size"]
    tmp = 0
    for value in record.stage2["table_value_attr"].values():
        tmp += value
    res += (record.stage2["table_key"] + record.stage2["table_value_s_time"] 
            + record.stage2["table_value_e_time"] + record.stage2["table_value_count"] + tmp) * size
    return res/(1024*1024)
    
def calc_trace_size(trace_size):
    tmp = sum(trace_size.values()) - trace_size["trace_sum"]
    tmp *= trace_size["trace_sum"]
    return tmp/(1024*1024)

def _size_of_int(value):
    if value < 256:
        return 1 
    elif value < 65536:
        return 2  
    elif value < 4294967296:
        return 4  
    else:
        return 8 
    
if __name__ == '__main__':

    parser.add_argument("--mapping", type=str, help="Workload mapping file")
    parser.add_argument("--arch", type=str, help="Architecture configuration file")
    parser.add_argument("--report", type=str, help="Report file")

    parser.add_argument("--normal", type=str, help="Path to trace without fail-slow")
    parser.add_argument("--detect", type=str, help="Path to real runtime trace")

    parser.add_argument("--hash", type=int, default=5, help="Fail-Slow sketch parameter")
    parser.add_argument("--bucket", type=int, default=1024, help="Fail-Slow sketch parameter")
    parser.add_argument("--size", type=int, default=8192, help="Fail-Slow sketch parameter")
    parser.add_argument("--threshold", type=int, default=10, help="Fail-Slow sketch parameter")

    parser.add_argument("--output", type=str, help="Path to record the max total overhead")
    parser.add_argument("--record", type=str, help="Path to record the max overhead for each variable")

    args = parser.parse_args()

    net = json_analyzer(args.mapping)
    arch_configs = config_analyzer(args.arch)

    normal_comm_path = os.path.join(args.normal, "comm_trace.json")
    normal_comm_trace = comm_analyzer(normal_comm_path)
    normal_comp_path = os.path.join(args.normal, "comp_trace.json")
    normal_comp_trace = comp_analyzer(normal_comp_path)

    detect_comm_path = os.path.join(args.detect, "comm_trace.json")
    detect_comm_trace = comm_analyzer(detect_comm_path)
    detect_comp_path = os.path.join(args.detect, "comp_trace.json")
    detect_comp_trace = comp_analyzer(detect_comp_path)


    core_num = arch_configs.core.x * arch_configs.core.y
    layer_mapping = [[] for _ in range(len(net.layers))]
    layer_group_start_time = {}
    layer_group_end_time = {}

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

    group_layers = []
    for ind in range(cur_layer_id, len(net.layers)):
        layer2group[ind] = net.layers[cur_layer_id].layer_group_id
        group_layers.append(ind)

    cur_layer_id = id
    layer_group_divide.append(group_layers)
    
    
    
    if not os.path.isfile(args.record):
        max_overhead = record_overhead()
        max_compressed_comm = {
            "pe_id": 1,
            "layer_id": 1,
            "start_time": 1,
            "end_time": 1,
            "inference_time": 1,
            "avg_time": 4,
            "data_size": 1,
            "src_id": 1,
            "dst_id": 1,
            "trace_sum": 0
        }
        max_compressed_comp = {
            "pe_id": 1,
            "layer_id": 1,
            "start_time": 1,
            "end_time": 1,
            "inference_time": 1,
            "flops": 4,
            "trace_sum": 0
        }
    else:
        with open(args.record, "r") as f:
            json_string = f.read()
        data = json.loads(json_string)
        max_overhead = record_overhead.model_validate(data["data_structure"])
        max_compressed_comm = data["compressed_comm"]
        max_compressed_comp = data["compressed_comp"]


    total_overhead=0
    total_comm_trace_size=2.861328125
    total_comp_trace_size=1.767578125
    total_compressed_comm_size=0
    total_compressed_comp_size=0
    compress_rate=1
    if os.path.isfile(args.output) :
        with open(args.output, "r") as f:
            json_string = f.read()
        data = json.loads(json_string)
        if "overhead" in data.keys():
            total_overhead = data["overhead"]
            total_compressed_comm_size = data["compressed_comm_size"]
            total_compressed_comp_size = data["compressed_comp_size"]
            compress_rate = data["rate"]
        
    
    normal_comm_compress, normal_comp_compress, _ , _  = compress(normal_comm_trace, normal_comp_trace, args.hash, args.bucket, args.size, args.threshold)
    detect_comm_compress, detect_comp_compress, stage1_overhead, stage2_overhead = compress(detect_comm_trace, detect_comp_trace, args.hash, args.bucket, args.size, args.threshold)

    max_overhead = update_overhead(max_overhead, stage1_overhead, stage2_overhead)
    total_overhead = max(total_overhead,calc_overhead(max_overhead, args.hash, args.bucket, args.size))
    
    update_compressed_comm(max_compressed_comm,detect_comm_compress)
    update_compressed_comp(max_compressed_comp,detect_comp_compress)
    
    total_compressed_comm_size = max(total_compressed_comm_size,calc_trace_size(max_compressed_comm))
    total_compressed_comp_size = max(total_compressed_comp_size,calc_trace_size(max_compressed_comp))
    
    compress_rate = float((total_compressed_comm_size+total_compressed_comp_size)/(total_comm_trace_size+total_comp_trace_size))
    
    record={ "data_structure" : max_overhead.to_dict(),
             "compressed_comm" : max_compressed_comm,
             "compressed_comp" : max_compressed_comp
            }
    output_dict = { "overhead" : total_overhead,
                    "comm_trace_size" : total_comm_trace_size,
                    "comp_trace_size" : total_comp_trace_size,
                    "compressed_comm_size" : total_compressed_comm_size,
                    "compressed_comp_size" : total_compressed_comp_size,
                    "rate" : compress_rate
                   }
    with open(args.record, 'w') as f:
            json.dump(record, f, indent=4)
    with open(args.output, 'w') as f:
            json.dump(output_dict, f, indent=4)
    
    print("="*40)
    print("Detecting potential failslow links:")
    
    link_level_detection(inference_time=1, ground_truth=True, data_compress=normal_comm_compress)
    failslow_link = link_level_detection(inference_time=16, data_compress=detect_comm_compress)

    failslow_period = set()
    for link in failslow_link:
        failslow_period.add(link[0])

    failslows = FailSlows()
    for period in range(16):
        comp_trace_layer = [[] for _ in range(len(net.layers))]
        tag = [[False for __ in range(16)] for _ in range(200)]

        for inst_trace in detect_comp_compress.trace:
            if inst_trace.inference_time != period:
                continue

            comp_trace_layer[inst_trace.layer_id].append(inst_trace)
            tag[inst_trace.layer_id][inst_trace.pe_id] = True

        for inst_trace in detect_comp_trace.trace:
            if inst_trace.inference_time != period:
                continue
            if tag[inst_trace.layer_id][inst_trace.pe_id] is False:
                comp_trace_layer[inst_trace.layer_id].append(inst_trace)

        comm_trace_inference = []
        for inst_trace in detect_comm_trace.trace:
            if inst_trace.inference_time != period:
                continue
            comm_trace_inference.append(inst_trace)

        print("="*40)
        print("Building RCA dependency graph:")
        mesh = Mesh(group_num=len(layer_group_divide), x=arch_configs.core.x, y=arch_configs.core.y)
        comm_graph = CommGraph(comm_trace_inference, mesh)
        print(f"[Info] Finish building comm_graph, {len(comm_graph.nodes)} nodes and {len(comm_graph.edges)} edges in total.")
        comm_graph.ConstructMesh()
        print("[Info] Finish building dependency graph.")
        
        print("="*40)
        print("Initializing failrank values:")
        for layer_trace in comp_trace_layer:
            layer_id = layer_trace[0].layer_id
            average_flops, start_time, end_time = core_level_detection(layer_trace, layer_mapping[layer_id])
            for id, pe_id in enumerate(layer_mapping[layer_id]):
                mesh.core_prob_init(layer_group=layer2group[layer_id], pe_id=pe_id, flops=average_flops[id],
                                    start_time=start_time[pe_id], end_time=end_time[pe_id])
        print("[Info] Finish initializing core PR values.")

        mesh.link_prob_init()
        mesh.link_variance_init(failslow_link)
        print("[Info] Finish initializing link weights.")

        print("="*40)
        print("Root Cause Analysis:")
        mesh.failrank()
        failslow = mesh.failrank_summary()
        for fail in failslow.data:
            failslows.insert(fail)

    link_fail = mesh.link_summary()
    for fail in link_fail.data:
        failslows.insert(fail)

    with open(args.report, "w") as file:
        fail_json = failslows.model_dump_json(indent=4)
        print(fail_json, file=file)