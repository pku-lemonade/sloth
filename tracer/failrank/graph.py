from evaluater.sim_type import TaskType
from tracer.failrank.stats import calc_window, interval_merge


class Node:
    def __init__(self, node_id: int):
        self.id = node_id
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
            return (0, 0)
        return min(event[0] for event in self.events), max(event[0] + event[1] for event in self.events)

    def sum(self):
        return sum(event[2] for event in self.events)

    def start_time_range(self):
        if not self.events:
            return None
        times = [event[0] for event in self.events]
        return min(times), max(times)

    def get_time_window(self, window_size: int, step: int):
        if not self.events:
            return []

        self.events.sort()
        windows = []
        cur_pos = 0
        while cur_pos < len(self.events):
            window_events = []
            for event_num in range(window_size):
                if cur_pos + event_num >= len(self.events):
                    break
                window_events.append(self.events[cur_pos + event_num][1])
            windows.append((self.events[cur_pos][0], window_events))
            cur_pos += step
        return windows

    def failslow_detect(self, window_size, step, threshold):
        windows = self.get_time_window(window_size=window_size, step=step)
        stats = calc_window(windows)
        failslow = []
        fail_start = -1
        for idx in range(1, len(stats)):
            t0, avg0, _ = stats[idx - 1]
            t1, avg1, _ = stats[idx]
            if avg0 > 0 and avg1 / avg0 > threshold and fail_start == -1:
                fail_start = t0
            if avg1 > 0 and avg0 / avg1 > threshold and fail_start != -1:
                failslow.append((fail_start, t1))
                fail_start = -1

        if fail_start != -1 and windows:
            failslow.append((fail_start, windows[-1][0]))
        if not failslow:
            return []
        return interval_merge(failslow)


class CommGraph:
    def __init__(self, context, trace, mesh):
        self.context = context
        self.mesh = mesh
        self.failslow_edge = []
        self.nodes = {}
        self.node_id = {}
        self.node_info = {}
        self.edges = {}
        self.vis = {}
        self.node_num = 0
        self.build_graph(trace)

    def pe2pe(self, src, dst, start_time: int, exe_time: float, size: int):
        if src not in self.nodes:
            self.node_num += 1
            self.node_id[src] = self.node_num
            self.node_info[self.node_num] = src
            self.nodes[src] = PENode(node_id=self.node_num, layer_group_id=src[0])

        if dst not in self.nodes:
            self.node_num += 1
            self.node_id[dst] = self.node_num
            self.node_info[self.node_num] = dst
            self.nodes[dst] = PENode(node_id=self.node_num, layer_group_id=dst[0])

        if (self.node_id[src], self.node_id[dst]) not in self.edges:
            edge = DepEdge(src=self.nodes[src], dst=self.nodes[dst])
            edge.insert(start_time=start_time, exe_time=exe_time, size=size)
            self.edges[(self.node_id[src], self.node_id[dst])] = edge
            self.nodes[src].out_edges.append(edge)
            self.nodes[dst].in_edges.append(edge)
        else:
            self.edges[(self.node_id[src], self.node_id[dst])].insert(start_time=start_time, exe_time=exe_time, size=size)

    def pe2dram(self, src, dram_id: int, start_time: int, exe_time: float, size: int):
        if src not in self.nodes:
            self.node_num += 1
            self.node_id[src] = self.node_num
            self.node_info[self.node_num] = src
            self.nodes[src] = PENode(node_id=self.node_num, layer_group_id=src[0])

        dram_node = (dram_id, -1)
        if dram_node not in self.nodes:
            self.node_num += 1
            self.node_id[dram_node] = self.node_num
            self.node_info[self.node_num] = dram_node
            self.nodes[dram_node] = DRAMNode(node_id=self.node_num, interval_id=dram_id)

        edge_key = (self.node_id[src], self.node_id[dram_node])
        if edge_key not in self.edges:
            edge = DepEdge(src=self.nodes[src], dst=self.nodes[dram_node])
            edge.insert(start_time=start_time, exe_time=exe_time, size=size)
            self.edges[edge_key] = edge
            self.nodes[src].out_edges.append(edge)
            self.nodes[dram_node].in_edges.append(edge)
        else:
            self.edges[edge_key].insert(start_time=start_time, exe_time=exe_time, size=size)

    def dram2pe(self, dram_id: int, dst, start_time: int, exe_time: float, size: int):
        dram_node = (dram_id, -1)
        if dram_node not in self.nodes:
            self.node_num += 1
            self.node_id[dram_node] = self.node_num
            self.node_info[self.node_num] = dram_node
            self.nodes[dram_node] = DRAMNode(node_id=self.node_num, interval_id=dram_id)

        if dst not in self.nodes:
            self.node_num += 1
            self.node_id[dst] = self.node_num
            self.node_info[self.node_num] = dst
            self.nodes[dst] = PENode(node_id=self.node_num, layer_group_id=dst[0])

        edge_key = (self.node_id[dram_node], self.node_id[dst])
        if edge_key not in self.edges:
            edge = DepEdge(src=self.nodes[dram_node], dst=self.nodes[dst])
            edge.insert(start_time=start_time, exe_time=exe_time, size=size)
            self.edges[edge_key] = edge
            self.nodes[dram_node].out_edges.append(edge)
            self.nodes[dst].in_edges.append(edge)
        else:
            self.edges[edge_key].insert(start_time=start_time, exe_time=exe_time, size=size)

    def build_graph(self, trace):
        if not trace:
            return

        for inst_trace in trace:
            if not hasattr(inst_trace, "instruction_type"):
                src = (self.context.layer_to_group[inst_trace.layer_id], inst_trace.src_id)
                dst = (self.context.layer_to_group[inst_trace.layer_id], inst_trace.dst_id)
                exe_time = inst_trace.avg_time
                for _ in range(getattr(inst_trace, "count", 1)):
                    self.pe2pe(src=src, dst=dst, start_time=inst_trace.start_time, exe_time=exe_time, size=inst_trace.data_size)
            elif inst_trace.instruction_type == TaskType.RECV:
                src = (self.context.layer_to_group[inst_trace.layer_id], inst_trace.src_id)
                dst = (self.context.layer_to_group[inst_trace.layer_id], inst_trace.dst_id)
                exe_time = inst_trace.end_time - inst_trace.start_time
                self.pe2pe(src=src, dst=dst, start_time=inst_trace.start_time, exe_time=exe_time / inst_trace.data_size, size=inst_trace.data_size)
            elif inst_trace.instruction_type == TaskType.READ:
                dst = (self.context.layer_to_group[inst_trace.layer_id], inst_trace.pe_id)
                self.dram2pe(
                    dram_id=self.context.layer_to_group[inst_trace.layer_id] - 1,
                    dst=dst,
                    start_time=inst_trace.start_time,
                    exe_time=0.25,
                    size=inst_trace.data_size,
                )
            elif inst_trace.instruction_type == TaskType.WRITE:
                src = (self.context.layer_to_group[inst_trace.layer_id], inst_trace.pe_id)
                self.pe2dram(
                    src=src,
                    dram_id=self.context.layer_to_group[inst_trace.layer_id],
                    start_time=inst_trace.start_time,
                    exe_time=0.25,
                    size=inst_trace.data_size,
                )

    def _dfs(self, cur_node, threshold):
        key = self.node_info[cur_node.id]
        if key in self.vis:
            return

        self.vis[key] = 1
        for edge in cur_node.out_edges:
            window_size = len(edge.events) // 10
            step = max(window_size // 2, 1)
            failslow = edge.failslow_detect(window_size, step, threshold)

            if failslow:
                self.mesh.mapping(self.node_info[edge.src.id], self.node_info[edge.dst.id], edge.sum(), edge.range(), True)
                edge.failslow.extend(failslow)
                self.failslow_edge.append(edge)
            else:
                self.mesh.mapping(self.node_info[edge.src.id], self.node_info[edge.dst.id], edge.sum(), edge.range(), False)

            self._dfs(edge.dst, threshold)

    def construct_mesh(self, threshold=2):
        for node in self.nodes.values():
            if len(node.in_edges) == 0:
                self._dfs(node, threshold)
