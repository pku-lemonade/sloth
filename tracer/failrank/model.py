from typing import List

import numpy as np
from pydantic import BaseModel, Field

from common.distribution import CoreDist
from tracer.failrank.stats import softmax

DEFAULT_PE_SOFTMAX_BETA = 200.0
DEFAULT_LINK_SOFTMAX_BETA = 1.0
DEFAULT_LINK_SUMMARY_BETA = 1.0
DEFAULT_LINK_COUNT_WEIGHT = 10.0
DEFAULT_LINK_VARIANCE_WEIGHT = 60.0
DEFAULT_PE_PROB_WEIGHT = 30.0


class FailSlow(BaseModel):
    kind: str
    id: int
    dst_id: int = -1
    start_time: int = 0
    end_time: int = 0


class FailSlows(BaseModel):
    data: List[FailSlow] = Field(default_factory=list)

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


class Mesh:
    def __init__(self, context):
        self.context = context
        self.group_num = context.group_count
        self.num = context.topology.node_count
        self.N = self.group_num * self.num + 1
        self.time_range = [(0, 0) for _ in range(self.N)]
        self.link_time_range = {}

        self.core_dist = CoreDist(mu=1024, sigma=100)
        self.edges = [[] for _ in range(self.N)]
        self.mp = [{} for _ in range(self.N)]
        self.count = [{} for _ in range(self.N)]
        self.link_count = np.zeros((self.N, self.N))
        self.link_size_count = np.zeros((self.N, self.N))
        self.link_hit_count = np.zeros((self.num, self.num))
        self.core_failslow_prob = np.zeros(self.N)
        self.link_failslow_prob = np.zeros((self.N, self.N))
        self.transition_matrix = np.zeros((self.N, self.N))
        self.link_variance = np.zeros((self.num, self.num))

    def mapping(self, src, dst, size, time_range, failslow):
        if src[1] == -1:
            dst_x = dst[1] // self.y
            dst_y = dst[1] % self.y
            dst_id = (dst[0] - 1) * self.num + dst_x * self.y + dst_y
            self.link_size_count[self.N - 1][dst_id] += size
            self.link_size_count[dst_id][self.N - 1] += size
            if failslow:
                self.link_count[self.N - 1][dst_id] += 1
                self.link_count[dst_id][self.N - 1] += 1
            return

        if dst[1] == -1:
            src_x = src[1] // self.y
            src_y = src[1] % self.y
            src_id = (src[0] - 1) * self.num + src_x * self.y + src_y
            self.link_size_count[src_id][self.N - 1] += size
            self.link_size_count[self.N - 1][src_id] += size
            if failslow:
                self.link_count[src_id][self.N - 1] += 1
                self.link_count[self.N - 1][src_id] += 1
            return

        group_offset = (src[0] - 1) * self.num
        for link_src, link_dst in self.context.topology.route_links(src[1], dst[1]):
            cur_node = group_offset + link_src
            next_node = group_offset + link_dst
            self._record_link(cur_node, next_node, size, time_range, failslow)

    def _record_link(self, cur_node: int, next_node: int, size: int, time_range, failslow: bool) -> None:
        src_core_id = cur_node % self.num
        dst_core_id = next_node % self.num
        low_core_id, high_core_id = sorted((src_core_id, dst_core_id))

        if failslow:
            self.link_hit_count[low_core_id][high_core_id] += 1
            self.link_hit_count[high_core_id][low_core_id] += 1

        if next_node not in self.mp[cur_node]:
            self.edges[cur_node].append((next_node, []))
            self.mp[cur_node][next_node] = len(self.edges[cur_node]) - 1

        self.link_size_count[cur_node][next_node] += size
        self.link_size_count[next_node][cur_node] += size
        if failslow:
            self.link_count[cur_node][next_node] += 1
            self.link_count[next_node][cur_node] += 1

        low_node, high_node = sorted((cur_node, next_node))
        if (low_node, high_node) not in self.link_time_range:
            self.link_time_range[(low_node, high_node)] = time_range
        else:
            self.link_time_range[(low_node, high_node)] = (
                min(self.link_time_range[(low_node, high_node)][0], time_range[0]),
                max(self.link_time_range[(low_node, high_node)][1], time_range[1]),
            )

    def link_variance_init(self, candidates) -> None:
        for candidate in candidates:
            self.link_variance[candidate.src_id][candidate.dst_id] = candidate.score
            self.link_variance[candidate.dst_id][candidate.src_id] = candidate.score

    def link_prob_init(self):
        max_fail_count = self.link_count.max()
        for i in range(self.N):
            path_num = self.link_size_count[i].sum()
            if path_num == 0:
                continue
            for j in range(self.N):
                self.transition_matrix[i][j] = self.link_size_count[i][j] / path_num
                self.link_failslow_prob[i][j] = self.link_count[i][j] / max_fail_count if max_fail_count != 0 else 0

    def core_prob_init(self, layer_group, pe_id, flops, start_time, end_time):
        core_id = (layer_group - 1) * self.num + pe_id
        self.core_failslow_prob[core_id] = self.core_dist.failslow_prob(flops)
        self.time_range[core_id] = (start_time, end_time)

    def failrank(self, alpha=0.6, tol=1e-4, max_iter=1000):
        jump_prob = self.core_failslow_prob

        for _ in range(max_iter):
            new_prob = alpha * self.transition_matrix.T @ self.core_failslow_prob
            jump_sum = jump_prob.sum()
            for idx in range(self.N):
                if jump_sum != 0:
                    new_prob[idx] += (1 - alpha) * jump_prob[idx] / jump_sum
            if np.linalg.norm(new_prob - self.core_failslow_prob, ord=1) < tol:
                break
            self.core_failslow_prob = new_prob

        return self.core_failslow_prob

    def link_summary(self, threshold=0.8, beta=DEFAULT_LINK_SUMMARY_BETA):
        failslow = FailSlows()
        links = []
        values = []

        for src, dst in self.context.topology.physical_links():
            values.append(self.link_hit_count[src][dst])
            links.append((src, dst))

        if not values:
            return failslow

        softmax_values = softmax(values, beta=beta)
        for idx, value in enumerate(softmax_values):
            if value > threshold:
                failslow.insert(FailSlow(kind="link", id=links[idx][0], dst_id=links[idx][1]))
        return failslow

    def failrank_summary(
        self,
        threshold=0.65,
        pe_softmax_beta=DEFAULT_PE_SOFTMAX_BETA,
        link_softmax_beta=DEFAULT_LINK_SOFTMAX_BETA,
        link_count_weight=DEFAULT_LINK_COUNT_WEIGHT,
        link_variance_weight=DEFAULT_LINK_VARIANCE_WEIGHT,
        pe_prob_weight=DEFAULT_PE_PROB_WEIGHT,
    ):
        failslow = FailSlows()
        for group_id in range(self.group_num):
            start_id = group_id * self.num
            end_id = start_id + self.num

            group_prob = softmax(self.core_failslow_prob[start_id:end_id], beta=pe_softmax_beta)
            for idx in range(self.num):
                if group_prob[idx] < threshold:
                    continue
                node_id = start_id + idx
                print(
                    f"[FailSlow-PE] Id: {node_id % self.num} Duration: "
                    f"[{self.time_range[node_id][0]},{self.time_range[node_id][1]}] Prob: {group_prob[idx] * 100:.2f}%."
                )
                failslow.insert(
                    FailSlow(
                        kind="pe",
                        id=node_id % self.num,
                        start_time=self.time_range[node_id][0],
                        end_time=self.time_range[node_id][1],
                    )
                )

            self.link_failslow_prob[:] = 0
            for j in range(start_id, end_id):
                for i in range(self.N):
                    self.link_failslow_prob[i][j] = (
                        self.link_count[i][j] / self.link_count.max() * link_count_weight
                        if self.link_count.max() != 0
                        else 0
                    )

                    src_core_id = i % self.num
                    dst_core_id = j % self.num
                    if src_core_id > dst_core_id:
                        src_core_id, dst_core_id = dst_core_id, src_core_id

                    self.link_failslow_prob[i][j] += self.link_variance[src_core_id][dst_core_id] * link_variance_weight
                    self.link_failslow_prob[i][j] += group_prob[j % self.num] * pe_prob_weight

            flat = self.link_failslow_prob.flatten()
            group_prob = softmax(flat, beta=link_softmax_beta).reshape(self.link_failslow_prob.shape)
            max_prob = np.max(group_prob)
            failslow_link = np.unravel_index(np.argmax(group_prob), self.link_failslow_prob.shape)

            if max_prob < threshold:
                continue

            if failslow_link[0] % self.num > failslow_link[1] % self.num:
                failslow_link = (failslow_link[1], failslow_link[0])

            if failslow_link not in self.link_time_range:
                continue

            fail_range = self.link_time_range[failslow_link]
            print(
                f"[FailSlow-Link] Id: {failslow_link[0] % self.num}-{failslow_link[1] % self.num} "
                f"Duration: [{fail_range[0]},{fail_range[1]}]"
            )
            failslow.insert(FailSlow(kind="link", id=failslow_link[0] % self.num, dst_id=failslow_link[1] % self.num))

        return failslow
