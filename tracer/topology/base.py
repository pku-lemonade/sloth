from abc import ABC, abstractmethod


class Topology(ABC):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    @property
    def node_count(self) -> int:
        return self.x * self.y

    def normalize_link(self, src_id: int, dst_id: int) -> tuple[int, int]:
        if src_id > dst_id:
            return dst_id, src_id
        return src_id, dst_id

    @abstractmethod
    def physical_links(self) -> list[tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def route_links(self, src_id: int, dst_id: int) -> list[tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def neighbor_for_direction(self, router_id: int, direction: int) -> int:
        raise NotImplementedError


class GridTopology(Topology):
    def to_id(self, x: int, y: int) -> int:
        return x * self.y + y

    def to_xy(self, node_id: int) -> tuple[int, int]:
        return node_id // self.y, node_id % self.y
