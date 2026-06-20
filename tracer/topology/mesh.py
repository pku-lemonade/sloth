from evaluater.sim_type import Direction

from tracer.topology.base import GridTopology


class MeshTopology(GridTopology):
    name = "mesh"

    def physical_links(self) -> list[tuple[int, int]]:
        links = set()
        for row in range(self.x):
            for col in range(self.y):
                node_id = self.to_id(row, col)
                if row < self.x - 1:
                    links.add(self.normalize_link(node_id, self.to_id(row + 1, col)))
                if col > 0:
                    links.add(self.normalize_link(node_id, self.to_id(row, col - 1)))
        return sorted(links)

    def _next_router(self, current_id: int, target_id: int) -> int:
        now_x, now_y = self.to_xy(current_id)
        target_x, target_y = self.to_xy(target_id)
        if now_x != target_x:
            return self.to_id(now_x + 1, now_y) if target_x > now_x else self.to_id(now_x - 1, now_y)
        if now_y != target_y:
            return self.to_id(now_x, now_y + 1) if target_y > now_y else self.to_id(now_x, now_y - 1)
        return current_id

    def route_links(self, src_id: int, dst_id: int) -> list[tuple[int, int]]:
        path = []
        current = src_id
        while current != dst_id:
            nxt = self._next_router(current, dst_id)
            path.append(self.normalize_link(current, nxt))
            current = nxt
        return path

    def neighbor_for_direction(self, router_id: int, direction: int) -> int:
        x, y = self.to_xy(router_id)
        if direction == Direction.NORTH:
            if y + 1 >= self.y:
                raise ValueError(f"Router {router_id} has no NORTH neighbor")
            return self.to_id(x, y + 1)
        if direction == Direction.SOUTH:
            if y - 1 < 0:
                raise ValueError(f"Router {router_id} has no SOUTH neighbor")
            return self.to_id(x, y - 1)
        if direction == Direction.EAST:
            if x + 1 >= self.x:
                raise ValueError(f"Router {router_id} has no EAST neighbor")
            return self.to_id(x + 1, y)
        if direction == Direction.WEST:
            if x - 1 < 0:
                raise ValueError(f"Router {router_id} has no WEST neighbor")
            return self.to_id(x - 1, y)
        raise ValueError(f"Unsupported direction: {direction}")
