from evaluater.sim_type import Direction

from tracer.topology.base import GridTopology


class TorusTopology(GridTopology):
    name = "torus"

    def physical_links(self) -> list[tuple[int, int]]:
        links = set()
        for row in range(self.x):
            for col in range(self.y):
                node_id = self.to_id(row, col)
                links.add(self.normalize_link(node_id, self.to_id((row + 1) % self.x, col)))
                links.add(self.normalize_link(node_id, self.to_id(row, (col - 1) % self.y)))
        return sorted(links)

    def _next_router(self, current_id: int, target_id: int) -> int:
        now_x, now_y = self.to_xy(current_id)
        target_x, target_y = self.to_xy(target_id)

        if now_x != target_x:
            delta_x = target_x - now_x
            if abs(delta_x) > self.x / 2:
                return self.to_id((now_x - 1) % self.x, now_y) if delta_x > 0 else self.to_id((now_x + 1) % self.x, now_y)
            return self.to_id((now_x + 1) % self.x, now_y) if delta_x > 0 else self.to_id((now_x - 1) % self.x, now_y)

        if now_y != target_y:
            delta_y = target_y - now_y
            if abs(delta_y) > self.y / 2:
                return self.to_id(now_x, (now_y - 1) % self.y) if delta_y > 0 else self.to_id(now_x, (now_y + 1) % self.y)
            return self.to_id(now_x, (now_y + 1) % self.y) if delta_y > 0 else self.to_id(now_x, (now_y - 1) % self.y)

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
            return self.to_id(x, (y + 1) % self.y)
        if direction == Direction.SOUTH:
            return self.to_id(x, (y - 1) % self.y)
        if direction == Direction.EAST:
            return self.to_id((x + 1) % self.x, y)
        if direction == Direction.WEST:
            return self.to_id((x - 1) % self.x, y)
        raise ValueError(f"Unsupported direction: {direction}")
