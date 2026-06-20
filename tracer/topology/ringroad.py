from evaluater.sim_type import Direction

from tracer.topology.base import GridTopology


class RingRoadTopology(GridTopology):
    name = "ringroad"

    def get_layer(self, row: int, col: int) -> int:
        return min(row, self.x - 1 - row, col, self.y - 1 - col)

    def is_corner(self, row: int, col: int) -> bool:
        return min(row, self.x - 1 - row) == min(col, self.y - 1 - col)

    def outer(self, row: int, col: int, layer_id: int) -> int:
        new_x = row - 1 if row == layer_id else row + 1
        new_y = col - 1 if col == layer_id else col + 1
        return self.to_id(new_x, new_y)

    def inner(self, row: int, col: int, layer_id: int) -> int:
        new_x = row + 1 if row == layer_id else row - 1
        new_y = col + 1 if col == layer_id else col - 1
        return self.to_id(new_x, new_y)

    def ring_next(self, row: int, col: int, layer_id: int) -> int:
        if self.is_corner(row, col):
            if row == layer_id:
                if col == layer_id:
                    return self.to_id(row, col + 1)
                return self.to_id(row + 1, col)
            if col == layer_id:
                return self.to_id(row - 1, col)
            return self.to_id(row, col - 1)

        if row == layer_id:
            return self.to_id(row, col + 1)
        if row == self.x - 1 - layer_id:
            return self.to_id(row, col - 1)
        if col == layer_id:
            return self.to_id(row - 1, col)
        return self.to_id(row + 1, col)

    def ring_prev(self, row: int, col: int, layer_id: int) -> int:
        ring = self.get_ring_nodes_ordered(layer_id)
        ring_ids = [self.to_id(x, y) for x, y in ring]
        idx = ring_ids.index(self.to_id(row, col))
        return ring_ids[(idx - 1) % len(ring_ids)]

    def get_ring_nodes_ordered(self, layer_id: int):
        nodes = []
        low_x, low_y = layer_id, layer_id
        high_x, high_y = self.x - 1 - layer_id, self.y - 1 - layer_id
        if low_x == high_x and low_y == high_y:
            return [(low_x, low_y)]
        for y in range(low_y, high_y):
            nodes.append((low_x, y))
        for x in range(low_x, high_x):
            nodes.append((x, high_y))
        for y in range(high_y, low_y, -1):
            nodes.append((high_x, y))
        for x in range(high_x, low_x, -1):
            nodes.append((x, low_y))
        return nodes

    def physical_links(self) -> list[tuple[int, int]]:
        links = set()
        num_layers = (min(self.x, self.y) + 1) // 2
        for layer_id in range(num_layers):
            ring_nodes = self.get_ring_nodes_ordered(layer_id)
            for idx, current in enumerate(ring_nodes):
                nxt = ring_nodes[(idx + 1) % len(ring_nodes)]
                links.add(self.normalize_link(self.to_id(*current), self.to_id(*nxt)))

            if layer_id < num_layers - 1:
                low = layer_id
                high_x = self.x - 1 - layer_id
                high_y = self.y - 1 - layer_id
                next_low = layer_id + 1
                next_high_x = self.x - 1 - next_low
                next_high_y = self.y - 1 - next_low
                corners = [
                    ((low, low), (next_low, next_low)),
                    ((low, high_y), (next_low, next_high_y)),
                    ((high_x, high_y), (next_high_x, next_high_y)),
                    ((high_x, low), (next_high_x, next_low)),
                ]
                for outer_pos, inner_pos in corners:
                    links.add(self.normalize_link(self.to_id(*outer_pos), self.to_id(*inner_pos)))
        return sorted(links)

    def _next_router(self, current_id: int, target_id: int) -> int:
        now_x, now_y = self.to_xy(current_id)
        target_x, target_y = self.to_xy(target_id)
        now_layer = self.get_layer(now_x, now_y)
        target_layer = self.get_layer(target_x, target_y)
        if now_layer != target_layer:
            if self.is_corner(now_x, now_y):
                return self.outer(now_x, now_y, now_layer) if now_layer > target_layer else self.inner(now_x, now_y, now_layer)
            return self.ring_next(now_x, now_y, now_layer)
        return self.ring_next(now_x, now_y, now_layer)

    def route_links(self, src_id: int, dst_id: int) -> list[tuple[int, int]]:
        path = []
        current = src_id
        while current != dst_id:
            nxt = self._next_router(current, dst_id)
            path.append(self.normalize_link(current, nxt))
            current = nxt
        return path

    def neighbor_for_direction(self, router_id: int, direction: int) -> int:
        row, col = self.to_xy(router_id)
        layer_id = self.get_layer(row, col)
        if direction == Direction.EAST:
            return self.ring_next(row, col, layer_id)
        if direction == Direction.WEST:
            return self.ring_prev(row, col, layer_id)
        if direction == Direction.NORTH:
            if not self.is_corner(row, col) or layer_id == 0:
                raise ValueError(f"Router {router_id} has no NORTH ringroad neighbor")
            return self.outer(row, col, layer_id)
        if direction == Direction.SOUTH:
            if not self.is_corner(row, col) or layer_id >= (min(self.x, self.y) + 1) // 2 - 1:
                raise ValueError(f"Router {router_id} has no SOUTH ringroad neighbor")
            return self.inner(row, col, layer_id)
        raise ValueError(f"Unsupported direction: {direction}")
