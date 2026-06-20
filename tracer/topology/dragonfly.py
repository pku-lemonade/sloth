from evaluater.sim_type import Direction

from tracer.topology.base import GridTopology


class DragonflyTopology(GridTopology):
    name = "dragonfly"

    GROUPS = [
        [0, 4, 1, 5],
        [8, 12, 9, 13],
        [2, 6, 3, 7],
        [10, 14, 11, 15],
    ]
    LOCAL_PORTS = [Direction.NORTH, Direction.SOUTH, Direction.EAST]

    def get_dragonfly_info(self, router_id: int):
        for gid, nodes in enumerate(self.GROUPS):
            if router_id in nodes:
                return gid, nodes.index(router_id), nodes
        raise ValueError(f"Router ID {router_id} not valid for 4x4 Dragonfly mapping")

    def physical_links(self) -> list[tuple[int, int]]:
        links = set()
        for nodes in self.GROUPS:
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    links.add(self.normalize_link(nodes[i], nodes[j]))

        num_groups = len(self.GROUPS)
        for src_gid in range(num_groups):
            for dst_gid in range(src_gid + 1, num_groups):
                src_id = self.GROUPS[src_gid][dst_gid]
                dst_id = self.GROUPS[dst_gid][src_gid]
                links.add(self.normalize_link(src_id, dst_id))
        return sorted(links)

    def _global_neighbor(self, router_id: int) -> int:
        gid, local_idx, _ = self.get_dragonfly_info(router_id)
        target_group = self.GROUPS[local_idx]
        return target_group[gid]

    def _next_router(self, current_id: int, target_id: int) -> int:
        my_gid, my_lidx, my_group_nodes = self.get_dragonfly_info(current_id)
        target_gid, _, target_group_nodes = self.get_dragonfly_info(target_id)

        if my_gid == target_gid:
            return target_id

        gateway_lidx = target_gid
        if my_lidx == gateway_lidx:
            return target_group_nodes[my_gid]
        return my_group_nodes[gateway_lidx]

    def route_links(self, src_id: int, dst_id: int) -> list[tuple[int, int]]:
        path = []
        current = src_id
        while current != dst_id:
            nxt = self._next_router(current, dst_id)
            path.append(self.normalize_link(current, nxt))
            current = nxt
        return path

    def neighbor_for_direction(self, router_id: int, direction: int) -> int:
        gid, _, nodes = self.get_dragonfly_info(router_id)
        local_neighbors = sorted(node for node in nodes if node != router_id)
        if direction in self.LOCAL_PORTS:
            return local_neighbors[self.LOCAL_PORTS.index(direction)]
        if direction == Direction.WEST:
            return self._global_neighbor(router_id)
        raise ValueError(f"Unsupported direction: {direction}")
