import simpy
import logging
import contextlib
from enum import IntEnum
from common.arch_config import LinkConfig, RouterConfig, NoCConfig
from common.runtime_config import MonitoringConfig
from evaluater.sim_type import Data, Message, Packet, ceil, Slice, Direction
from common.common import MonitoredResource
from common.distribution import NoCDist

logger = logging.getLogger("NoC")

class Link:
    def __init__(self, env, config, monitoring_config: MonitoringConfig):
        self.env = env
        self.width = config.width
        self.delay = config.delay
        self.store = simpy.Store(env)
        self.delay_factor = 1
        self.hop = 0
        self.linkentry = MonitoredResource(env, monitoring_config=monitoring_config)
        self.tag = False

        self.rate = 0.5
        self.shape = self.width * self.rate
        self.para_dist = NoCDist(shape=self.shape, rate=self.rate)

        self.per_word_transfer_time = 1 / self.width

        self.tot_size = 0
        self.layer_size = {}

    def bind(self, idx1, idx2, tag):
        self.corefrom = idx1
        self.coreto = idx2
        self.tag = tag
        
    def calc_latency(self, msg):
        slice = Slice(tensor_slice=msg.data.tensor_slice)

        true_width = self.para_dist.generate()
        transmission_time = ceil(slice.size(), true_width)

        latency = self.delay + transmission_time
        latency = latency * self.delay_factor

        self.tot_size += slice.size()
        if msg.ins.layer_id not in self.layer_size:
            self.layer_size[msg.ins.layer_id] = slice.size()
        else:
            self.layer_size[msg.ins.layer_id] += slice.size()

        self.hop += slice.size()/64
        
        yield self.linkentry.execute("SEND"+str(msg.data.index),latency,msg.ins,attributes=msg.dst)
        
        self.store.put(msg)
    
    def put(self, msg):
        return self.env.process(self.calc_latency(msg))
    
    def calc_latency_hop(self, packet):
        yield self.linkentry.execute("SEND"+str(packet.ins.index), 16*self.per_word_transfer_time, packet.ins, attributes=packet.dst)
        self.store.put(packet)
    
    def put_hop(self, packet):
        return self.env.process(self.calc_latency_hop(packet))
    
    def insert(self, msg):
        yield self.store.put(msg)

    def get(self):
        return self.store.get()
    
    def len(self):
        return len(self.store.items)

    def change_delay(self, times):
        self.delay_factor *= times

    def recover_delay(self, times):
        self.delay_factor /= times

class Router:
    def __init__(self, env, config: RouterConfig, id: int, x: int, y:int, model: str):
        self.env = env
        self.x = x
        self.y = y
        self.id = id

        self.type = config.type
        self.vc = config.vc

        self.per_hop_time = 1
        self.model = model

        self.core_in, self.core_out = None, None
        self.north_in, self.north_out = None, None
        self.south_in, self.south_out = None, None
        self.east_in, self.east_out = None, None
        self.west_in, self.west_out = None, None

        self.env.process(self.run())

    def bound_with_north(self, north_in, north_out):
        self.north_in = north_in
        self.north_out = north_out

    def bound_with_south(self, south_in, south_out):
        self.south_in = south_in
        self.south_out = south_out

    def bound_with_east(self, east_in, east_out):
        self.east_in = east_in
        self.east_out = east_out

    def bound_with_west(self, west_in, west_out):
        self.west_in = west_in
        self.west_out = west_out

    def bound_with_core(self, core_in, core_out):
        self.core_in = core_in
        self.core_out = core_out

    def router_fail(self, times):
        self.north_in.change_delay(times)
        self.north_out.change_delay(times)

        self.south_in.change_delay(times)
        self.south_out.change_delay(times)

        self.east_in.change_delay(times)
        self.east_out.change_delay(times)

        self.west_in.change_delay(times)
        self.west_out.change_delay(times)

    def router_recover(self, times):
        self.north_in.recover_delay(times)
        self.north_out.recover_delay(times)
        
        self.south_in.recover_delay(times)
        self.south_out.recover_delay(times)

        self.east_in.recover_delay(times)
        self.east_out.recover_delay(times)
        
        self.west_in.recover_delay(times)
        self.west_out.recover_delay(times)

    def route(self, msg: Message, next_dir, next_router):
        match next_dir:
            case Direction.NORTH:
                yield self.north_out.put(msg)
            case Direction.SOUTH:
                yield self.south_out.put(msg)
            case Direction.EAST:
                yield self.east_out.put(msg)
            case Direction.WEST:
                yield self.west_out.put(msg)

        logger.debug("Time %.2f: Router%d finish sending data%d to router%d(dst:%d).",
            self.env.now, self.id, msg.data.index, next_router, msg.dst)

    def route_core(self, msg):
        yield self.core_out.put(msg)
        logger.debug("Time %.2f: Finish putting data%d to PE%d", self.env.now, msg.data.index, self.id)

    def routing(self, msg):
        if msg.dst == self.id:
            logger.debug("Time %.2f: Routing data%d to router%d.", self.env.now, msg.data.index, self.id)
            yield self.env.process(self.route_core(msg))
        else:
            yield self.env.timeout(self.per_hop_time)
            next_dir, next_router = self.calculate_next_router(msg.dst)
            logger.debug("Time %.2f: Router%d start sending data%d to router%d(dst:%d).",
                self.env.now, self.id, msg.data.index, next_router, msg.dst)
            yield self.env.process(self.route(msg, next_dir, next_router))

    def route_hop(self, packet: Packet, next_dir, next_router):
        match next_dir:
            case Direction.NORTH:
                yield self.north_out.put_hop(packet)
            case Direction.SOUTH:
                yield self.south_out.put_hop(packet)
            case Direction.EAST:
                yield self.east_out.put_hop(packet)
            case Direction.WEST:
                yield self.west_out.put_hop(packet)


    def route_core_hop(self, packet):
        yield self.core_out.put_hop(packet)

    def routing_hop(self, packet):
        yield self.env.timeout(self.per_hop_time)
        
        if packet.dst == self.id:
            yield self.env.process(self.route_core_hop(packet))
        else:
            next_dir, next_router = self.calculate_next_router(packet.dst)
            yield self.env.process(self.route_hop(packet, next_dir, next_router))

    def run(self):
        while True:
            all_possible_channels = [(self.north_in, 0), (self.south_in, 1), (self.east_in, 2), (self.west_in, 3), (self.core_in, 4)]
            all_channels = [channel for channel in all_possible_channels if channel[0] is not None]

            with contextlib.ExitStack() as stack:
                all_events = [stack.enter_context(channel[0].get()) for channel in all_channels]

                events = self.env.any_of(all_events)
                result = yield events

                for id, event in enumerate(all_events):
                    
                    if event.triggered:
                        msg = event.value
                        if self.model == "basic":
                            self.env.process(self.routing(msg))
                        elif self.model == "packet":
                            self.env.process(self.routing_hop(msg))
                        
                        channel = None
                        match all_channels[id][1]:
                            case 0: channel = self.north_in
                            case 1: channel = self.south_in
                            case 2: channel = self.east_in
                            case 3: channel = self.west_in
                            case 4: channel = self.core_in
                        
                        while channel.len() > 0:
                            msg = yield channel.get()
                            if self.model == "basic":
                                self.env.process(self.routing(msg))
                            elif self.model == "packet":
                                self.env.process(self.routing_hop(msg))

    def trans(self,start_time,link,flow):
        yield self.env.timeout(start_time)
        yield self.env.process(link.transmit(flow))
        
        
    def addtion_flow(self,env,linklist,timelist,flowlist):
        lenth=len(linklist)
        assert lenth==len(timelist)
        for i in range(lenth):
            start_time=timelist[i]
            link=linklist[i]
            flow=flowlist[i]    
            env.process(self.trans(start_time,link,flow))
        

    def get_layer(self, x: int, y: int):
        return min(x, self.x - 1 - x, y, self.y - 1 - y)
    

    def is_corner(self, x: int, y: int):
        return min(x, self.x - 1 - x) == min(y, self.y - 1 - y)
    

    def outer(self, x: int, y: int, layer_id: int):
        new_x = x - 1 if x == layer_id else x + 1
        new_y = y - 1 if y == layer_id else y + 1
        return self.to_x(new_x, new_y)
    

    def inner(self, x: int, y: int, layer_id: int):
        new_x = x + 1 if x == layer_id else x - 1
        new_y = y + 1 if y == layer_id else y - 1
        return self.to_x(new_x, new_y)
    

    def ring_next(self, x: int, y: int, layer_id: int):
        if self.is_corner(x, y):
            if x == layer_id:
                if y == layer_id:
                    return self.to_x(x, y + 1)
                else:
                    return self.to_x(x + 1, y)
            else:
                if y == layer_id:
                    return self.to_x(x - 1, y)
                else:
                    return self.to_x(x, y - 1)
        else:
            if x == layer_id:
                return self.to_x(x, y + 1)
            if x == self.x - 1 - layer_id:
                return self.to_x(x, y - 1)
            if y == layer_id:
                return self.to_x(x - 1, y)
            if y == self.y - 1 - layer_id:
                return self.to_x(x + 1, y)
            

    # mapping mesh-style id to dragonfly id
    def get_dragonfly_info(self, router_id):
        """
        Group 0 (BL): [0, 1, 4, 5]
        Group 1 (BR): [8, 9, 12, 13]
        Group 2 (TL): [2, 3, 6, 7]
        Group 3 (TR): [10, 11, 14, 15]
        """
        groups = [
            [0, 4, 1, 5],      # Group 0
            [8, 12, 9, 13],    # Group 1
            [2, 6, 3, 7],      # Group 2
            [10, 14, 11, 15]   # Group 3
        ]
        
        gid = -1
        l_idx = -1
        
        for g_idx, nodes in enumerate(groups):
            if router_id in nodes:
                gid = g_idx
                l_idx = nodes.index(router_id)
                return gid, l_idx, nodes
        
        raise ValueError(f"Router ID {router_id} not valid for 4x4 Dragonfly mapping")
        

    def calculate_next_router(self, target_id):
        if self.type == "XY":
            now_x, now_y = self.to_xy(self.id)
            tar_x, tar_y = self.to_xy(target_id)

            if now_x != tar_x:
                if tar_x > now_x:
                    return Direction.EAST, self.to_x(now_x + 1, now_y)
                else:
                    return Direction.WEST, self.to_x(now_x - 1, now_y)
        
            if now_y != tar_y:
                if tar_y > now_y:
                    return Direction.NORTH, self.to_x(now_x, now_y + 1)
                else:
                    return Direction.SOUTH, self.to_x(now_x, now_y - 1)

        elif self.type == "Torus_XY":
            now_x, now_y = self.to_xy(self.id)
            tar_x, tar_y = self.to_xy(target_id)

            if now_x != tar_x:
                delta_x = tar_x - now_x
                if abs(delta_x) > self.x / 2:
                    if delta_x > 0:
                        return Direction.WEST, self.to_x(now_x - 1, now_y)
                    else:
                        return Direction.EAST, self.to_x(now_x + 1, now_y)
                else:
                    if delta_x > 0:
                        return Direction.EAST, self.to_x(now_x + 1, now_y)
                    else:
                        return Direction.WEST, self.to_x(now_x - 1, now_y)
                    
            if now_y != tar_y:
                delta_y = tar_y - now_y
                if abs(delta_y) > self.y / 2:
                    if delta_y > 0:
                        return Direction.SOUTH, self.to_x(now_x, now_y - 1)
                    else:
                        return Direction.NORTH, self.to_x(now_x, now_y + 1)
                else:
                    if delta_y > 0:
                        return Direction.NORTH, self.to_x(now_x, now_y + 1)
                    else:
                        return Direction.SOUTH, self.to_x(now_x, now_y - 1)
                    
        elif self.type == "RingRoad":
            now_x, now_y = self.to_xy(self.id)
            tar_x, tar_y = self.to_xy(target_id)

            now_layer = self.get_layer(now_x, now_y)
            tar_layer = self.get_layer(tar_x, tar_y)

            # route to the same layer first
            if now_layer != tar_layer:
                if self.is_corner(now_x, now_y):
                    if now_layer > tar_layer:
                        return Direction.NORTH, self.outer(now_x, now_y, now_layer)
                    else:
                        return Direction.SOUTH, self.inner(now_x, now_y, now_layer)
                else:
                    return Direction.EAST, self.ring_next(now_x, now_y, now_layer)
            else:
                return Direction.EAST, self.ring_next(now_x, now_y, now_layer)
            
        elif self.type == "Dragonfly":
            my_gid, my_lidx, my_group_nodes = self.get_dragonfly_info(self.id)
            tgt_gid, tgt_lidx, tgt_group_nodes = self.get_dragonfly_info(target_id)
            
            # Case 1: intra-group routing
            if my_gid == tgt_gid:
                next_hop_id = target_id
                
            # Case 2: inter-group routing
            else:
                gateway_lidx = tgt_gid
                
                if my_lidx == gateway_lidx:
                    next_hop_id = tgt_group_nodes[my_gid] 
                    
                    next_x, next_y = self.to_xy(next_hop_id)
                    return Direction.WEST, self.to_x(next_x, next_y)
                
                else:
                    next_hop_id = my_group_nodes[gateway_lidx]

            neighbors = sorted([n for n in my_group_nodes if n != self.id])
            neighbor_pos = neighbors.index(next_hop_id)
            
            intra_ports = [Direction.NORTH, Direction.SOUTH, Direction.EAST]
            next_dir = intra_ports[neighbor_pos]
            
            next_x, next_y = self.to_xy(next_hop_id)
            return next_dir, self.to_x(next_x, next_y)
        
        else:
            pass


    def to_x(self, x, y):
        return x * self.y + y
    
    def to_xy(self, id):
        x = id // self.y
        y = id % self.y
        return x, y


class NoC:
    def __init__(self, env, config: NoCConfig, model: str, monitoring_config: MonitoringConfig):
        self.env = env
        self.x = config.x
        self.y = config.y
        self.router_config = config.router
        self.link_config = config.link
        self.r2r_links = []
        self.routers = []

        self.model = model
        self.monitoring_config = monitoring_config

    def build_connection(self):
        for id in range(self.x * self.y):
            self.id = id
            self.routers.append(Router(self.env, self.router_config, id, self.x, self.y, self.model))

        for row in range(self.x):
            for col in range(self.y):
                router_id = row * self.y + col
                if row < self.x - 1:
                    east_router_id = (row + 1) * self.y + col
                    
                    link1 = Link(self.env, self.link_config, self.monitoring_config)
                    link2 = Link(self.env, self.link_config, self.monitoring_config)

                    link2.bind((row,col), (row+1,col), True)
                    link1.bind((row+1,col), (row,col), True)

                    self.routers[router_id].bound_with_east(link1, link2)
                    self.routers[east_router_id].bound_with_west(link2, link1)

                    self.r2r_links.append(link1)
                    self.r2r_links.append(link2)

                if col > 0:
                    south_router_id = row * self.y + (col - 1)

                    link1 = Link(self.env, self.link_config, self.monitoring_config)
                    link2 = Link(self.env, self.link_config, self.monitoring_config)

                    link2.bind((row,col), (row,col-1), True)
                    link1.bind((row,col-1), (row,col), True)

                    self.routers[router_id].bound_with_south(link1, link2)
                    self.routers[south_router_id].bound_with_north(link2, link1)
                    
                    self.r2r_links.append(link1)
                    self.r2r_links.append(link2)
                    
        return self


    def build_connection_torus(self):
        for id in range(self.x * self.y):
            self.routers.append(Router(self.env, self.router_config, id, self.x, self.y, self.model))

        for row in range(self.x):
            for col in range(self.y):
                # horizontal links
                router_id = row * self.y + col
                
                east_row = (row + 1) % self.x
                east_router_id = east_row * self.y + col
                
                link1 = Link(self.env, self.link_config, self.monitoring_config)
                link2 = Link(self.env, self.link_config, self.monitoring_config)

                link2.bind((row, col), (east_row, col), True)
                link1.bind((east_row, col), (row, col), True)

                self.routers[router_id].bound_with_east(link1, link2)
                self.routers[east_router_id].bound_with_west(link2, link1)

                self.r2r_links.append(link1)
                self.r2r_links.append(link2)

                # vertical links
                south_col = (col - 1) % self.y
                south_router_id = row * self.y + south_col

                link1 = Link(self.env, self.link_config, self.monitoring_config)
                link2 = Link(self.env, self.link_config, self.monitoring_config)

                link2.bind((row, col), (row, south_col), True)
                link1.bind((row, south_col), (row, col), True)

                self.routers[router_id].bound_with_south(link1, link2)
                self.routers[south_router_id].bound_with_north(link2, link1)
                
                self.r2r_links.append(link1)
                self.r2r_links.append(link2)
                    
        return self

    # calculate ring ID
    def get_layer(self, row, col):
        return min(row, self.x - 1 - row, col, self.y - 1 - col)
    

    def get_ring_nodes_ordered(self, k: int):
        nodes = []
        low_x, low_y = k, k
        high_x, high_y = self.x - 1 - k, self.y - 1 - k

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
    

    def build_connection_ring_road(self):
        # Clockwise -> Port EAST
        # Counter-Clockwise -> Port WEST
        # To Outer Layer -> Port NORTH
        # To Inner Layer -> Port SOUTH
        
        for id in range(self.x * self.y):
            self.routers.append(Router(self.env, self.router_config, id, self.x, self.y, self.model))

        num_layers = (min(self.x, self.y) + 1) // 2

        for k in range(num_layers):
            # intra-layer connection
            ring_nodes = self.get_ring_nodes_ordered(k)
            num_nodes = len(ring_nodes)

            if num_nodes > 1:
                for i in range(num_nodes):
                    curr_coords = ring_nodes[i]
                    next_coords = ring_nodes[(i + 1) % num_nodes]

                    curr_id = curr_coords[0] * self.y + curr_coords[1]
                    next_id = next_coords[0] * self.y + next_coords[1]

                    link_cw = Link(self.env, self.link_config, self.monitoring_config)
                    link_ccw = Link(self.env, self.link_config, self.monitoring_config)

                    link_cw.bind(curr_coords, next_coords, True)
                    link_ccw.bind(next_coords, curr_coords, True)
                    
                    self.r2r_links.append(link_cw)
                    self.r2r_links.append(link_ccw)

                    self.routers[curr_id].bound_with_east(link_cw, link_ccw)
                    self.routers[next_id].bound_with_west(link_ccw, link_cw)

            # inter-layer connection
            if k < num_layers - 1:
                # cur: Outer; next: (Inner)
                low = k
                high_x = self.x - 1 - k
                high_y = self.y - 1 - k
                
                next_low = k + 1
                next_high_x = self.x - 1 - (k + 1)
                next_high_y = self.y - 1 - (k + 1)

                # (OuterCoords, InnerCoords)
                corners = [
                    ((low, low),       (next_low, next_low)),
                    ((low, high_y),    (next_low, next_high_y)),
                    ((high_x, high_y), (next_high_x, next_high_y)),
                    ((high_x, low),    (next_high_x, next_low))
                ]

                for (outer_pos, inner_pos) in corners:
                    outer_id = outer_pos[0] * self.y + outer_pos[1]
                    inner_id = inner_pos[0] * self.y + inner_pos[1]

                    link_inward = Link(self.env, self.link_config, self.monitoring_config)
                    link_outward = Link(self.env, self.link_config, self.monitoring_config)

                    link_inward.bind(outer_pos, inner_pos, True)
                    link_outward.bind(inner_pos, outer_pos, True)
                    
                    self.r2r_links.append(link_inward)
                    self.r2r_links.append(link_outward)

                    self.routers[outer_id].bound_with_south(link_inward, link_outward)
                    self.routers[inner_id].bound_with_north(link_outward, link_inward)

        return self


    def build_connection_dragonfly(self):
        for id in range(self.x * self.y):
            self.routers.append(Router(self.env, self.router_config, id, self.x, self.y, self.model))

        all_groups = [
            [0, 4, 1, 5],      # Group 0
            [8, 12, 9, 13],    # Group 1
            [2, 6, 3, 7],      # Group 2
            [10, 14, 11, 15]   # Group 3
        ]

        local_ports = [Direction.NORTH, Direction.SOUTH, Direction.EAST]
        global_port = Direction.WEST

        # intra-group clique
        for g_idx, nodes in enumerate(all_groups):
            group_size = len(nodes)

            for i in range(group_size):
                for j in range(i + 1, group_size):
                    u_id = nodes[i]
                    v_id = nodes[j]

                    u_neighbors = [n for n in nodes if n != u_id]
                    u_port_idx = u_neighbors.index(v_id)
                    dir_u = local_ports[u_port_idx]

                    v_neighbors = [n for n in nodes if n != v_id]
                    v_port_idx = v_neighbors.index(u_id)
                    dir_v = local_ports[v_port_idx]

                    link1 = Link(self.env, self.link_config, self.monitoring_config)
                    link2 = Link(self.env, self.link_config, self.monitoring_config)

                    coord_u = self.routers[u_id].to_xy(u_id)
                    coord_v = self.routers[v_id].to_xy(v_id)
                    
                    link1.bind(coord_u, coord_v, True)
                    link2.bind(coord_v, coord_u, True)

                    match dir_u:
                        case Direction.NORTH:
                            self.routers[u_id].bound_with_north(link2, link1)
                        case Direction.SOUTH:
                            self.routers[u_id].bound_with_south(link2, link1)
                        case Direction.EAST:
                            self.routers[u_id].bound_with_east(link2, link1)

                    match dir_v:
                        case Direction.NORTH:
                            self.routers[v_id].bound_with_north(link1, link2)
                        case Direction.SOUTH:
                            self.routers[v_id].bound_with_south(link1, link2)
                        case Direction.EAST:
                            self.routers[v_id].bound_with_east(link1, link2)

                    self.r2r_links.append(link1)
                    self.r2r_links.append(link2)

        # inter-group global links
        num_groups = len(all_groups)
        for g_src in range(num_groups):
            for g_dst in range(g_src + 1, num_groups):
                
                src_group_nodes = all_groups[g_src]
                src_id = src_group_nodes[g_dst]

                dst_group_nodes = all_groups[g_dst]
                dst_id = dst_group_nodes[g_src]

                link_out = Link(self.env, self.link_config, self.monitoring_config)
                link_in = Link(self.env, self.link_config, self.monitoring_config)

                coord_src = self.routers[src_id].to_xy(src_id)
                coord_dst = self.routers[dst_id].to_xy(dst_id)

                link_out.bind(coord_src, coord_dst, True)
                link_in.bind(coord_dst, coord_src, True)

                self.routers[src_id].bound_with_west(link_in, link_out)
                self.routers[dst_id].bound_with_west(link_out, link_in)
                
                self.r2r_links.append(link_in)
                self.r2r_links.append(link_out)

        return self