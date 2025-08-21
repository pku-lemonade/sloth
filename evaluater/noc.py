import simpy
import logging
import contextlib
from enum import IntEnum
from common.arch_config import LinkConfig, RouterConfig, NoCConfig
from evaluater.sim_type import Data, Message, Packet, ceil, Slice, Direction
from common.common import MonitoredResource
from common.distribution import NoCDist

logger = logging.getLogger("NoC")

class Link:
    def __init__(self, env, config):
        self.env = env
        self.width = config.width
        self.delay = config.delay
        self.store = simpy.Store(env)
        self.delay_factor = 1
        self.hop = 0
        self.linkentry = MonitoredResource(env)
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
        else:
            return target_id

    def to_x(self, x, y):
        return x * self.y + y
    
    def to_xy(self, id):
        x = id // self.y
        y = id % self.y
        return x, y


class NoC:
    def __init__(self, env, config: NoCConfig, model: str):
        self.env = env
        self.x = config.x
        self.y = config.y
        self.router_config = config.router
        self.link_config = config.link
        self.r2r_links = []
        self.routers = []

        self.model = model

    def build_connection(self):
        for id in range(self.x * self.y):
            self.id = id
            self.routers.append(Router(self.env, self.router_config, id, self.x, self.y, self.model))

        for row in range(self.x):
            for col in range(self.y):
                router_id = row * self.y + col
                if row < self.x - 1:
                    east_router_id = (row + 1) * self.y + col
                    
                    link1 = Link(self.env, self.link_config)
                    link2 = Link(self.env, self.link_config)

                    link2.bind((row,col), (row+1,col), True)
                    link1.bind((row+1,col), (row,col), True)

                    self.routers[router_id].bound_with_east(link1, link2)
                    self.routers[east_router_id].bound_with_west(link2, link1)

                    self.r2r_links.append(link1)
                    self.r2r_links.append(link2)

                if col > 0:
                    south_router_id = row * self.y + (col - 1)

                    link1 = Link(self.env, self.link_config)
                    link2 = Link(self.env, self.link_config)

                    link2.bind((row,col), (row,col-1), True)
                    link1.bind((row,col-1), (row,col), True)

                    self.routers[router_id].bound_with_south(link1, link2)
                    self.routers[south_router_id].bound_with_north(link2, link1)
                    
                    self.r2r_links.append(link1)
                    self.r2r_links.append(link2)
                    
        return self