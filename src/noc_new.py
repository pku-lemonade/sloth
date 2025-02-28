import simpy
import logging
import contextlib
from enum import IntEnum
from src.arch_config import LinkConfig, RouterConfig, NoCConfig
from src.sim_type import Data, Message, ceil
from src.common import MonitoredResource

logger = logging.getLogger("NoC")

class Direction(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

# class Link:
#     def __init__(self, env, config: LinkConfig):
#         self.env = env
#         self.width = config.width
#         self.delay = config.delay
#         # self.store = simpy.Store(env)

#     def transmit(self, size):
#         transmission_time = ceil(size, self.width)
#         latency = self.delay + transmission_time

#         # self.store.put(size)
#         yield self.env.timeout(latency)
#         # self.store.get()

#     def change_width(self, times):
#         self.width *= times

class Link:
    def __init__(self, env, config):
        self.env = env
        self.width = config.width
        self.delay = config.delay
        self.store = simpy.Store(env)
        self.linkentry = MonitoredResource(env,capacity=1)

        
    def calc_latency(self, msg):
        #calc latency:
        transmission_time = ceil(msg.data.size, self.width)
        latency = self.delay + transmission_time
        yield self.linkentry.execute(latency)
        self.store.put(msg)
    
    def put(self,msg):
        return self.env.process(self.calc_latency(msg))

    def get(self):
        return self.store.get()
    
    def len(self):
        return len(self.store.items)


    def change_delay(self,times):
        self.delay *= times



class Router:
    def __init__(self, env, config: RouterConfig, id: int, x: int, y:int):
        self.env = env
        self.x = x
        self.y = y
        self.id = id

        self.type = config.type
        self.vc = config.vc

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

        logger.info(f"Time {self.env.now:.2f}: Router{self.id} finish sending data{msg.data.index} to router{next_router}(dst:{msg.dst}).")

    def route_core(self, msg):
        yield self.core_out.put(msg)
        logger.info(f"Time {self.env.now:.2f}: Finish putting data{msg.data.index} to PE{self.id}")
 
    def run(self):
        while True:
            all_possible_channels = [(self.north_in, 0), (self.south_in, 1), (self.east_in, 2), (self.west_in, 3), (self.core_in, 4)]
            #有的没有四个方向
            all_channels = [channel for channel in all_possible_channels if channel[0] is not None]

            # with self.north_in.get() as n, self.south_in.get() as s, self.east_in.get() as e, self.west_in.get() as w, self.core_in.get() as c:
            with contextlib.ExitStack() as stack:
                # all_in_channels = [self.north_in.get(), self.south_in.get(), self.east_in.get(), self.west_in.get(), self.core_in.get()]
                all_events = [stack.enter_context(channel[0].get()) for channel in all_channels]

                events = self.env.any_of(all_events)
                #等到至少有一个触发
                result = yield events

                for id, event in enumerate(all_events):
                    
                    if event.triggered:
                        msg = event.value

                        if msg.dst == self.id:
                            logger.info(f"Time {self.env.now:.2f}: Finish routing data{msg.data.index} to router{self.id}.")
                            self.env.process(self.route_core(msg))
                        else:
                            next_dir, next_router = self.calculate_next_router(msg.dst)
                            logger.info(f"Time {self.env.now:.2f}: Router{self.id} start sending data{msg.data.index} to router{next_router}(dst:{msg.dst}).")
                            self.env.process(self.route(msg, next_dir, next_router))
                        
                        channel = None
                        match all_channels[id][1]:
                            case 0: channel = self.north_in
                            case 1: channel = self.south_in
                            case 2: channel = self.east_in
                            case 3: channel = self.west_in
                            case 4: channel = self.core_in
                        
                        while channel.len() > 0:
                            msg = yield channel.get()

                            if msg.dst == self.id:
                                logger.info(f"Time {self.env.now:.2f}: Finish routing data{msg.data.index} to router{self.id}.")
                                self.env.process(self.route_core(msg))
                            else:
                                next_dir, next_router = self.calculate_next_router(msg.dst)
                                logger.info(f"Time {self.env.now:.2f}: Router{self.id} finished sending data{msg.data.index} to router{next_router}(dst:{msg.dst}).")
                                self.env.process(self.route(msg, next_dir, next_router))

    #模拟其它流量造成的网络拥堵
    def trans(self,start_time,link,flow):
        yield self.env.timeout(start_time)
        yield self.env.process(link.transmit(flow))
        
        
    def addtion_flow(self,env,linklist,timelist,flowlist):
        #加入额外的流量，用list来记录，linklist是List[(link)],timelist是List[(time)]
        lenth=len(linklist)
        assert lenth==len(timelist)
        for i in range(lenth):
            start_time=timelist[i]
            link=linklist[i]
            flow=flowlist[i]    
            env.process(self.trans(start_time,link,flow))
        
        

    def calculate_next_router(self, target_id):
        if self.type == "XY":
            # switch id
            now_x, now_y = self.to_xy(self.id)
            tar_x, tar_y = self.to_xy(target_id)

            # X first
            if now_x != tar_x:
                if tar_x > now_x:
                    return Direction.EAST, self.to_x(now_x + 1, now_y)
                else:
                    return Direction.WEST, self.to_x(now_x - 1, now_y)
        
            # then Y
            if now_y != tar_y:
                if tar_y > now_y:
                    return Direction.NORTH, self.to_x(now_x, now_y + 1)
                else:
                    return Direction.SOUTH, self.to_x(now_x, now_y - 1)
        else:
            return target_id

    # to1D id, 0-indexed
    def to_x(self, x, y):
        return x * self.y + y
    
    # to2D id, 0-indexed
    def to_xy(self, id):
        x = id // self.y
        y = id % self.y
        return x, y


class NoC:
    def __init__(self, env, config: NoCConfig):
        self.env = env
        self.x = config.x
        self.y = config.y
        self.router_config = config.router
        self.link_config = config.link
        self.r2r_links = []
        self.routers = []

    # connections between routers
    def build_connection(self):
        for id in range(self.x * self.y):
            self.id = id
            self.routers.append(Router(self.env, self.router_config, id, self.x, self.y))

        for row in range(self.x):
            for col in range(self.y):
                router_id = row * self.y + col
                # connect with the east Router
                if row < self.x - 1:
                    east_router_id = (row + 1) * self.y + col
                    
                    link1 = Link(self.env, self.link_config)
                    link2 = Link(self.env, self.link_config)

                    self.routers[router_id].bound_with_east(link1, link2)
                    self.routers[east_router_id].bound_with_west(link2, link1)

                    self.r2r_links.append(link1)
                    self.r2r_links.append(link2)

                # connect with the south Router
                if col > 0:
                    south_router_id = row * self.y + (col - 1)

                    link1 = Link(self.env, self.link_config)
                    link2 = Link(self.env, self.link_config)

                    self.routers[router_id].bound_with_south(link1, link2)
                    self.routers[south_router_id].bound_with_north(link2, link1)
                    
                    self.r2r_links.append(link1)
                    self.r2r_links.append(link2)
                    
        return self