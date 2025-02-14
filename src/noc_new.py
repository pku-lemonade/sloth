import simpy
import logging
from enum import IntEnum
from src.arch_config import LinkConfig, RouterConfig, NoCConfig
from src.sim_type import Data, Message, ceil

logger = logging.getLogger("NoC")

class Direction(IntEnum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4

class Link:
    def __init__(self, env, config: LinkConfig):
        self.env = env
        self.width = config.width
        self.delay = config.delay
        self.store = simpy.Store(env)

    def transmit(self, size):
        transmission_time = ceil(size, self.width)
        latency = self.delay + transmission_time

        self.store.put(size)
        yield self.env.timeout(latency)
        self.store.get()

    def change_width(self, times):
        self.width *= times


class Router:
    def __init__(self, env, config: RouterConfig, id: int, x: int, y:int):
        self.env = env
        self.x = x
        self.y = y
        self.id = id

        self.type = config.type
        self.vc = config.vc

        self.core_in = simpy.Store(self.env)
        self.core_out = None
        self.core_link = None

        self.north_in = simpy.Store(self.env)
        self.north_out = None
        self.north_link = None

        self.south_in = simpy.Store(self.env)
        self.south_out = None
        self.south_link = None

        self.east_in = simpy.Store(self.env)
        self.east_out = None
        self.east_link = None

        self.west_in = simpy.Store(self.env)
        self.west_out = None
        self.west_link = None

        
        self.core = None

        self.env.process(self.run())

    def bound_with_store(self, north_out, south_out, east_out, west_out):
        self.north_out = north_out
        self.south_out = south_out
        self.east_out = east_out
        self.west_out = west_out

    def bound_with_link(self, north_link, south_link, east_link, west_link):
        self.north_link = north_link
        self.south_link = south_link
        self.east_link = east_link
        self.west_link = west_link

    def bound_with_core(self, link: Link, core):
        self.core_out = core.data_queue
        self.core_link = link
        self.core = core

    def route(self, msg: Message, next_dir):
        match next_dir:
            case Direction.NORTH:
                yield self.env.process(self.north_link.transmit(msg.data.size))
                self.north_out.put(msg)
            case Direction.SOUTH:
                yield self.env.process(self.south_link.transmit(msg.data.size))
                self.south_out.put(msg)
            case Direction.EAST:
                yield self.env.process(self.east_link.transmit(msg.data.size))
                self.east_out.put(msg)
            case Direction.WEST:
                yield self.env.process(self.west_link.transmit(msg.data.size))
                self.west_out.put(msg)
 
    def run(self):
        while True:
            all_in_channels = []
            # all_in_channels = [self.north_in.get(), self.south_in.get(), self.east_in.get(), self.west_in.get(), self.core_in.get()]

            while len(self.north_in.items) > 0:
                all_in_channels.append(self.north_in.get())
            while len(self.south_in.items) > 0:
                all_in_channels.append(self.south_in.get())
            while len(self.east_in.items) > 0:
                all_in_channels.append(self.east_in.get())
            while len(self.west_in.items) > 0:
                all_in_channels.append(self.west_in.get())
            while len(self.core_in.items) > 0:
                all_in_channels.append(self.core_in.get())

            all_in_channels.extend([self.north_in.get(), self.south_in.get(), self.east_in.get(), self.west_in.get(), self.core_in.get()])


            events = self.env.any_of(all_in_channels)
            result = yield events

            for event in all_in_channels:
                if event.triggered:
                    msg = event.value
                    print(f"id:{self.id} is processing data{msg.data.index}(to {msg.dst})")
                    if msg.dst == self.id:
                        yield self.env.process(self.core_link.transmit(msg.data.size))
                        logger.info(f"Time {self.env.now:.2f}: Finish routing data{msg.data.index} to router{self.id}.")
                        self.core_out.put(msg)
                        logger.debug(f"router{self.id}'s data_queue_len is {self.core.data_len()}")
                    else:
                        next_dir, next_router = self.calculate_next_router(msg.dst)
                        print(f"next_dir is {next_dir}, next_router is {next_router}")
                        # print(f"self_id:{self.id} -> next_id:{next_router}, direction:{next_dir}")
                        logger.info(f"Time {self.env.now:.2f}: Router{self.id} send data{msg.data.index} to router{next_router}.")
                        self.env.process(self.route(msg, next_dir))
                        logger.info(f"Time {self.env.now:.2f}: Router{self.id} finished sending data{msg.data.index} to router{next_router}.")

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
        self.routers = []

    # connections between routers
    def build_connection(self):
        for id in range(self.x * self.y):
            self.id = id
            self.routers.append(Router(self.env, self.router_config, id, self.x, self.y))
        
        link = Link(self.env, self.link_config)
        broken_link = Link(self.env, self.link_config)

        for row in range(self.x):
            for col in range(self.y):
                router_id = row * self.y + col
                north_out, south_out, east_out, west_out = None, None, None, None
                # connect with the west Router
                if row > 0:
                    west_router_id = (row - 1) * self.y + col
                    west_out = self.routers[west_router_id].east_in
                    if router_id == 12:
                        print(west_router_id)
                # connect with the east Router
                if row < self.x - 1:
                    east_router_id = (row + 1) * self.y + col
                    east_out = self.routers[east_router_id].west_in
                # connect with the north Router
                if col < self.y - 1:
                    north_router_id = row * self.y + (col + 1)
                    north_out = self.routers[north_router_id].south_in
                # connect with the south Router
                if col > 0:
                    south_router_id = row * self.y + (col - 1)
                    south_out = self.routers[south_router_id].north_in
                
                self.routers[router_id].bound_with_store(north_out, south_out, east_out, west_out)
                self.routers[router_id].bound_with_link(link, link, link, link)