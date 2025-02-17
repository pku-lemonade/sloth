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


class Router:
    def __init__(self, env, config: RouterConfig, id: int, x: int, y:int):
        self.env = env
        self.type = config.type
        self.vc = config.vc
        self.x = x
        self.y = y
        self.output_links = {}
        self.id = id

        self.route_queue_len = 0
        self.compute_queue_len = 0
        self.route_queue = simpy.Store(self.env)

        self.core_link = None
        self.core = None

        self.env.process(self.run())

    def connect(self, link: Link, neighbor_router):
        self.output_links[neighbor_router.id] = (link, neighbor_router)

    def bound_with_core(self, link: Link, core):
        self.core_link = link
        self.core = core

    def route(self, msg: Message, next_router_id: int):
        if next_router_id not in self.output_links:
            raise ValueError(f"There is no connection between Router{self.router_id} and Router{next_router_id}.")

        link, next_router = self.output_links[next_router_id]

        yield self.env.process(link.transmit(msg.data.size))
        # next_router.route_queue_len += 1
        yield next_router.route_queue.put(msg)

    def run(self):
        while True:
            msg = yield self.route_queue.get()
            # self.route_queue_len -= 1

            if msg.dst == self.id:
                yield self.env.process(self.core_link.transmit(msg.data.size))
                # self.compute_queue_len += 1
                # print(f"put data{data.index} into core{self.core.id}")
                logger.info(f"Time {self.env.now:.2f}: Finish routing data{msg.data.index} to router{self.id}.")
                self.core.data_queue.put(msg)
                logger.debug(f"router{self.id}'s data_queue_len is {self.core.data_len()}")
            else:
                # print(f"Router{self.id} is sending data{data.index} to router{data.dst} at time {self.env.now:.2f}")
                next_router = self.calculate_next_router(msg.dst)
                logger.info(f"Time {self.env.now:.2f}: Router{self.id} send data{msg.data.index} to router{next_router}.")
                self.env.process(self.route(msg, next_router))
                logger.info(f"Time {self.env.now:.2f}: Router{self.id} finished sending data{msg.data.index} to router{next_router}.")

    def calculate_next_router(self, target_id):
        if self.type == "XY":
            # switch id
            now_x, now_y = self.to_xy(self.id)
            tar_x, tar_y = self.to_xy(target_id)

            # X first
            if now_x != tar_x:
                if tar_x > now_x:
                    return self.to_x(now_x + 1, now_y)
                else:
                    return self.to_x(now_x - 1, now_y)
        
            # then Y
            if now_y != tar_y:
                if tar_y > now_y:
                    return self.to_x(now_x, now_y + 1)
                else:
                    return self.to_x(now_x, now_y - 1)
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
        self.links = []

    # connections between routers
    def build_connection(self):
        for id in range(self.x * self.y):
            self.id = id
            self.routers.append(Router(self.env, self.router_config, id, self.x, self.y))
        
        for row in range(self.x):
            for col in range(self.y):
                router_id = row * self.y + col
                # connect with the right Router
                if col < self.y - 1:
                    right_router_id = row * self.y + (col + 1)
                    link = Link(self.env, self.link_config)
                    # a couple of directed edges
                    self.routers[router_id].connect(link, self.routers[right_router_id])
                    self.routers[right_router_id].connect(link, self.routers[router_id])

                self.links.append(link)
                # connect with the down Router
                if row < self.x - 1:
                    down_router_id = (row + 1) * self.y + col
                    link = Link(self.env, self.link_config)
                    self.routers[router_id].connect(link, self.routers[down_router_id])
                    self.routers[down_router_id].connect(link, self.routers[router_id])

                self.links.append(link)
    
