from tracer.topology.dragonfly import DragonflyTopology
from tracer.topology.mesh import MeshTopology
from tracer.topology.ringroad import RingRoadTopology
from tracer.topology.torus import TorusTopology


def create_topology(arch_config):
    noc_type = arch_config.noc.type.lower()
    router_type = arch_config.noc.router.type.lower()
    x = arch_config.noc.x
    y = arch_config.noc.y

    if noc_type == "mesh" or router_type == "xy":
        return MeshTopology(x, y)
    if noc_type == "torus" or router_type == "torus_xy":
        return TorusTopology(x, y)
    if noc_type == "ringroad" or router_type == "ringroad":
        return RingRoadTopology(x, y)
    if noc_type == "dragonfly" or router_type == "dragonfly":
        return DragonflyTopology(x, y)
    raise ValueError(f"Unsupported tracer topology for noc.type={arch_config.noc.type} router.type={arch_config.noc.router.type}")
