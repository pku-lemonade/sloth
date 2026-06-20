from dataclasses import dataclass
from typing import Any

from tracer.topology import create_topology


@dataclass(frozen=True)
class AnalysisContext:
    arch_config: Any
    network: Any
    layer_to_group: dict[int, int]
    layer_group_divide: list[list[int]]
    layer_mapping: list[list[int]]
    topology: Any

    @property
    def mesh_x(self) -> int:
        return self.topology.x

    @property
    def mesh_y(self) -> int:
        return self.topology.y

    @property
    def pe_count(self) -> int:
        return self.topology.node_count

    @property
    def layer_count(self) -> int:
        return len(self.network.layers)

    @property
    def group_count(self) -> int:
        return len(self.layer_group_divide)

    def get_id(self, x: int, y: int) -> int:
        return x * self.arch_config.core.y + y

    @classmethod
    def from_inputs(cls, network: Any, arch_config: Any) -> "AnalysisContext":
        layer_to_group = {}
        layer_group_divide = []
        layer_mapping = [[] for _ in range(len(network.layers))]

        cur_layer_id = 0
        for layer_id, layer in enumerate(network.layers):
            if layer.layer_group_id != network.layers[cur_layer_id].layer_group_id:
                group_layers = []
                for grouped_layer_id in range(cur_layer_id, layer_id):
                    layer_to_group[grouped_layer_id] = network.layers[cur_layer_id].layer_group_id
                    group_layers.append(grouped_layer_id)
                layer_group_divide.append(group_layers)
                cur_layer_id = layer_id

            for output in layer.output_feature:
                for block in output.blocks:
                    for pe in block.cores:
                        layer_mapping[layer_id].append(pe.x * arch_config.core.y + pe.y)

        final_group_layers = []
        for grouped_layer_id in range(cur_layer_id, len(network.layers)):
            layer_to_group[grouped_layer_id] = network.layers[cur_layer_id].layer_group_id
            final_group_layers.append(grouped_layer_id)
        layer_group_divide.append(final_group_layers)

        return cls(
            arch_config=arch_config,
            network=network,
            layer_to_group=layer_to_group,
            layer_group_divide=layer_group_divide,
            layer_mapping=layer_mapping,
            topology=create_topology(arch_config),
        )
