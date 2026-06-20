from dataclasses import dataclass
from typing import Any

from recorder.compression_types import CompSummary
from recorder.trace_format import CompInst


@dataclass(frozen=True)
class CoreCandidateObservation:
    pe_id: int
    flops: float
    start_time: int
    end_time: int


def build_period_core_trace_layers(
    period: int,
    compressed_comp_summary: CompSummary,
    context,
    raw_comp_trace=None,
) -> list[list[Any]]:
    layer_traces = [[] for _ in range(context.layer_count)]
    seen = [set() for _ in range(context.layer_count)]

    for inst_trace in compressed_comp_summary.trace:
        if inst_trace.inference_time != period:
            continue
        layer_traces[inst_trace.layer_id].append(inst_trace)
        seen[inst_trace.layer_id].add(inst_trace.pe_id)

    if raw_comp_trace is not None:
        for inst_trace in raw_comp_trace.trace:
            if inst_trace.inference_time != period:
                continue
            if inst_trace.pe_id in seen[inst_trace.layer_id]:
                continue
            layer_traces[inst_trace.layer_id].append(inst_trace)

    return layer_traces


def detect_core_candidates(trace, layer_mapping: list[int]) -> list[CoreCandidateObservation]:
    total_flops = {pe_id: 0.0 for pe_id in layer_mapping}
    instruction_count = {pe_id: 0 for pe_id in layer_mapping}
    start_time = {}
    end_time = {}

    for inst_trace in trace:
        if hasattr(inst_trace, "start_time") and hasattr(inst_trace, "end_time"):
            duration = inst_trace.end_time - inst_trace.start_time
        else:
            duration = 0

        trace_count = getattr(inst_trace, "count", 1)

        if isinstance(inst_trace, CompInst) and duration > 0:
            total_flops[inst_trace.pe_id] += inst_trace.flops / duration
        elif hasattr(inst_trace, "flops"):
            total_flops[inst_trace.pe_id] += inst_trace.flops * trace_count

        instruction_count[inst_trace.pe_id] += trace_count
        start_time[inst_trace.pe_id] = min(inst_trace.start_time, start_time.get(inst_trace.pe_id, inst_trace.start_time))
        end_time[inst_trace.pe_id] = max(inst_trace.end_time, end_time.get(inst_trace.pe_id, inst_trace.end_time))

    observations = []
    for pe_id in layer_mapping:
        if instruction_count[pe_id] == 0:
            continue
        observations.append(
            CoreCandidateObservation(
                pe_id=pe_id,
                flops=total_flops[pe_id] / instruction_count[pe_id],
                start_time=start_time[pe_id],
                end_time=end_time[pe_id],
            )
        )
    return observations
