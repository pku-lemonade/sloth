import argparse
import json
import os
import sys
from types import SimpleNamespace

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from compiler.instruction_generator import config_analyzer, json_analyzer
from recorder.sketch_compressor import (
    EXACT_RETENTION_MODE,
    NO_EVICTION_POLICY,
    OLDEST_EVICTION_POLICY,
    SKETCH_RETENTION_MODE,
    available_retention_modes,
    available_stage2_eviction_policies,
    compress_traces,
)
from recorder.compression_types import CompressionResult
from recorder.online_recorder import TRACE_FORMAT_COMPRESSED
from recorder.storage_model import CompressionAggregateMetrics, CompressionStorageRecord
from recorder.trace_io import load_comm_summary, load_comm_trace, load_comp_summary, load_comp_trace
from tracer.analysis_context import AnalysisContext
from tracer.candidates.core_candidates import build_period_core_trace_layers, detect_core_candidates
from tracer.candidates.link_candidates.service import DEFAULT_LINK_DETECTOR, available_link_detectors, detect_link_candidates
from tracer.failrank.graph import CommGraph
from tracer.failrank.model import (
    DEFAULT_LINK_COUNT_WEIGHT,
    DEFAULT_LINK_SOFTMAX_BETA,
    DEFAULT_LINK_SUMMARY_BETA,
    DEFAULT_LINK_VARIANCE_WEIGHT,
    DEFAULT_PE_PROB_WEIGHT,
    DEFAULT_PE_SOFTMAX_BETA,
    FailSlows,
    Mesh,
)


parser = argparse.ArgumentParser()

DEFAULT_FAILRANK_ALPHA = 0.6
DEFAULT_FAILRANK_EPSILON = 1e-4
DEFAULT_FAILRANK_MAX_ITER = 1000
DEFAULT_FAILRANK_SUMMARY_THRESHOLD = 0.65
DEFAULT_LINK_SUMMARY_THRESHOLD = 0.8
DEFAULT_LINK_CANDIDATE_THRESHOLD = 5.0


def build_parser():
    parser.add_argument("--mapping", type=str, help="Workload mapping file")
    parser.add_argument("--arch", type=str, help="Architecture configuration file")
    parser.add_argument("--report", type=str, help="Report file")
    parser.add_argument("--normal", type=str, help="Path to trace without fail-slow")
    parser.add_argument("--detect", type=str, help="Path to real runtime trace")
    parser.add_argument("--hash", type=int, default=5, help="Fail-Slow sketch parameter")
    parser.add_argument("--bucket", type=int, default=1024, help="Fail-Slow sketch parameter")
    parser.add_argument("--size", type=int, default=8192, help="Fail-Slow sketch parameter")
    parser.add_argument("--threshold", type=int, default=10, help="Fail-Slow sketch parameter")
    parser.add_argument("--output", type=str, help="Path to record the max total overhead")
    parser.add_argument("--record", type=str, help="Path to record the max overhead for each variable")
    parser.add_argument(
        "--link-detector",
        choices=available_link_detectors(),
        default=DEFAULT_LINK_DETECTOR,
        help="Select the link candidate inference backend.",
    )
    parser.add_argument(
        "--link-candidate-retention",
        choices=available_retention_modes(),
        default=SKETCH_RETENTION_MODE,
        help="Select the communication retention mode used for link candidate detection.",
    )
    parser.add_argument(
        "--link-candidate-eviction",
        choices=("default",) + available_stage2_eviction_policies(),
        default="default",
        help="Select the stage-two eviction policy for link candidate retention experiments.",
    )
    parser.add_argument(
        "--link-candidate-stage2-size",
        type=int,
        default=None,
        help="Optional stage-two capacity override for link candidate retention experiments.",
    )
    parser.add_argument(
        "--failrank-alpha",
        type=float,
        default=DEFAULT_FAILRANK_ALPHA,
        help="FailRank damping factor.",
    )
    parser.add_argument(
        "--failrank-epsilon",
        type=float,
        default=DEFAULT_FAILRANK_EPSILON,
        help="FailRank convergence tolerance.",
    )
    parser.add_argument(
        "--failrank-max-iter",
        type=int,
        default=DEFAULT_FAILRANK_MAX_ITER,
        help="FailRank maximum iterations.",
    )
    parser.add_argument(
        "--failrank-summary-threshold",
        type=float,
        default=DEFAULT_FAILRANK_SUMMARY_THRESHOLD,
        help="FailRank summary threshold.",
    )
    parser.add_argument(
        "--link-summary-threshold",
        type=float,
        default=DEFAULT_LINK_SUMMARY_THRESHOLD,
        help="Link fail-slow summary threshold.",
    )
    parser.add_argument(
        "--link-candidate-threshold",
        type=float,
        default=DEFAULT_LINK_CANDIDATE_THRESHOLD,
        help="Threshold used by the link candidate detector.",
    )
    parser.add_argument(
        "--pe-softmax-beta",
        type=float,
        default=DEFAULT_PE_SOFTMAX_BETA,
        help="Softmax beta for PE FailRank summary probabilities.",
    )
    parser.add_argument(
        "--link-softmax-beta",
        type=float,
        default=DEFAULT_LINK_SOFTMAX_BETA,
        help="Softmax beta for link FailRank summary probabilities.",
    )
    parser.add_argument(
        "--link-summary-beta",
        type=float,
        default=DEFAULT_LINK_SUMMARY_BETA,
        help="Softmax beta for final physical-link summary.",
    )
    parser.add_argument(
        "--link-count-weight",
        type=float,
        default=DEFAULT_LINK_COUNT_WEIGHT,
        help="Weight for dependency-graph link count evidence.",
    )
    parser.add_argument(
        "--link-variance-weight",
        type=float,
        default=DEFAULT_LINK_VARIANCE_WEIGHT,
        help="Weight for link-candidate variance evidence.",
    )
    parser.add_argument(
        "--pe-prob-weight",
        type=float,
        default=DEFAULT_PE_PROB_WEIGHT,
        help="Weight for propagated PE probability evidence in link scoring.",
    )


def ensure_output_paths(*paths: str) -> None:
    for path in paths:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def write_report(path: str, failslows: FailSlows) -> None:
    ensure_output_paths(path)
    with open(path, "w", encoding="utf-8") as file:
        print(failslows.model_dump_json(indent=4), file=file)


def load_persistent_state(record_path: str, output_path: str):
    if not os.path.isfile(record_path):
        storage_record = CompressionStorageRecord()
    else:
        with open(record_path, "r", encoding="utf-8") as file:
            storage_record = CompressionStorageRecord.from_record_payload(json.load(file))

    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as file:
            aggregate_metrics = CompressionAggregateMetrics.from_output_payload(json.load(file))
    else:
        aggregate_metrics = CompressionAggregateMetrics()

    return storage_record, aggregate_metrics


def resolve_link_candidate_retention(args):
    if args.link_candidate_eviction == "default":
        eviction_policy = NO_EVICTION_POLICY if args.link_candidate_retention == EXACT_RETENTION_MODE else OLDEST_EVICTION_POLICY
    else:
        eviction_policy = args.link_candidate_eviction
    return args.link_candidate_retention, eviction_policy, args.link_candidate_stage2_size


def is_compressed_trace_dir(trace_dir: str) -> bool:
    meta_path = os.path.join(trace_dir, "trace_meta.json")
    if not os.path.isfile(meta_path):
        return False
    with open(meta_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("format") == TRACE_FORMAT_COMPRESSED


def load_trace_inputs(trace_dir: str):
    if is_compressed_trace_dir(trace_dir):
        compression = CompressionResult(
            comm=load_comm_summary(os.path.join(trace_dir, "comm_trace.json")),
            comp=load_comp_summary(os.path.join(trace_dir, "comp_trace.json")),
        )
        return SimpleNamespace(
            compressed=True,
            compression=compression,
            raw_comm_trace=None,
            raw_comp_trace=None,
        )

    raw_comm_trace = load_comm_trace(os.path.join(trace_dir, "comm_trace.json"))
    raw_comp_trace = load_comp_trace(os.path.join(trace_dir, "comp_trace.json"))
    return SimpleNamespace(
        compressed=False,
        compression=None,
        raw_comm_trace=raw_comm_trace,
        raw_comp_trace=raw_comp_trace,
    )


def copy_online_compression_metrics(detect_trace_dir: str, record_path: str, output_path: str) -> None:
    source_record = os.path.join(detect_trace_dir, "record.json")
    source_output = os.path.join(detect_trace_dir, "overall.json")
    ensure_output_paths(record_path, output_path)

    if os.path.isfile(source_record):
        with open(source_record, "r", encoding="utf-8") as src, open(record_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
    if os.path.isfile(source_output):
        with open(source_output, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())


def max_inference_period(comp_summary) -> int:
    return max((trace.inference_time for trace in comp_summary.trace), default=-1)


def get_or_create_compression(trace_inputs, args):
    if trace_inputs.compressed:
        return SimpleNamespace(compression=trace_inputs.compression)

    comm_retention_mode, stage2_eviction_policy, comm_stage2_size = resolve_link_candidate_retention(args)
    return compress_traces(
        trace_inputs.raw_comm_trace,
        trace_inputs.raw_comp_trace,
        args.hash,
        args.bucket,
        args.size,
        args.threshold,
        comm_retention_mode=comm_retention_mode,
        stage2_eviction_policy=stage2_eviction_policy,
        comm_stage2_size=comm_stage2_size,
    )


def get_detect_comm_events(detect_trace_inputs):
    if detect_trace_inputs.compressed:
        return detect_trace_inputs.compression.comm.trace
    return detect_trace_inputs.raw_comm_trace.trace


def run_root_cause_analysis(args) -> FailSlows:
    network = json_analyzer(args.mapping)
    arch_config = config_analyzer(args.arch)
    context = AnalysisContext.from_inputs(network, arch_config)

    normal_trace = load_trace_inputs(args.normal)
    detect_trace = load_trace_inputs(args.detect)

    normal_compression = get_or_create_compression(normal_trace, args)
    detect_compression = get_or_create_compression(detect_trace, args)

    if detect_trace.compressed:
        copy_online_compression_metrics(args.detect, args.record, args.output)
    else:
        storage_record, aggregate_metrics = load_persistent_state(args.record, args.output)
        storage_record.merge(detect_compression.storage_model)
        aggregate_metrics.update(storage_record, args.hash, args.bucket, detect_compression.effective_stage2_size)

        ensure_output_paths(args.record, args.output, args.report)
        with open(args.record, "w", encoding="utf-8") as file:
            json.dump(storage_record.to_record_payload(), file, indent=4)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(aggregate_metrics.to_output_payload(), file, indent=4)

    detect_comm_events = get_detect_comm_events(detect_trace)
    raw_comp_trace = None if detect_trace.compressed else detect_trace.raw_comp_trace

    print("=" * 40)
    print("Detecting potential failslow links:")
    link_candidates = detect_link_candidates(
        backend_name=args.link_detector,
        normal_summary=normal_compression.compression.comm,
        detect_summary=detect_compression.compression.comm,
        context=context,
        threshold=args.link_candidate_threshold,
    )

    failslows = FailSlows()
    max_period = max_inference_period(detect_compression.compression.comp)
    mesh = Mesh(context)
    for period in range(max_period + 1):
        layer_traces = build_period_core_trace_layers(
            period=period,
            compressed_comp_summary=detect_compression.compression.comp,
            raw_comp_trace=raw_comp_trace,
            context=context,
        )
        comm_trace_inference = [trace for trace in detect_comm_events if trace.inference_time == period]

        print("=" * 40)
        print("Building RCA dependency graph:")
        mesh = Mesh(context)
        comm_graph = CommGraph(context, comm_trace_inference, mesh)
        print(f"[Info] Finish building comm_graph, {len(comm_graph.nodes)} nodes and {len(comm_graph.edges)} edges in total.")
        comm_graph.construct_mesh()
        print("[Info] Finish building dependency graph.")

        print("=" * 40)
        print("Initializing failrank values:")
        for layer_trace in layer_traces:
            if not layer_trace:
                continue
            layer_id = layer_trace[0].layer_id
            observations = detect_core_candidates(layer_trace, context.layer_mapping[layer_id])
            for observation in observations:
                mesh.core_prob_init(
                    layer_group=context.layer_to_group[layer_id],
                    pe_id=observation.pe_id,
                    flops=observation.flops,
                    start_time=observation.start_time,
                    end_time=observation.end_time,
                )
        print("[Info] Finish initializing core PR values.")

        mesh.link_prob_init()
        mesh.link_variance_init(link_candidates)
        print("[Info] Finish initializing link weights.")

        print("=" * 40)
        print("Root Cause Analysis:")
        mesh.failrank(
            alpha=args.failrank_alpha,
            tol=args.failrank_epsilon,
            max_iter=args.failrank_max_iter,
        )
        period_failslow = mesh.failrank_summary(
            threshold=args.failrank_summary_threshold,
            pe_softmax_beta=args.pe_softmax_beta,
            link_softmax_beta=args.link_softmax_beta,
            link_count_weight=args.link_count_weight,
            link_variance_weight=args.link_variance_weight,
            pe_prob_weight=args.pe_prob_weight,
        )
        for failure in period_failslow.data:
            failslows.insert(failure)

    link_failures = mesh.link_summary(
        threshold=args.link_summary_threshold,
        beta=args.link_summary_beta,
    )
    for failure in link_failures.data:
        failslows.insert(failure)

    return failslows


def run_root_cause_analysis_with_config(
    *,
    mapping_path: str,
    arch_path: str,
    normal_trace_dir: str,
    detect_trace_dir: str,
    report_path: str,
    output_path: str,
    record_path: str,
    link_detector: str = DEFAULT_LINK_DETECTOR,
    link_candidate_retention: str = SKETCH_RETENTION_MODE,
    link_candidate_eviction: str = "default",
    link_candidate_stage2_size: int | None = None,
    hash_num: int = 5,
    bucket_num: int = 1024,
    sketch_size: int = 8192,
    sketch_threshold: int = 10,
    failrank_alpha: float = DEFAULT_FAILRANK_ALPHA,
    failrank_epsilon: float = DEFAULT_FAILRANK_EPSILON,
    failrank_max_iter: int = DEFAULT_FAILRANK_MAX_ITER,
    failrank_summary_threshold: float = DEFAULT_FAILRANK_SUMMARY_THRESHOLD,
    link_summary_threshold: float = DEFAULT_LINK_SUMMARY_THRESHOLD,
    link_candidate_threshold: float = DEFAULT_LINK_CANDIDATE_THRESHOLD,
    pe_softmax_beta: float = DEFAULT_PE_SOFTMAX_BETA,
    link_softmax_beta: float = DEFAULT_LINK_SOFTMAX_BETA,
    link_summary_beta: float = DEFAULT_LINK_SUMMARY_BETA,
    link_count_weight: float = DEFAULT_LINK_COUNT_WEIGHT,
    link_variance_weight: float = DEFAULT_LINK_VARIANCE_WEIGHT,
    pe_prob_weight: float = DEFAULT_PE_PROB_WEIGHT,
):
    args = SimpleNamespace(
        mapping=mapping_path,
        arch=arch_path,
        report=report_path,
        normal=normal_trace_dir,
        detect=detect_trace_dir,
        hash=hash_num,
        bucket=bucket_num,
        size=sketch_size,
        threshold=sketch_threshold,
        output=output_path,
        record=record_path,
        link_detector=link_detector,
        link_candidate_retention=link_candidate_retention,
        link_candidate_eviction=link_candidate_eviction,
        link_candidate_stage2_size=link_candidate_stage2_size,
        failrank_alpha=failrank_alpha,
        failrank_epsilon=failrank_epsilon,
        failrank_max_iter=failrank_max_iter,
        failrank_summary_threshold=failrank_summary_threshold,
        link_summary_threshold=link_summary_threshold,
        link_candidate_threshold=link_candidate_threshold,
        pe_softmax_beta=pe_softmax_beta,
        link_softmax_beta=link_softmax_beta,
        link_summary_beta=link_summary_beta,
        link_count_weight=link_count_weight,
        link_variance_weight=link_variance_weight,
        pe_prob_weight=pe_prob_weight,
    )
    failslows = run_root_cause_analysis(args)
    write_report(report_path, failslows)
    return failslows


def main():
    build_parser()
    args = parser.parse_args()
    failslows = run_root_cause_analysis(args)
    write_report(args.report, failslows)


if __name__ == "__main__":
    main()
