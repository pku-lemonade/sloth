import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from common.arch_config import ArchConfig
from common.runtime_config import (
    LoggingConfig,
    MonitoringConfig,
    ProbeConfig,
    RecorderConfig,
    SimulatorConfig,
    SimulatorInputConfig,
    SimulatorRuntimeConfig,
    TRACE_END_CYCLE_DEFAULT,
)
from evaluater.architecture import Arch, SimulationRunSummary
from evaluater.sim_type import FailSlow, Workload

T = TypeVar("T", bound=BaseModel)


class SimulatorError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SLOTH many-core fail-slow simulator."
    )

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument(
        "--workload",
        default="data/workload_example.json",
        help="Path to the workload JSON file.",
    )
    input_group.add_argument(
        "--arch",
        default="data/arch_example.json",
        help="Path to the architecture JSON file.",
    )
    input_group.add_argument(
        "--failslow",
        "--fail",
        dest="failslow",
        default="data/fail_example.json",
        help="Path to the fail-slow JSON file.",
    )

    probe_group = parser.add_argument_group("Probe")
    probe_group.add_argument(
        "--probe-fragment",
        "--fragment",
        dest="probe_fragment",
        choices=("Exec", "Route", "Mem"),
        required=True,
        help="Probe fragment to collect.",
    )
    probe_group.add_argument(
        "--probe-kind",
        "--type",
        dest="probe_kind",
        choices=("Comm", "Comp", "IO"),
        required=True,
        help="Probe kind to collect.",
    )
    probe_group.add_argument(
        "--probe-location",
        "--location",
        dest="probe_location",
        choices=("Post", "Pre", "Surround"),
        default=ProbeConfig.model_fields["location"].default,
        help="Probe location relative to the instrumented operation.",
    )
    probe_group.add_argument(
        "--probe-level",
        "--plevel",
        dest="probe_level",
        choices=("Inst", "Stage"),
        default=ProbeConfig.model_fields["level"].default,
        help="Probe level to collect.",
    )
    probe_group.add_argument(
        "--probe-structure",
        "--structure",
        dest="probe_structure",
        choices=("List", "Sketch"),
        default=ProbeConfig.model_fields["structure"].default,
        help="Probe storage structure to use.",
    )

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument(
        "--noc-model",
        "--model",
        dest="noc_model",
        choices=("basic", "packet"),
        default="basic",
        help="NoC timing model to use.",
    )
    runtime_group.add_argument(
        "--inference-count",
        "--times",
        dest="inference_count",
        type=int,
        default=1,
        help="Number of inferences to simulate.",
    )

    recorder_group = parser.add_argument_group("Recorder")
    recorder_group.add_argument(
        "--recorder-hash",
        type=int,
        default=RecorderConfig.model_fields["hash"].default,
        help="Recorder sketch hash count.",
    )
    recorder_group.add_argument(
        "--recorder-bucket",
        type=int,
        default=RecorderConfig.model_fields["bucket"].default,
        help="Recorder sketch bucket count.",
    )
    recorder_group.add_argument(
        "--recorder-size",
        type=int,
        default=RecorderConfig.model_fields["size"].default,
        help="Recorder stage-two sketch size.",
    )
    recorder_group.add_argument(
        "--recorder-threshold",
        type=int,
        default=RecorderConfig.model_fields["threshold"].default,
        help="Recorder sketch promotion threshold.",
    )

    monitoring_group = parser.add_argument_group("Monitoring")
    monitoring_group.add_argument(
        "--trace-start-cycle",
        "--simstart",
        dest="trace_start_cycle",
        type=int,
        default=0,
        help="First cycle included in trace and resource monitoring output.",
    )
    monitoring_group.add_argument(
        "--trace-end-cycle",
        "--simend",
        dest="trace_end_cycle",
        type=int,
        default=TRACE_END_CYCLE_DEFAULT,
        help="Last cycle included in trace and resource monitoring output.",
    )
    monitoring_group.add_argument(
        "--enable-flow-trace",
        "--flow",
        dest="enable_flow_trace",
        action="store_true",
        help="Enable flow trace generation in the monitoring artifacts.",
    )

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--log-file",
        "--log",
        dest="log_file",
        default="logging/simulation.log",
        help="Path to the simulator log file.",
    )
    logging_group.add_argument(
        "--log-level",
        "--level",
        dest="log_level",
        choices=("debug", "info", "warning", "error", "critical"),
        default="debug",
        help="Logging verbosity for the simulator log file.",
    )

    return parser


def parse_simulator_config(parser: argparse.ArgumentParser) -> SimulatorConfig:
    args = parser.parse_args()
    try:
        return SimulatorConfig(
            inputs=SimulatorInputConfig(
                workload=args.workload,
                arch=args.arch,
                failslow=args.failslow,
            ),
            probe=ProbeConfig(
                fragment=args.probe_fragment,
                kind=args.probe_kind,
                location=args.probe_location,
                level=args.probe_level,
                structure=args.probe_structure,
            ),
            runtime=SimulatorRuntimeConfig(
                noc_model=args.noc_model,
                inference_count=args.inference_count,
            ),
            recorder=RecorderConfig(
                hash=args.recorder_hash,
                bucket=args.recorder_bucket,
                size=args.recorder_size,
                threshold=args.recorder_threshold,
            ),
            monitoring=MonitoringConfig(
                trace_start_cycle=args.trace_start_cycle,
                trace_end_cycle=args.trace_end_cycle,
                enable_flow_trace=args.enable_flow_trace,
            ),
            logging=LoggingConfig(
                log_file=args.log_file,
                log_level=args.log_level,
            ),
        )
    except ValidationError as exc:
        parser.exit(2, f"error: invalid simulator configuration\n{exc}\n")


def load_model(path: Path, model_type: type[T], label: str) -> T:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise SimulatorError(f"{label} file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SimulatorError(f"{label} file is not valid JSON: {path}: {exc}") from exc

    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise SimulatorError(f"{label} file failed validation: {path}\n{exc}") from exc


def setup_logging(config: LoggingConfig) -> None:
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=config.python_level(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=config.log_file,
        filemode="w",
    )


def print_section(title: str) -> None:
    print(title)


def print_field(label: str, value: str) -> None:
    print(f"  {label:<16} {value}")


def render_configuration(config: SimulatorConfig) -> None:
    print("SLOTH Simulator")
    print()
    print_section("Inputs")
    print_field("workload:", str(config.inputs.workload))
    print_field("arch:", str(config.inputs.arch))
    print_field("failslow:", str(config.inputs.failslow))
    print()
    print_section("Runtime")
    print_field("noc model:", config.runtime.noc_model)
    print_field("inferences:", str(config.runtime.inference_count))
    print_field(
        "trace window:",
        f"[{config.monitoring.trace_start_cycle}, {config.monitoring.trace_end_cycle}]",
    )
    print_field(
        "flow trace:",
        "enabled" if config.monitoring.enable_flow_trace else "disabled",
    )
    print_field("log file:", str(config.logging.log_file))
    print_field("log level:", config.logging.log_level)
    print()
    print_section("Recorder")
    print_field("hash:", str(config.recorder.hash))
    print_field("bucket:", str(config.recorder.bucket))
    print_field("size:", str(config.recorder.size))
    print_field("threshold:", str(config.recorder.threshold))
    print()
    print_section("Probe")
    print_field("fragment:", config.probe.fragment)
    print_field("kind:", config.probe.kind)
    print_field("location:", config.probe.location)
    print_field("level:", config.probe.level)
    print_field("structure:", config.probe.structure)
    print()


def render_summary(
    config: SimulatorConfig,
    summary: SimulationRunSummary,
    simulation_time: float,
) -> None:
    print_section("Summary")
    print_field("workload:", summary.workload_name)
    print_field("pe count:", str(summary.pe_count))
    print_field("cycles:", f"{summary.total_cycles:.0f}")
    print_field("wall time:", f"{simulation_time:.2f} s")
    print()
    print_section("Per-PE")
    for pe_summary in summary.per_pe_stats:
        print(
            "  "
            f"PE{pe_summary.pe_id:02d}  "
            f"tasks {pe_summary.processed_tasks}/{pe_summary.total_tasks}  "
            f"spm peak {pe_summary.max_buffer_usage} "
            f"[{pe_summary.remaining_capacity}/{pe_summary.buffer_capacity}]"
        )
    print()
    print_section("Artifacts")
    print_field("traces:", summary.trace_output_dir)
    print_field("monitor:", summary.monitor_output_dir)
    print_field("log:", str(config.logging.log_file))


def main() -> int:
    parser = build_parser()
    config = parse_simulator_config(parser)
    setup_logging(config.logging)
    render_configuration(config)

    print_section("Progress")
    print("  [1/3] Load input models...")
    arch_config = load_model(config.inputs.arch, ArchConfig, "Architecture")
    fail_slow = load_model(config.inputs.failslow, FailSlow, "Fail-slow")
    workload = load_model(config.inputs.workload, Workload, "Workload")
    print("  [1/3] Load input models... OK")

    print("  [2/3] Build architecture...")
    arch = Arch(
        arch=arch_config,
        program=[pe.insts for pe in workload.pes],
        fail=fail_slow,
        net_name=workload.name,
        fail_kind=str(config.inputs.failslow),
        model=config.runtime.noc_model,
        inference_time=config.runtime.inference_count,
        probe=config.probe.as_runtime_list(),
        recorder_config=config.recorder,
        monitoring_config=config.monitoring,
        stage=None,
    )
    print("  [2/3] Build architecture... OK")

    print("  [3/3] Run simulation...")
    start_time = time.time()
    summary = arch.run()
    simulation_time = time.time() - start_time
    print("  [3/3] Run simulation... OK")
    print()

    render_summary(config, summary, simulation_time)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SimulatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
