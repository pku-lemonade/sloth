# SLOTH

SLOTH is a Python framework for simulating fail-slow behavior in many-core
accelerator systems and localizing the likely root cause from runtime traces.

The repository contains three main components:

- **Evaluator**: executes a mapped workload on a configurable NoC architecture.
- **Recorder**: collects probe data online and writes compressed trace summaries.
- **Tracer**: runs fail-slow detection and root-cause analysis from compressed or
  legacy raw traces.

## Repository Layout

```text
common/      Shared configuration, distributions, and runtime settings
compiler/    Workload and probing utilities
evaluater/   Many-core simulator and NoC models
recorder/    Online trace compression and trace I/O
tracer/      Candidate detection, FailRank, topology support, and baselines
scripts/     Convenience scripts for simulation, detection, and experiments
data/        Small example inputs and optional local experiment inputs
```

Generated traces, logs, caches, sweep outputs, and large experiment datasets are
ignored by default. Keep reusable examples small enough to review.

## Requirements

SLOTH targets Python 3.10+ and uses the packages listed in
`requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Simulation

Run the evaluator with a workload, architecture, fail-slow description, and probe
configuration:

```bash
python3 evaluate.py \
  --workload data/workload_example.json \
  --arch data/arch_example.json \
  --failslow data/fail_example.json \
  --probe-fragment Exec \
  --probe-kind Comp \
  --inference-count 1 \
  --log-level info
```

The probe defaults match the paper configuration:

```text
Location = Surround
Level    = Stage
Structure= Sketch
```

The evaluator records traces online. At runtime, each collected probe event is
passed to the Recorder and compressed directly. The raw trace does not need to be
written and recompressed by the tracer.

Compressed traces are written to:

```text
trace/<workload_name>/<failslow_file_stem>/
```

Each trace directory contains:

```text
comm_trace.json    compressed communication summaries
comp_trace.json    compressed compute summaries
record.json        compact-field/storage record
overall.json       aggregate compression memory summary
trace_meta.json    trace format and Recorder parameter metadata
```

Recorder parameters can be adjusted from the evaluator CLI:

```bash
--recorder-hash 5
--recorder-bucket 1024
--recorder-size 8192
--recorder-threshold 10
```

## Running Detection

Run root-cause analysis with a mapping, architecture, normal trace, and detected
trace:

```bash
python3 tracer/root_cause_analysis.py \
  --mapping <mapping.json> \
  --arch <arch.json> \
  --normal <normal-trace-dir> \
  --detect <fail-trace-dir> \
  --report trace/result/report.json \
  --output trace/result/overall.json \
  --record trace/result/record.json
```

The tracer checks `trace_meta.json` automatically. If a trace directory is marked
as compressed, `comm_trace.json` and `comp_trace.json` are consumed directly. If
the directory is an older raw trace directory, the tracer keeps the legacy
offline compression path for compatibility.

The detection report is written as a JSON list of predicted fail-slow components.

## Topology Support

The architecture file controls the NoC topology used by simulation and
detection. The topology-aware tracer code is under `tracer/topology/`, and the
simulator-side NoC behavior is implemented in `evaluater/noc.py`.

Use the architecture config that matches the trace and mapping:

```bash
python3 tracer/root_cause_analysis.py \
  --mapping <mapping.json> \
  --arch <arch.json> \
  --normal <normal-trace-dir> \
  --detect <fail-trace-dir> \
  --report <report.json> \
  --output <overall.json> \
  --record <record.json>
```

## Useful CLI Options

Evaluator options:

- `--workload`: workload JSON path.
- `--arch`: architecture JSON path.
- `--failslow`: fail-slow JSON path.
- `--probe-fragment`: probe fragment, one of `Exec`, `Route`, or `Mem`.
- `--probe-kind`: probe kind, one of `Comm`, `Comp`, or `IO`.
- `--probe-location`: probe location, default `Surround`.
- `--probe-level`: probe level, default `Stage`.
- `--probe-structure`: probe storage structure, default `Sketch`.
- `--noc-model`: NoC timing model.
- `--inference-count`: number of inferences to simulate.
- `--trace-start-cycle`, `--trace-end-cycle`: trace collection window.
- `--enable-flow-trace`: emit detailed flow traces under `gen/`.
- `--log-file`, `--log-level`: simulator logging controls.

Tracer options:

- `--mapping`: workload mapping JSON path.
- `--normal`: trace directory without fail-slow.
- `--detect`: trace directory to diagnose.
- `--report`: output path for predicted fail-slow components.
- `--output`: output path for aggregate compression metrics.
- `--record`: output path for detailed compression storage records.
- `--link-detector`: link candidate detector backend.
- `--failrank-alpha`: FailRank damping factor.
- `--failrank-summary-threshold`: PE/link summary threshold.
- `--link-summary-threshold`: final link summary threshold.
- `--link-candidate-threshold`: link candidate detector threshold.
