# SLOTH

SLOTH is an automated framework for detecting and locating fail-slow failures in many-core systems.

# Quick Start

You can run SLOTH evaluator with the following command:

```shell
$ ./scripts/simulation.sh
```

And run SLOTH tracer with the following command:

```shell
$ ./scripts/analysis.sh
```

The scripts support the following parameters:

- `--workload`: The workload file path
- `--arch`: The hardware configuration path
- `--fail`: The fail-slow configuration path
- `--fragment`: The probe's parameter fragment
- `--type`: The probe's parameter type
- `--location`: The probe's parameter location
- `--plevel`: The probe's parameter level
- `--structure`: The probe's parameter structure
- `--log`: The log file path
- `--level`: The logging level
- `--times`: The number of inferences
- `--mapping`: The mapping file path
- `--report`: The path to write fail-slow report
- `--normal`: The path of trace without fail-slow
- `--detect`: The path of trace with fail-slow
- `--output`: The path to write overall.json
- `--record`: The path to write record.json