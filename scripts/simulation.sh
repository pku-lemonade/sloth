#!/bin/bash

python3 evaluate.py \
    --workload data/workload_example.json \
    --arch data/arch_example.json \
    --failslow data/fail_example.json \
    --probe-fragment Exec \
    --probe-kind Comp \
    --inference-count 16 \
    --recorder-threshold 1 \
    --log-level error