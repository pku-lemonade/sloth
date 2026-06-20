#!/bin/bash
python3 tracer/root_cause_analysis.py \
    --mapping data/mapping_example.json \
    --arch data/arch_example.json \
    --normal trace/example/normal_example \
    --detect trace/example/fail_example \
    --report trace/result/fail_example/report.json \
    --output trace/result/fail_example/overall.json \
    --record trace/result/fail_example/record.json
