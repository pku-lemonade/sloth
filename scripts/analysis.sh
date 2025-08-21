#!/bin/bash

python tracer/root_cause_analysis.py \
    --mapping data/mapping_example.json \
    --arch data/arch_example.json \
    --report trace/result/report.json \
    --normal trace/example/normal_example \
    --detect trace/example/rca_example \
    --output trace/result/overall.json \
    --record trace/result/record.json