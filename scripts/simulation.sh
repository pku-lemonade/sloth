#!/bin/bash

python evaluate.py \
    --workload data/workload_example.json \
    --arch data/arch_example.json \
    --fail data/fail_example.json \
    --fragment Exec \
    --type Comp \
    --location Surround \
    --plevel Inst \
    --structure Sketch \
    --log logging/simulation.log \
    --level debug \
    --times 16