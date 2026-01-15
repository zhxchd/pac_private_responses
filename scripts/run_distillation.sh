#!/bin/bash
# you can choose to parallize this based on your setup
# you may set CUDA_VISIBLE_DEVICES accordingly to run the jobs on different GPUs
set -e

source .venv/bin/activate
mkdir -p final_results

COUNT=(0 1 2 3 4)
for i in "${COUNT[@]}"; do
    python run_distillation.py
done
