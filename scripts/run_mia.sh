#!/bin/bash
# the following jobs are parallelized (24 jobs in total)
set -e

source .venv/bin/activate
mkdir -p final_results

ARGS1=(0 1 2 3 4 5)
COUNT=(0 1 2 3)

for i in "${ARGS1[@]}"; do
  for j in "${COUNT[@]}"; do
    python run_mia.py "$i" &
  done
done

wait