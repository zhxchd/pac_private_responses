#!/bin/bash
# the following jobs are parallelized (24 jobs in total)
set -e

source .venv/bin/activate
mkdir -p final_results

ZERO_ARGS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
ONE_AGGS=(0 2 4 6 8 10 12 14)

for i in "${ZERO_ARGS[@]}"; do
  python run_priv_responses.py 0 "$i" &
done

for i in "${ONE_AGGS[@]}"; do
  python run_priv_responses.py 1 "$i" &
done

wait