#!/bin/bash

DATASETS=(bank adult cifar10 cifar100 imdb ag_news)

for dataset in "${DATASETS[@]}"; do
  python generate_random_splits.py "$dataset"
done
