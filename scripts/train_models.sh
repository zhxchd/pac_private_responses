#!/bin/bash
# note that the following script is purely sequential
# you can choose to parallize this based on your setup
# you may set CUDA_VISIBLE_DEVICES accordingly to run the jobs on different GPUs
# this is useful for the vision/nlp tasks which uses the GPU
# tabular tasks do not use GPU
set -e

source .venv/bin/activate
mkdir -p saved_models

SPLITS=(U X_0..127)
TABULAR_DATASETS=(adult bank)
VISION_DATASETS=(cifar10 cifar100)
NLP_DATASETS=(imdb ag_news)

for dataset in "${TABULAR_DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    python train_one_model_tabular.py "$dataset" "$split" xgboost
  done
done

for dataset in "${VISION_DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    python train_one_model_vision.py "$dataset" "$split" wide_resnet_28_10
  done
done

for dataset in "${NLP_DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    python train_one_model_nlp.py "$dataset" "$split" bert_small
  done
done
