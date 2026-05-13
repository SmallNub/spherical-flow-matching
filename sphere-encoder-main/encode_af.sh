#!/bin/bash

python encode_dataset.py \
  --data_path workspace/datasets/animal-faces \
  --checkpoint workspace/experiments/sphere-small-small-animal-faces-256px/ckpt \
  --output_path workspace/experiments/sphere-small-small-animal-faces-256px/encoding \
  --output_name encoded_dataset.npz \
  --dataset_name animal-faces \
  --batch_size 64 \
  --num_workers 8 \
  --seed 42 \
  --deterministic False \
  --save_dtype bfloat16 \
  --use_ema True \
  --compile_model True