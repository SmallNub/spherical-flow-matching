#!/bin/bash

python eval_encodings.py \
  --encoding_path workspace/experiments/sphere-small-small-animal-faces-256px/encoding/output_dataset.npz \
  --checkpoint workspace/experiments/sphere-small-small-animal-faces-256px/ckpt \
  --output_dir workspace/experiments/sphere-small-small-animal-faces-256px/decoded_eval \
  --dataset_name animal-faces \
  --batch_size 128 \
  --use_ema True \
  --dtype bfloat16 \
  --normalize_latents True \
  --save_images True \
  --seed 42 \
  --deterministic False \
  --compile_model True
