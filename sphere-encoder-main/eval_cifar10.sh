#!/bin/bash
# Decode encoded dataset and run evaluation (FID / ISC)
# Config-driven (reads cfg.json automatically from checkpoint dir)

python eval_encodings.py \
  --encoding_path workspace/experiments/sphere-small-small-cifar-10-32px/encoding/encoded_dataset.pt \
  --checkpoint workspace/experiments/sphere-small-small-cifar-10-32px/ckpt \
  --output_dir workspace/experiments/sphere-small-small-cifar-10-32px/decoded_eval \
  --dataset_name cifar-10 \
  --batch_size 128 \
  --use_ema True \
  --dtype bfloat16 \
  --normalize_latents True \
  --save_images True \
  --seed 42 \
  --deterministic False \
  --compile_model True
