#!/bin/bash

python eval_quality_diversity.py \
  --encoding_path workspace/experiments/sphere-small-small-cifar-10-32px/encoding/output_encodings.npz \
  --checkpoint workspace/experiments/sphere-small-small-cifar-10-32px/ckpt \
  --output_dir workspace/experiments/sphere-small-small-cifar-10-32px/decoded_eval \
  --dataset_name cifar-10 \
  --batch_size 16 \
  --use_ema True \
  --dtype bfloat16 \
  --normalize_latents True \
  --eval_per_class False \
  --save_images False \
  --seed 42 \
  --deterministic False \
  --compile_model True \
  --nearest_k 5 \
  --use_isc False \
  --metrics_output workspace/experiments/sphere-small-small-cifar-10-32px/decoded_eval/metrics_summary.json
