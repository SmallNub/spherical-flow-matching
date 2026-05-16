#!/bin/bash

python eval_encodings.py \
  --encoding_path workspace/experiments/sphere-small-small-cifar-10-32px/encoding/output_encodings.npz \
  --checkpoint workspace/experiments/sphere-small-small-cifar-10-32px/ckpt \
  --output_dir workspace/experiments/sphere-small-small-cifar-10-32px/decoded_eval \
  --dataset_name cifar-10 \
  --batch_size 128 \
  --use_ema True \
  --dtype bfloat16 \
  --normalize_latents True \
  --eval_per_class False \
  --save_images False \
  --seed 42 \
  --deterministic False \
  --compile_model True
