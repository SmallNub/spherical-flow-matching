#!/bin/bash
# Decode precomputed encodings and run evaluation

python eval_encodings.py \
  --encoding_path workspace/experiments/sphere-small-small-cifar-10-32px/encoding/encoded_dataset.pt \
  --checkpoint workspace/experiments/sphere-small-small-cifar-10-32px/ \
  --output_dir workspace/experiments/sphere-small-small-cifar-10-32px/decoded_eval \
  --image_size 32 \
  --dataset_name cifar-10 \
  --batch_size 128 \
  --use_ema True
