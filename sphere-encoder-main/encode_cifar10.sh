#!/bin/bash
# Encode CIFAR-10 dataset with the trained sphere-encoder model
# Uses exact parameters from training
# Encodes both train and test splits into one file

python dataset.py \
  --data_path workspace/datasets/cifar-10 \
  --checkpoint workspace/experiments/sphere-small-small-cifar-10-32px/ckpt \
  --output_path workspace/experiments/sphere-small-small-cifar-10-32px/encoding \
  --image_size 32 \
  --batch_size 128 \
  --num_workers 8 \
  --dataset_name cifar-10 \
  --split all \
  --vit_enc_model_size small \
  --vit_dec_model_size small \
  --vit_enc_latent_mlp_mixer_depth 2 \
  --vit_dec_latent_mlp_mixer_depth 2 \
  --affine_latent_mlp_mixer True \
  --cond_generator True \
  --pixel_head_type conv \
  --compression_ratio 3 \
  --noise_sigma_max_angle 80 \
  --num_classes 10 \
  --strict_ckpt False
