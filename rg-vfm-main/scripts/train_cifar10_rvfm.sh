#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


#SBATCH --job-name=sphere-encoder-cifar10-rvfm
#SBATCH --output=slurm/sphere-encoder-cifar10-rvfm_%j.log
#SBATCH --error=slurm/sphere-encoder-cifar10-rvfm_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate sphere_hyper

python -m scripts.main_sphere --flow variational --geometry riemannian --support intrinsic --p0_distribution gaussian --data_path ../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/encoded_dataset.pt --num_epoch 3 --batch_size 8

python -m scripts.main_sphere \
  --flow variational \
  --geometry riemannian \
  --support intrinsic \
  --p0_distribution uniform \
  --data_path ../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/encoded_dataset.pt \
  --num_epoch 3 \
  --batch_size 8