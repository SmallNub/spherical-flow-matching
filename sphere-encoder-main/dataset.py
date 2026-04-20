import datetime
import glob
import json
import argparse
import os
import os.path as osp
import time
import logging
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
import wandb
from cli_utils import str2bool
from sphere.ema import SimpleEMA
from sphere.loader import create_loader, ListDataset
from sphere.logger import append_log, setup_logging
from sphere.loss import ReconstructionLoss
from sphere.model import G
from sphere.utils import cosine_scheduler, load_ckpt, save_ckpt, visualize
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Inference")
# --- directory
parser.add_argument("--data_path", type=str, required=True, help="path to dataset")
parser.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint")
parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
parser.add_argument("--output_name", type=str, default="encoded_dataset.pt")
# --- model
parser.add_argument("--model_size", type=str, default="small", choices=["xsmall", "small", "base", "large", "xlarge", "huge", "giant"])
parser.add_argument("--token_channels", type=int, default=16)
parser.add_argument("--num_classes", type=int, default=0)
parser.add_argument("--in_context_size", type=int, default=0)
parser.add_argument("--halve_model_size", type=str2bool, default=False)
parser.add_argument("--spherify_model", type=str2bool, default=False)
parser.add_argument("--vit_enc_latent_mlp_mixer_depth", type=int, default=0)
parser.add_argument("--vit_dec_latent_mlp_mixer_depth", type=int, default=0)
parser.add_argument("--affine_latent_mlp_mixer", type=str2bool, default=True)
# --- dataset
parser.add_argument("--image_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--load_from_zip", type=str2bool, default=False)
parser.add_argument("--max_samples", type=int, default=-1)


def main(args):
    # setup logging
    setup_logging()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    model = G(
        input_size=args.image_size,
        vit_enc_model_size=args.model_size,
        vit_dec_model_size=args.model_size,
        token_channels=args.token_channels,
        num_classes=args.num_classes,
        in_context_size=args.in_context_size,
        halve_model_size=args.halve_model_size,
        spherify_model=args.spherify_model,
        vit_enc_latent_mlp_mixer_depth=args.vit_enc_latent_mlp_mixer_depth,
        vit_dec_latent_mlp_mixer_depth=args.vit_dec_latent_mlp_mixer_depth,
        affine_latent_mlp_mixer=args.affine_latent_mlp_mixer,
    )
    model.to(device)
    model.eval()

    # load checkpoint
    load_ckpt(model, args.checkpoint, strict=True)

    # create dataset
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ListDataset(
        root=args.data_path,
        split=args.split,
        transform=transform,
        max_samples=args.max_samples,
        load_from_zip=args.load_from_zip,
    )

    # create dataloader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # encode dataset
    encodings = []
    paths = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
                x = x.to(device)
            else:
                x = batch.to(device)
                y = None

            z = model.encoder(x, y)  # [B, N, D]
            encodings.append(z.cpu())
            # Note: if you want to save paths, you can modify dataset to return paths

    encodings = torch.cat(encodings, dim=0)  # [total_samples, N, D]

    # save
    output_path = os.path.join(args.data_path, args.output_name)
    torch.save(encodings, output_path)
    logger.info(f"Saved encoded dataset to {output_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)