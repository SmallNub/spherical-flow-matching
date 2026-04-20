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
from tqdm import tqdm
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


def find_checkpoint(checkpoint_path):
    """Find checkpoint file if given a directory."""
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    elif os.path.isdir(checkpoint_path):
        # List all .pth files and pick the latest one
        pth_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.pth")))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in {checkpoint_path}")
        latest_ckpt = pth_files[-1]
        logger.info(f"Found checkpoint: {latest_ckpt}")
        return latest_ckpt
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Inference")
# --- directory
parser.add_argument("--data_path", type=str, required=True, help="path to dataset")
parser.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint")
parser.add_argument("--output_path", type=str, default=None, help="path to save encoded dataset (default: data_path)")
parser.add_argument("--output_name", type=str, default="encoded_dataset.pt")
parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="which split to encode (for CIFAR-10/CIFAR-100), 'all' encodes both")
# --- dataset
parser.add_argument(
    "--dataset_name",
    type=str,
    default="cifar-10",
    choices=[
        "cifar-10",
        "cifar-100",
        "food-101",
        "flowers-102",
        "animal-faces",
        "imagenet",
    ],
)
# --- model
parser.add_argument("--model_size", type=str, default="small", choices=["xsmall", "small", "base", "large", "xlarge", "huge", "giant"])
parser.add_argument("--vit_enc_model_size", type=str, default="base")
parser.add_argument("--vit_dec_model_size", type=str, default="base")
parser.add_argument("--token_channels", type=int, default=16)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--in_context_size", type=int, default=0)
parser.add_argument("--halve_model_size", type=str2bool, default=False)
parser.add_argument("--spherify_model", type=str2bool, default=False)
parser.add_argument("--vit_enc_latent_mlp_mixer_depth", type=int, default=2)
parser.add_argument("--vit_dec_latent_mlp_mixer_depth", type=int, default=2)
parser.add_argument("--affine_latent_mlp_mixer", type=str2bool, default=True)
parser.add_argument("--cond_generator", type=str2bool, default=True)
parser.add_argument("--pixel_head_type", type=str, default="linear", choices=["linear", "conv"])
# --- noise
parser.add_argument("--noise_sigma_max_angle", type=int, default=85)
# --- latent & noise
parser.add_argument("--compression_ratio", type=float, default=3.0)
parser.add_argument("--latent_resolution", type=str, default="high", choices=["low", "high"])
# --- ema
parser.add_argument("--use_ema", type=str2bool, default=True)
parser.add_argument("--ema_model_decay", type=float, default=0.9997)
# --- dataset
parser.add_argument("--image_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--load_from_zip", type=str2bool, default=False)
parser.add_argument("--max_samples", type=int, default=-1)
parser.add_argument("--strict_ckpt", type=str2bool, default=True, help="whether to strictly match checkpoint dimensions")
# --- additional parameters for proper model reconstruction
parser.add_argument("--patch_size", type=int, default=None, help="patch size (auto-calculated if None)")


def main(args):
    # determine output directory
    output_dir = args.output_path if args.output_path is not None else args.data_path
    os.makedirs(output_dir, exist_ok=True)
    
    # setup logging
    log_path = os.path.join(output_dir, "encode.log")
    setup_logging(log_path)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # calculate derived parameters (same as in train.py)
    if args.patch_size is None:
        if args.image_size in [32, 64]:
            args.patch_size = 4
        elif args.image_size in [128]:
            args.patch_size = 8
        elif args.image_size in [256]:
            args.patch_size = 16
        elif args.image_size in [512]:
            args.patch_size = 32
        else:
            args.patch_size = 16  # default fallback
        
        # apply latent_resolution scaling (high = patch_size // 2)
        args.patch_size = args.patch_size // 2
    
    # compute token channels based on compression_ratio
    latent_resolution_value = args.image_size // args.patch_size
    computed_token_channels = int(
        3 * args.image_size**2 / latent_resolution_value**2 / args.compression_ratio
    )
    
    # override token_channels if computed value differs significantly
    if args.token_channels != computed_token_channels:
        logger.info(f"Overriding token_channels: {args.token_channels} -> {computed_token_channels}")
        args.token_channels = computed_token_channels
    
    logger.info(f"Model params: patch_size={args.patch_size}, token_channels={args.token_channels}, latent_resolution={latent_resolution_value}")

    # use separate enc/dec model sizes if specified, otherwise use model_size
    vit_enc_model_size = args.vit_enc_model_size if args.vit_enc_model_size != "base" else args.model_size
    vit_dec_model_size = args.vit_dec_model_size if args.vit_dec_model_size != "base" else args.model_size

    # create model
    model = G(
        input_size=args.image_size,
        patch_size=args.patch_size,
        vit_enc_model_size=vit_enc_model_size,
        vit_dec_model_size=vit_dec_model_size,
        token_channels=args.token_channels,
        num_classes=args.num_classes if args.cond_generator else 0,
        halve_model_size=args.halve_model_size,
        in_context_size=args.in_context_size,
        pixel_head_type=args.pixel_head_type,
        spherify_model=args.spherify_model,
        use_pixel_consistency=False,
        use_latent_consistency=False,
        noise_sigma_max_angle=args.noise_sigma_max_angle,
        mix_hard_cases=False,
        mix_hard_cases_prob=0.0,
        mix_hard_cases_max_angle=89,
        vit_enc_latent_mlp_mixer_depth=args.vit_enc_latent_mlp_mixer_depth,
        vit_dec_latent_mlp_mixer_depth=args.vit_dec_latent_mlp_mixer_depth,
        affine_latent_mlp_mixer=args.affine_latent_mlp_mixer,
    )
    model.to(device)
    model.eval()

    # load checkpoint
    ckpt_path = find_checkpoint(args.checkpoint)
    load_ckpt(model, ckpt_path, strict=args.strict_ckpt)

    # determine dataset class based on dataset_name
    if args.dataset_name == "cifar-10":
        dataset_cls = datasets.CIFAR10
    elif args.dataset_name == "cifar-100":
        dataset_cls = datasets.CIFAR100
    else:
        # load images from a list of file paths and labels
        dataset_cls = ListDataset

    # create dataset
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # determine which splits to encode
    splits_to_encode = []
    if args.split == "all":
        if dataset_cls == datasets.CIFAR10 or dataset_cls == datasets.CIFAR100:
            splits_to_encode = [("train", True), ("test", False)]
        else:
            splits_to_encode = [("data", True)]  # for non-CIFAR, just encode everything
    else:
        if dataset_cls == datasets.CIFAR10 or dataset_cls == datasets.CIFAR100:
            splits_to_encode = [(args.split, args.split == "train")]
        else:
            splits_to_encode = [("data", True)]

    # encode dataset(s)
    all_encodings = []
    all_labels = []
    all_split_ids = []
    
    for split_name, is_train in splits_to_encode:
        logger.info(f"{'='*60}")
        logger.info(f"Encoding {split_name.upper()} split...")
        logger.info(f"{'='*60}")
        
        if dataset_cls == datasets.CIFAR10 or dataset_cls == datasets.CIFAR100:
            dataset = dataset_cls(
                root=args.data_path,
                train=is_train,
                transform=transform,
                download=False,
            )
        else:
            dataset = dataset_cls(
                root=args.data_path,
                transform=transform,
                max_samples=args.max_samples,
                load_from_zip=args.load_from_zip,
            )

        logger.info(f"Dataset size: {len(dataset)} samples")

        # create dataloader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        encodings = []
        labels = []
        
        num_batches = len(loader)
        logger.info(f"Processing {num_batches} batches (batch_size={args.batch_size})")

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Encoding {split_name}", total=num_batches, unit="batch")
            for batch_idx, batch in enumerate(pbar):
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                else:
                    x = batch.to(device, non_blocking=True)
                    y = None

                z = model.encoder(x, y)  # [B, N, D]
                encodings.append(z.cpu())
                if y is not None:
                    labels.append(y.cpu())
                
                # Update progress bar with current batch info
                pbar.set_postfix({
                    'shape': f"{z.shape}",
                    'device': str(device)
                })

        encodings = torch.cat(encodings, dim=0)  # [total_samples, N, D]
        all_encodings.append(encodings)
        
        if len(labels) > 0:
            labels = torch.cat(labels, dim=0)
            all_labels.append(labels)
            # create split IDs (0 for train, 1 for test, etc.)
            split_id = 0 if is_train else 1
            split_ids = torch.full((labels.shape[0],), split_id, dtype=torch.long)
            all_split_ids.append(split_ids)
        
        logger.info(f"✓ Encoded {split_name}: encodings.shape={encodings.shape}")
        if len(labels) > 0:
            logger.info(f"  Labels: shape={labels.shape}, unique={torch.unique(labels).tolist()}")

    # concatenate all splits
    logger.info(f"\n{'='*60}")
    logger.info(f"Concatenating splits...")
    logger.info(f"{'='*60}")
    
    encodings = torch.cat(all_encodings, dim=0)  # [total_samples, N, D]
    logger.info(f"Final encodings shape: {encodings.shape}")
    logger.info(f"Memory usage: {encodings.element_size() * encodings.nelement() / 1e9:.2f} GB")
    
    # save with labels and split IDs
    output_data = {
        'encodings': encodings,
        'labels': torch.cat(all_labels, dim=0) if all_labels else None,
    }
    
    if all_split_ids:
        output_data['split_ids'] = torch.cat(all_split_ids, dim=0)
        output_data['split_names'] = ['train', 'test']  # mapping for split_ids
        
        logger.info(f"Split breakdown:")
        for split_id, split_name in enumerate(output_data['split_names']):
            count = (output_data['split_ids'] == split_id).sum().item()
            logger.info(f"  {split_name}: {count} samples")
    
    output_file = os.path.join(output_dir, args.output_name)
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving encoded dataset...")
    logger.info(f"Output file: {output_file}")
    torch.save(output_data, output_file)
    logger.info(f"✓ Successfully saved to {output_file}")
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"ENCODING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total samples: {encodings.shape[0]}")
    logger.info(f"Encoding shape: {encodings.shape}")
    logger.info(f"File size: {os.path.getsize(output_file) / 1e9:.2f} GB")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)