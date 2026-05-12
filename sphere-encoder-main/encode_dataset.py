import os
import os.path as osp
import json
import argparse
import logging
import random
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from torchvision import datasets, transforms

from cli_utils import str2bool
from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.loader import ListDataset
from sphere.utils import load_ckpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# CLI
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Encode dataset")

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--output_name", type=str, default="encoded_dataset.npz")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=8)

parser.add_argument("--dataset_name", type=str, default="cifar-10")
parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])

parser.add_argument("--max_samples", type=int, default=-1)
parser.add_argument("--load_from_zip", type=str2bool, default=False)

# reproducibility
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--deterministic", type=str2bool, default=False)

# precision
parser.add_argument("--save_dtype", type=str, default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])

# model behavior
parser.add_argument("--use_ema", type=str2bool, default=True)
parser.add_argument("--compile_model", type=str2bool, default=True)

cli_args = parser.parse_args()


# -------------------------------------------------
# SEEDING
# -------------------------------------------------
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# CHECKPOINT HELPER
# -------------------------------------------------
def find_checkpoint(path):
    if os.path.isfile(path):
        return path
    elif os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith(".pth")])
        if not files:
            raise FileNotFoundError(f"No .pth files in {path}")
        return os.path.join(path, files[-1])
    else:
        raise FileNotFoundError(path)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main(cli_args):
    # -------------------------------------------------
    # LOAD CONFIG
    # -------------------------------------------------
    exp_dir = osp.dirname(cli_args.checkpoint)
    cfg_path = osp.join(exp_dir, "cfg.json")

    logger.info(f"Loading config from {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg_args = json.load(f)

    cfg_args.update(vars(cli_args))
    args = SimpleNamespace(**cfg_args)

    # -------------------------------------------------
    # SEED
    # -------------------------------------------------
    set_seed(args.seed, args.deterministic)

    # -------------------------------------------------
    # DEVICE / DTYPE
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(getattr(args, "dtype", "bfloat16"), torch.bfloat16)

    target_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.save_dtype]

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    output_dir = args.output_path or args.data_path
    os.makedirs(output_dir, exist_ok=True)
    output_file = osp.join(output_dir, args.output_name)

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    model = G(
        input_size=args.image_size,
        patch_size=args.patch_size,
        vit_enc_model_size=args.vit_enc_model_size,
        vit_dec_model_size=args.vit_dec_model_size,
        token_channels=args.token_channels,
        num_classes=args.num_classes if args.cond_generator else 0,
        halve_model_size=args.halve_model_size,
        spherify_model=args.spherify_model,
        pixel_head_type=args.pixel_head_type,
        in_context_size=args.in_context_size,
        noise_sigma_max_angle=args.noise_sigma_max_angle,
        vit_enc_latent_mlp_mixer_depth=args.vit_enc_latent_mlp_mixer_depth,
        vit_dec_latent_mlp_mixer_depth=args.vit_dec_latent_mlp_mixer_depth,
        affine_latent_mlp_mixer=args.affine_latent_mlp_mixer,
    )

    model.to(device=device, dtype=ptdtype, memory_format=torch.channels_last)

    ema_model = SimpleEMA(model)

    ckpt_path = find_checkpoint(args.checkpoint)
    logger.info(f"Loading checkpoint: {ckpt_path}")

    load_ckpt(
        model,
        ckpt_path,
        ema_model=ema_model,
        strict=True,
        override_model_with_ema=args.use_ema,
        verbose=True,
    )

    if args.compile_model:
        model.compile()

    model.eval().requires_grad_(False)

    # -------------------------------------------------
    # DATASET SETUP
    # -------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    if args.dataset_name in ["cifar-10", "cifar-100"]:
        dataset_cls = datasets.__dict__[args.dataset_name.upper().replace("-", "")]
    else:
        dataset_cls = ListDataset

    splits = []
    if args.split == "all":
        if args.dataset_name in ["cifar-10", "cifar-100"]:
            splits = [("train", True), ("test", False)]
        else:
            splits = [("data", True)]
    else:
        if args.dataset_name in ["cifar-10", "cifar-100"]:
            splits = [(args.split, args.split == "train")]
        else:
            splits = [("data", True)]

    # -------------------------------------------------
    # ENCODING
    # -------------------------------------------------
    all_encodings = []
    all_labels = []
    all_split_ids = []

    for split_id, (split_name, is_train) in enumerate(splits):
        logger.info(f"Encoding split: {split_name}")

        if dataset_cls != ListDataset:
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

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        for batch in tqdm(loader, desc=f"{split_name}"):
            if isinstance(batch, (list, tuple)):
                x, y = batch
                y = y.to(device, non_blocking=True)
            else:
                x, y = batch, None

            x = x.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=ptdtype):
                z = model.encoder(x, y)
                z = model.spherify(z, sampling=False)

            z = z.cpu()

            z = z.to(target_dtype)

            all_encodings.append(z)

            if y is not None:
                y_cpu = y.cpu()
                all_labels.append(y_cpu)
                all_split_ids.append(
                    torch.full((y_cpu.shape[0],), split_id, dtype=torch.long)
                )

            torch.cuda.empty_cache()

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------
    encodings = torch.cat(all_encodings, dim=0)

    output = {
        "encodings": encodings,
        "labels": torch.cat(all_labels, dim=0) if all_labels else None,
        "split_ids": torch.cat(all_split_ids, dim=0) if all_split_ids else None,
        "split_names": [s[0] for s in splits],
    }

    np.savez_compressed(
        output_file,
        allow_pickle=False,
        encodings=output["encodings"].cpu().float().numpy(),
        labels=output["labels"].cpu().numpy(),
        split_ids=output["split_ids"].cpu().numpy(),
        split_names=np.array(output["split_names"], dtype=str),
    )

    logger.info(f"Saved to {output_file}")
    logger.info(f"Encodings shape: {encodings.shape}")


if __name__ == "__main__":
    main(cli_args)
