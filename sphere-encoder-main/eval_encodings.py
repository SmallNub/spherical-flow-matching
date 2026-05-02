import os
import os.path as osp
import json
import argparse
import logging
import shutil
import random
from types import SimpleNamespace
from cli_utils import str2bool

import numpy as np
import torch
import torch.nn.functional as F
import torch_fidelity
from tqdm import tqdm

from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.utils import load_ckpt, save_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# CLI
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Decode encodings + eval (split-aware)")

parser.add_argument("--encoding_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)

parser.add_argument("--output_dir", type=str, default="decoded_eval")
parser.add_argument("--dataset_name", type=str, default="cifar-10")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--use_ema", type=bool, default=True)
parser.add_argument("--compile_model", type=str2bool, default=True)

parser.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])

parser.add_argument("--normalize_latents", type=bool, default=True)
parser.add_argument("--save_images", type=bool, default=True)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--deterministic", type=str2bool, default=False)

cli_args = parser.parse_args()


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# LOAD ENCODINGS
# -------------------------------------------------
def load_data(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return (
        data["encodings"],
        data.get("labels"),
        data.get("split_ids"),
        data.get("split_names", None),
    )


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
# SPLIT UTILS
# -------------------------------------------------
def get_split_indices(split_ids, target_id):
    return (split_ids == target_id).nonzero(as_tuple=True)[0]


# -------------------------------------------------
# DECODE LOOP
# -------------------------------------------------
def decode_from_latents(model, z, y=None):
    x = model.decoder(z, y)
    x = torch.clamp(x * 0.5 + 0.5, 0, 1)
    return x


@torch.inference_mode()
def decode_and_save(model, z, y, save_dir, args, ptdtype, device):

    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    num_samples = z.shape[0]
    num_batches = int(np.ceil(num_samples / args.batch_size))

    logger.info(f"Decoding {num_samples} samples → {save_dir}")

    cnt = 0
    pbar = tqdm(range(num_batches))

    for batch_idx in pbar:

        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, num_samples)

        z_batch = z[start:end].to(device)

        y_batch = y[start:end].to(device) if y is not None else None

        # IMPORTANT: normalize latents (sphere models NEED this)
        if args.normalize_latents:
            z_batch = F.normalize(z_batch, dim=-1)

        # match model dtype
        z_batch = z_batch.to(ptdtype)

        with torch.autocast(device_type="cuda", dtype=ptdtype):
            x_rec = decode_from_latents(model, z_batch, y_batch)

        cnt += x_rec.shape[0]
        pbar.set_description(f"decoded {cnt}/{num_samples}")

        if args.save_images:
            save_image(
                x=x_rec,
                batch_idx=batch_idx,
                ddp_rank=0,
                save_dir=save_dir,
                force_image_size=args.image_size,
            )

        torch.cuda.empty_cache()

    logger.info("Decoding finished")


# -------------------------------------------------
# METRICS
# -------------------------------------------------
def run_metrics(img_dir, args, split_name):

    logger.info(f"Running metrics for split: {split_name}")

    input2 = None

    # match eval.py behavior
    if args.dataset_name == "cifar-10":
        input2 = "cifar10-train"

    metrics = torch_fidelity.calculate_metrics(
        input1=img_dir,
        input2=input2,
        cuda=torch.cuda.is_available(),
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )

    logger.info(f"[{split_name}] {metrics}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
@torch.inference_mode()
def main(cli_args):

    # -------------------------------------------------
    # LOAD CONFIG (same as eval/encode)
    # -------------------------------------------------
    exp_dir = osp.dirname(cli_args.checkpoint)
    cfg_path = osp.join(exp_dir, "cfg.json")

    logger.info(f"Loading config from {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg_args = json.load(f)

    cfg_args.update(vars(cli_args))
    args = SimpleNamespace(**cfg_args)

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
    }[args.dtype]

    # -------------------------------------------------
    # LOAD ENCODINGS
    # -------------------------------------------------
    z, y, split_ids, split_names = load_data(args.encoding_path)

    logger.info(f"Encodings: {z.shape}, dtype={z.dtype}")

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
    # SPLIT HANDLING
    # -------------------------------------------------
    splits_to_eval = []

    if split_ids is not None and split_names is not None:
        for i, name in enumerate(split_names):
            splits_to_eval.append((i, name))
    else:
        splits_to_eval.append((None, "all"))

    # -------------------------------------------------
    # RUN PER SPLIT
    # -------------------------------------------------
    for split_id, split_name in splits_to_eval:

        logger.info(f"\n==== Evaluating split: {split_name} ====")

        if split_id is not None:
            idx = get_split_indices(split_ids, split_id)
            z_split = z[idx]
            y_split = y[idx] if y is not None else None
        else:
            z_split, y_split = z, y

        split_dir = osp.join(args.output_dir, f"decoded_{split_name}")

        decode_and_save(
            model,
            z_split,
            y_split,
            split_dir,
            args,
            ptdtype,
            device,
        )

        run_metrics(split_dir, args, split_name)


# -------------------------------------------------
if __name__ == "__main__":
    main(cli_args)
