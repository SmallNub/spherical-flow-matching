import os
import os.path as osp
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from torchvision.utils import save_image

from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.utils import load_ckpt
import torch_fidelity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# ARGS
# -------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--encoding_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="decoded_eval")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=8)

parser.add_argument("--use_ema", type=bool, default=True)
parser.add_argument("--image_size", type=int, default=32)
parser.add_argument("--dataset_name", type=str, default="cifar-10")

cli_args = parser.parse_args()


# -------------------------------------------------
# LOAD ENCODINGS
# -------------------------------------------------
def load_data(path):
    data = torch.load(path, map_location="cpu")
    return data["encodings"], data.get("labels"), data.get("split_ids")


# -------------------------------------------------
# MODEL
# -------------------------------------------------
def build_model(args):
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
    return model


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main(cli_args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # LOAD ENCODINGS
    # ----------------------------
    z, y, split_ids = load_data(cli_args.encoding_path)

    logger.info(f"Loaded encodings: {z.shape}")

    z = z.to(device)
    if y is not None:
        y = y.to(device)

    # ----------------------------
    # LOAD CONFIG FROM CHECKPOINT DIR
    # ----------------------------
    exp_dir = osp.dirname(cli_args.checkpoint)
    cfg_path = osp.join(exp_dir, "cfg.json")

    import json
    from types import SimpleNamespace

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    args = SimpleNamespace(**cfg)
    args.image_size = cli_args.image_size
    args.dataset_name = cli_args.dataset_name

    # ----------------------------
    # BUILD MODEL
    # ----------------------------
    model = build_model(args).to(device)

    ema = SimpleEMA(model)

    load_ckpt(
        model,
        cli_args.checkpoint,
        ema_model=ema,
        strict=True,
        override_model_with_ema=cli_args.use_ema,
        verbose=True,
    )

    model.eval()

    # ----------------------------
    # DECODING
    # ----------------------------
    os.makedirs(cli_args.output_dir, exist_ok=True)

    reconstructions = []

    batch_size = cli_args.batch_size
    num_samples = z.shape[0]

    logger.info("Starting decoding...")

    for i in tqdm(range(0, num_samples, batch_size)):

        z_batch = z[i:i+batch_size]

        y_batch = y[i:i+batch_size] if y is not None else None

        with torch.no_grad():
            # IMPORTANT: assumes decoder exists
            x_rec = model.decoder(z_batch, y_batch)

        reconstructions.append(x_rec.cpu())

        # save images for inspection
        for j in range(x_rec.shape[0]):
            idx = i + j
            save_image(
                x_rec[j],
                osp.join(cli_args.output_dir, f"{idx:06d}.png"),
                normalize=True,
                value_range=(-1, 1),
            )

    reconstructions = torch.cat(reconstructions, dim=0)

    logger.info(f"Decoded shape: {reconstructions.shape}")

    # ----------------------------
    # FID EVAL (optional but aligned with eval.py)
    # ----------------------------
    logger.info("Computing FID...")

    metrics = torch_fidelity.calculate_metrics(
        input1=cli_args.output_dir,
        input2=None,  # you can plug real dataset dir here
        cuda=True,
        fid=True,
        isc=True,
        kid=False,
        verbose=True,
    )

    logger.info(metrics)


# -------------------------------------------------
if __name__ == "__main__":
    main(cli_args)
