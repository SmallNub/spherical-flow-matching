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
import torch_fidelity
from tqdm import tqdm
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from sphere.model import G
from sphere.ema import SimpleEMA
from sphere.utils import load_ckpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Memory-Efficient Custom Datasets
# -------------------------------------------------
class ClassSubsetDataset(Dataset):
    """Wraps a standard dataset and filters it to a single target class label."""
    def __init__(self, base_dataset, target_class):
        self.samples = []
        for img, label in base_dataset:
            if label == target_class:
                # Convert PIL Image to uint8 [0, 255] CHW tensor for torch_fidelity
                tensor_img = TF.pil_to_tensor(img)
                self.samples.append(tensor_img)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for class {target_class} in the base dataset.")

        logger.info(f"Created real reference subset for class {target_class} containing {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class InMemoryImageDataset(Dataset):
    """Holds decoded images completely in RAM as torch.Tensors for torch_fidelity."""
    def __init__(self, images=None):
        self.images = images if images is not None else []

    def append(self, img_tensor):
        self.images.append(img_tensor)

    def extend(self, other_dataset):
        self.images.extend(other_dataset.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


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
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
parser.add_argument("--normalize_latents", type=bool, default=True)
parser.add_argument("--image_size", type=int, default=32, help="Image resolution dimension.")

# TOGGLE CONTROL: Turn on/off per-class evaluation metrics
parser.add_argument("--eval_per_class", type=str2bool, default=True,
                    help="If true, computes evaluation metrics for each individual class label separately.")

parser.add_argument("--save_images", type=str2bool, default=False,
                    help="If true, saves images to disk in split/class/ directories. Otherwise, runs purely in-memory.")

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


def load_data(path):
    data = np.load(path, allow_pickle=False)
    z_input = torch.from_numpy(data["encodings"]).float()
    labels = torch.from_numpy(data["labels"]).long()
    split_ids = torch.from_numpy(data["split_ids"]).long()
    split_names = data["split_names"].tolist()
    return z_input, labels, split_ids, split_names


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


def get_split_indices(split_ids, target_id):
    return (split_ids == target_id).nonzero(as_tuple=True)[0]


def decode_from_latents(model, z, y=None):
    x = model.decoder(z, y)
    return torch.clamp(x * 0.5 + 0.5, 0, 1)


@torch.inference_mode()
def decode_and_collect(model, z, y, save_dir, args, ptdtype, device):
    """Decodes latents and returns an InMemoryImageDataset or writes to disk if configured."""
    if args.save_images:
        os.makedirs(save_dir, exist_ok=True)

    memory_ds = InMemoryImageDataset()
    num_samples = z.shape[0]
    num_batches = int(np.ceil(num_samples / args.batch_size))

    logger.info(f"Decoding {num_samples} samples...")
    pbar = tqdm(range(num_batches))

    image_size = getattr(args, "image_size", 32)

    for batch_idx in pbar:
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, num_samples)

        z_batch = z[start:end].to(device)
        y_batch = y[start:end].to(device) if y is not None else None

        if args.normalize_latents:
            z_batch = model.spherify(z_batch)

        with torch.autocast(device_type="cuda", dtype=ptdtype):
            x_rec = decode_from_latents(model, z_batch, y_batch)

        x_rec = (x_rec * 255.0).clamp(0, 255).to(torch.uint8).cpu()
        y_cpu = y_batch.cpu().numpy() if y_batch is not None else [None] * len(x_rec)

        for i, (img_tensor, label) in enumerate(zip(x_rec, y_cpu)):
            if args.save_images:
                from sphere.loader import resize_arr
                img_np = img_tensor.permute(1, 2, 0).numpy()
                image_name = f"label={label}_ord={batch_idx:05d}_idx={i:05d}.png"
                image_path = os.path.join(save_dir, image_name)
                image = Image.fromarray(img_np)
                if image_size > 0:
                    image = resize_arr(image, image_size=image_size)
                image.save(image_path, format="PNG", compress_level=0)

            memory_ds.append(img_tensor)

        torch.cuda.empty_cache()

    return memory_ds


def run_metrics(input1_src, args, eval_name, reference_input2, cache_input2_name=None):
    logger.info(f"Running metrics for: {eval_name}")
    
    kwargs = {
        "input1": input1_src,
        "input2": reference_input2,
        "cuda": torch.cuda.is_available(),
        "batch_size": args.batch_size,
        "isc": True,   # Inception Score
        "fid": True,   # Frechet Inception Distance
        "kid": False,  # Kernel Inception Distance, very slow
        "prc": False,  # Precision and Recall, slow
        "ppl": False,  # Perceptual Path Length, requires generator
        "verbose": True,
    }

    if isinstance(input1_src, str):
        kwargs["samples_find_deep"] = True

    if cache_input2_name is not None:
        kwargs["cache_input2_name"] = cache_input2_name

    metrics = torch_fidelity.calculate_metrics(**kwargs)
    logger.info(f"[{eval_name}] {metrics}")
    return metrics


@torch.inference_mode()
def main(cli_args):
    exp_dir = osp.dirname(cli_args.checkpoint)
    cfg_path = osp.join(exp_dir, "cfg.json")

    with open(cfg_path, "r") as f:
        cfg_args = json.load(f)

    cfg_args.update(vars(cli_args))
    args = SimpleNamespace(**cfg_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, args.deterministic)

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    z, y, split_ids, split_names = load_data(args.encoding_path)

    real_cifar10_train = None
    if args.dataset_name == "cifar-10":
        logger.info("Loading real CIFAR-10 train set for conditional reference matching...")
        real_cifar10_train = torchvision.datasets.CIFAR10(root="./workspace/datasets/cifar-10", train=True, download=True)

    model = G(
        input_size=args.image_size, patch_size=args.patch_size,
        vit_enc_model_size=args.vit_enc_model_size, vit_dec_model_size=args.vit_dec_model_size,
        token_channels=args.token_channels, num_classes=args.num_classes if args.cond_generator else 0,
        halve_model_size=args.halve_model_size, spherify_model=args.spherify_model,
        pixel_head_type=args.pixel_head_type, in_context_size=args.in_context_size,
        noise_sigma_max_angle=args.noise_sigma_max_angle,
        vit_enc_latent_mlp_mixer_depth=args.vit_enc_latent_mlp_mixer_depth,
        vit_dec_latent_mlp_mixer_depth=args.vit_dec_latent_mlp_mixer_depth,
        affine_latent_mlp_mixer=args.affine_latent_mlp_mixer,
    ).to(device=device, memory_format=torch.channels_last)

    ema_model = SimpleEMA(model)
    load_ckpt(model, find_checkpoint(args.checkpoint), ema_model=ema_model, strict=True, override_model_with_ema=args.use_ema)

    if args.compile_model:
        model.compile()
    model.eval().requires_grad_(False)

    splits_to_eval = [(i, name) for i, name in enumerate(split_names)] if split_ids is not None else [(None, "all")]
    metrics = {}

    for split_id, split_name in splits_to_eval:
        logger.info(f"\n==== Evaluating split: {split_name} ====")

        if split_id is not None:
            idx = get_split_indices(split_ids, split_id)
            z_split, y_split = z[idx], y[idx] if y is not None else None
        else:
            z_split, y_split = z, y

        split_dir = osp.join(args.output_dir, f"decoded_{split_name}")
        if args.save_images and osp.exists(split_dir):
            shutil.rmtree(split_dir)

        # Initialize the container for the entire split
        overall_split_ds = InMemoryImageDataset()

        # 1. Process Per-Class Evaluation (Only runs if eval_per_class is True)
        if args.eval_per_class and y_split is not None:
            unique_classes = torch.unique(y_split)
            for cls_id in unique_classes:
                cls_id = cls_id.item()
                cls_idx = (y_split == cls_id).nonzero(as_tuple=True)[0]

                z_cls, y_cls = z_split[cls_idx], y_split[cls_idx]
                cls_dir = osp.join(split_dir, f"class_{cls_id}")
                cls_ds = decode_and_collect(model, z_cls, y_cls, cls_dir, args, ptdtype, device)
                overall_split_ds.extend(cls_ds)

                cache_name = None
                if real_cifar10_train is not None:
                    input2_ref = ClassSubsetDataset(real_cifar10_train, target_class=cls_id)
                    cache_name = f"{args.dataset_name}_train_class_{cls_id}"
                else:
                    input2_ref = None

                cls_metric_key = f"{split_name}_class_{cls_id}"
                metric_input = cls_dir if args.save_images else cls_ds

                metrics[cls_metric_key] = run_metrics(
                    metric_input, args, cls_metric_key, input2_ref, cache_input2_name=cache_name
                )
        else:
            # If tracking per-class metrics is toggled off, decode the split altogether in one single batch pass
            overall_split_ds = decode_and_collect(model, z_split, y_split, split_dir, args, ptdtype, device)

        # 2. Evaluate Overall Split
        overall_input2 = "cifar10-train" if args.dataset_name == "cifar-10" else None

        metric_input_overall = split_dir if args.save_images else overall_split_ds
        metrics[split_name] = run_metrics(metric_input_overall, args, split_name, overall_input2)

    logger.info("===== Evaluation Metrics =====")
    for eval_key, split_metrics in metrics.items():
        logger.info("---- %s ----", eval_key)
        for k, v in split_metrics.items():
            logger.info("%-20s : %s", k, v)


if __name__ == "__main__":
    main(cli_args)
