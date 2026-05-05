import torch
import torch.nn.functional as F
from manifm.model_pl import ManifoldFMLitModule
from omegaconf import OmegaConf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. LOAD MODEL
cfg = OmegaConf.load("configs/train.yaml")
ckpt_path = "outputs/runs/sphere_encodings/fm/2026.05.05/150140/checkpoints/last.ckpt"

model = ManifoldFMLitModule.load_from_checkpoint(ckpt_path, cfg=cfg).to(DEVICE)
model.eval()

manifold = model.manifold
dim = model.dim


def normalize(z):
    N, T, D = z.shape  # e.g. 256, 4

    z = z.reshape(N, T * D)   # [N, 1024]

    z = z / z.norm(dim=-1, keepdim=True)
    return z


def unnormalize(z_flat, T=256, D=4):
    N = z_flat.shape[0]

    z = z_flat.reshape(N, T, D)

    # OPTIONAL: re-normalize per token (recommended)
    z = z / z.norm(dim=-1, keepdim=True)

    return z


# 2. VECTOR FIELD WRAPPER
def get_v(x, t_val):
    t = torch.full((x.shape[0], 1), t_val, device=DEVICE)
    return model.vecfield(t, x)

# 3. CORE INTEGRATION LOGIC
@torch.no_grad()
def integrate_flow(z_start, steps=100):
    """Generic Euler integrator for the manifold flow."""
    z = z_start.clone().to(DEVICE)
    dt = 1.0 / steps

    for i in range(steps):
        current_t = i / steps
        v = get_v(z, current_t)

        # Euler Step + Manifold Projection
        z = z + dt * v
        z = manifold.projx(z)

    return z

# 4. IMPROVE EXISTING ENCODINGS
@torch.no_grad()
def improve_encodings(path_to_existing, steps=50):
    """Loads existing .pt file and pushes encodings through the flow."""
    print(f"Loading existing encodings from {path_to_existing}")
    data = torch.load(path_to_existing, map_location=DEVICE)
    z_existing = data["encodings"].float()
    labels = data.get("labels")
    split_ids = data.get("split_ids")
    split_names = data.get("split_names")

    z_existing = normalize(z_existing)

    # If labels are None but the decoder needs them, initialize zeros
    if labels is None:
        labels = torch.zeros(z_existing.shape[0], dtype=torch.long)

    print(f"Refining {z_existing.shape[0]} samples...")
    z_improved = integrate_flow(z_existing, steps=steps)

    return z_improved, labels, split_ids, split_names


# 5. EXECUTION TOGGLE
MODE = "improve"  # Switch between "generate" or "improve"
EXISTING_PATH = "../../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/encoded_dataset.pt"

if MODE == "generate":
    n_samples = 50000
    z_init = manifold.random_base(n_samples, dim).to(DEVICE)
    labels = torch.zeros(n_samples, dtype=torch.long)
    z_final = integrate_flow(z_init, steps=100)
    split_ids = torch.zeros(n_samples, dtype=torch.long)
    split_names = ["generated"]
else:
    z_final, labels, split_ids, split_names = improve_encodings(EXISTING_PATH, steps=50)

z_final = unnormalize(z_final.cpu())

# 6. SAVE
torch.save({
    "encodings": z_final,
    "labels": labels if labels is not None else None,
    "split_ids": split_ids,
    "split_names": split_names,
}, "../../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/output_encodings.pt")

print(f"Done. Saved shape: {z_final.shape}")
