import tqdm
import torch
from manifm.model_pl import ManifoldFMLitModule
from omegaconf import OmegaConf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_DIR = "outputs/runs/sphere_encodings/fm/2026.05.05/172728"

cfg = OmegaConf.load(f"{RUN_DIR}/.hydra/config.yaml")
ckpt_path = f"{RUN_DIR}/checkpoints/last.ckpt"

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
    z = z / z.norm(dim=[1, 2], keepdim=True)
    z = z * torch.sqrt(torch.tensor(T * D))
    return z


def get_v(t_val, x):
    t = torch.full((x.shape[0], 1), t_val, device=DEVICE)
    return model.vecfield(t, x)


@torch.no_grad()
def integrate_flow(z_start, steps=1000, start_t=0.0):
    """Generic Euler integrator for the manifold flow."""
    z_old = z_start.clone().to(DEVICE)
    dt = (1.0 - start_t) / steps

    for i in tqdm.tqdm(range(steps), desc="Integrating flow"):
        current_t = start_t + dt * i
        v = get_v(current_t, z_old)
        u = dt * v
        print("Velocity norm:", u.norm(dim=-1).mean())

        # Euler Step + Manifold Projection
        z = manifold.expmap(z_old, dt * v)
        z = manifold.projx(z)

        measure_manifold_distance(z, z_old)
        z_old = z

    measure_manifold_distance(z_old, z_start)
    return z


@torch.no_grad()
def improve_encodings(path_to_existing, noise_std=0.0, blend_factor=1.0):
    """Loads existing .pt file and pushes encodings through the flow."""
    print(f"Loading existing encodings from {path_to_existing}")
    data = torch.load(path_to_existing, map_location=DEVICE)
    z_existing = data["encodings"].float()
    labels = data.get("labels")
    split_ids = data.get("split_ids")
    split_names = data.get("split_names")

    print("Norms:", z_existing.norm(dim=1).mean(), z_existing.norm(dim=2).mean())

    z_existing = normalize(z_existing)
    z_existing = manifold.projx(z_existing)

    # If labels are None but the decoder needs them, initialize zeros
    if labels is None:
        labels = torch.zeros(z_existing.shape[0], dtype=torch.long)

    print(f"Refining {z_existing.shape[0]} samples...")
    noise = torch.randn_like(z_existing) * noise_std
    z_noise = z_existing + noise
    z_noise = manifold.projx(z_noise)
    z_improved = integrate_flow(z_noise, steps=1, start_t=0.99)  # Start close to the end of the flow

    z_final = (1 - blend_factor) * z_existing + blend_factor * z_improved
    z_final = manifold.projx(z_final)

    z_init = manifold.random_base(len(z_existing), z_existing.shape[-1]).to(DEVICE)
    z_init = manifold.projx(z_init)

    print("Manifold distance stats:")
    measure_manifold_distance(z_init, z_existing, text="Full Noise vs Original")
    measure_manifold_distance(z_noise, z_existing, text="Noise vs Original")
    measure_manifold_distance(z_noise, z_final, text="Noise vs Improved")
    measure_manifold_distance(z_final, z_existing, text="Improved vs Original")

    return z_improved, labels, split_ids, split_names


@torch.no_grad()
def measure_manifold_distance(z_noisy, z_original, text="Manifold Distance"):
    """
    Measures the geodesic distance between two batches of points.
    """
    # Ensure they are on the same device
    z_noisy = z_noisy.to(DEVICE)
    z_original = z_original.to(DEVICE)

    # model.manifold.dist(x, y) computes the geodesic distance
    # Returns a tensor of shape [batch_size]
    distances = manifold.dist(z_noisy, z_original)

    avg_dist = distances.mean().item()
    std_dist = distances.std().item()
    max_dist = distances.max().item()
    min_dist = distances.min().item()

    print(f"{text} - Avg: {avg_dist:.4f}, Std: {std_dist:.4f}, Max: {max_dist:.4f}, Min: {min_dist:.4f}")


MODE = "improve"  # Switch between "generate" or "improve"
EXISTING_PATH = "../../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/encoded_dataset.pt"

if MODE == "generate":
    n_samples = 50000
    z_init = manifold.random_base(n_samples, dim).to(DEVICE)
    z_init = manifold.projx(z_init)
    labels = torch.zeros(n_samples, dtype=torch.long)
    z_final = integrate_flow(z_init, steps=1000, start_t=0.0)
    split_ids = torch.zeros(n_samples, dtype=torch.long)
    split_names = ["generated"]
else:
    z_final, labels, split_ids, split_names = improve_encodings(EXISTING_PATH)

z_final = unnormalize(z_final.cpu())

torch.save({
    "encodings": z_final,
    "labels": labels if labels is not None else None,
    "split_ids": split_ids,
    "split_names": split_names,
}, "../../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/output_encodings.pt")

print(f"Done. Saved shape: {z_final.shape}")
