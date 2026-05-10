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

    for label in labels.unique():
        mask = labels == label
        z_label = z_existing[mask]
        check_hypersphere_uniformity(z_label, text=f"Label {label}")
        z_label = normalize(z_label)
        z_label = manifold.projx(z_label)
        check_uniformity(z_label)

    check_hypersphere_uniformity(z_existing)
    z_existing = normalize(z_existing)
    z_existing = manifold.projx(z_existing)
    check_uniformity(z_existing)

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


@torch.no_grad()
def check_uniformity(z):
    z = z.float()
    N, D = z.shape

    # 1. Mean Norm (Should be near 0)
    mean_vec_norm = z.mean(dim=0).norm().item()

    # 2. SVD (Spread of singular values)
    _, S, _ = torch.svd(z)
    S_normalized = S / S.max()
    entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-8)).item()

    # 3. Pairwise Orthogonality
    # Sample 1000 random pairs to save memory
    idx1 = torch.randint(0, N, (1000,))
    idx2 = torch.randint(0, N, (1000,))
    cos_sim = torch.nn.functional.cosine_similarity(z[idx1], z[idx2])
    avg_cos = cos_sim.abs().mean().item()

    print(f"--- Uniformity Report ---")
    print(f"Mean Vector Norm: {mean_vec_norm:.4f} (Target: 0.0)")
    print(f"SVD Entropy:      {entropy:.4f} (Higher = More Uniform)")
    print(f"Avg Abs CosSim:   {avg_cos:.4f} (Target for high-D uniform: ~0.0)")


@torch.no_grad()
def check_hypersphere_uniformity(z, text="Raw Encodings"):
    """
    Checks uniformity for [N, 256, 4] encodings.
    Calculates statistics for a hypersphere in 1024-D space.
    """
    z = z.float().to(DEVICE)
    N, T, D = z.shape
    total_dim = T * D  # 1024

    # Flatten to [N, 1024] for high-D manifold analysis
    z_flat = z.reshape(N, total_dim)

    # 1. Check Radius (RMS Norm)
    # The paper says RMS Norm = sqrt(total_dim) / total_dim? 
    # Usually, RMSNorm(x) = sqrt(mean(x^2)). 
    # If they define the sphere as ||z|| = sqrt(D), let's check the norm:
    norms = torch.norm(z_flat, p=2, dim=-1)
    avg_norm = norms.mean().item()
    expected_norm = (total_dim)**0.5  # sqrt(1024) = 32

    # 2. Mean Vector (Center of mass)
    mean_vec_norm = z_flat.mean(dim=0).norm().item()

    # 3. SVD / Eigenvalue Distribution (Spectral Uniformity)
    # This measures if any specific dimensions are 'preferred'
    _, S, _ = torch.svd(z_flat)
    S_norm = S / S.max()
    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-8)).item()

    # 4. Pairwise Orthogonality
    # In high-D hyperspheres, uniform points are almost always orthogonal
    idx1 = torch.randint(0, N, (1000,))
    idx2 = torch.randint(0, N, (1000,))
    cos_sim = torch.nn.functional.cosine_similarity(z_flat[idx1], z_flat[idx2])
    avg_abs_cos = cos_sim.abs().mean().item()

    print(f"--- {text} Report ---")
    print(f"Shape:             {N} samples x {total_dim} dims")
    print(f"Avg L2 Norm:       {avg_norm:.4f} (Paper Expected: {expected_norm:.4f})")
    print(f"Mean Vector Norm:  {mean_vec_norm:.4f} (Closer to 0 is more uniform)")
    print(f"SVD Entropy:       {entropy:.4f}")
    print(f"Avg Abs CosSim:    {avg_abs_cos:.4f} (Closer to 0 is more uniform)")

    # Interpretation
    target_cos = (1 / total_dim)**0.5 # Expected cos sim for random 1024-D vectors
    print(f"Theoretical Random CosSim: ~{target_cos:.4f}")


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

# torch.save({
#     "encodings": z_final,
#     "labels": labels if labels is not None else None,
#     "split_ids": split_ids,
#     "split_names": split_names,
# }, "../../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/output_encodings.pt")

print(f"Done. Saved shape: {z_final.shape}")
