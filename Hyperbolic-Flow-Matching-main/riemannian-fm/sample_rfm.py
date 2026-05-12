import tqdm
import torch
from manifm.model_pl import ManifoldFMLitModule
from omegaconf import OmegaConf
from preprocess_data import manifold_squeeze

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECK_UNIFORMITY = False
RUNTIME_STATS = False

N_SAMPLES = 10000
NUM_CLASSES = 10
STEPS = 100
NOISE_STD = 1.0
START_T = 0.0
SQUEEZE = False
SQUEEZE_ALPHA = 0.5

# TODO: Fully finish CFG, change to csv

GENERATION = False
INPUT_PATH = "../../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/encoded_dataset.pt"

RUN_DIR = "outputs/runs/sphere_encodings/fm/2026.05.12/150428"

cfg = OmegaConf.load(f"{RUN_DIR}/.hydra/config.yaml")
ckpt_path = f"{RUN_DIR}/checkpoints/last.ckpt"

model = ManifoldFMLitModule.load_from_checkpoint(ckpt_path, cfg=cfg).to(DEVICE)
model.eval()
model.compile()

manifold = model.manifold
dim = model.dim


def dummy_labels(n_samples=50000):
    labels = []
    for i in range(NUM_CLASSES):
        class_i = torch.full((n_samples // NUM_CLASSES,), i, dtype=torch.long)
        labels.append(class_i)
    return torch.cat(labels, dim=0)


def normalize(z):
    N, T, D = z.shape
    z = z.reshape(N, T * D)
    z = z / z.norm(dim=-1, keepdim=True)
    return z


def unnormalize(z_flat, T=256, D=4):
    N = z_flat.shape[0]
    z = z_flat.reshape(N, T, D)
    z = z / z.norm(dim=[1, 2], keepdim=True)
    z = z * torch.sqrt(torch.tensor(T * D))
    return z


def get_v(t_val, x, y=None):
    t = torch.full((x.shape[0], 1), t_val, device=DEVICE)
    return model.vecfield(t, x, y=y)


@torch.no_grad()
def integrate_flow(z_start, labels=None, steps=1000, start_t=0.0):
    """Generic Euler integrator for the manifold flow."""
    if labels is not None:
        labels = labels.to(DEVICE)

    z_prev = z_start.clone().to(DEVICE)
    dt = (1.0 - start_t) / steps

    for i in tqdm.tqdm(range(steps), desc="Integrating flow"):
        current_t = start_t + dt * i
        v = get_v(current_t, z_prev, y=labels)
        u = dt * v

        # Euler Step + Manifold Projection
        z = manifold.expmap(z_prev, u)
        z = manifold.projx(z)

        if RUNTIME_STATS:
            print("Velocity norm:", u.norm(dim=-1).mean())
            measure_manifold_distance(z, z_prev, "Step Distance")
        z_prev = z

    measure_manifold_distance(z_prev, z_start, "Start vs End")
    return z_prev


@torch.no_grad()
def improve_encodings(
    input_path,
    noise_std=NOISE_STD,
    start_t=START_T,
    generation=False,
    generation_samples=N_SAMPLES,
):
    """Loads existing .pt file and pushes encodings through the flow."""
    print(f"Loading existing encodings from {input_path}")
    data = torch.load(input_path, map_location=DEVICE)
    z_input = data["encodings"].float()
    labels_input = data.get("labels")
    split_ids = data.get("split_ids")
    split_names = data.get("split_names")
    class_means = data.get("class_means", None)

    # labels = torch.full((z_noise.shape[0],), 1, dtype=torch.long).to(DEVICE)

    if CHECK_UNIFORMITY:
        for label in labels_input.unique():
            mask = labels_input == label
            z_label = z_input[mask]
            check_hypersphere_uniformity(z_label, text=f"Label {label}")
            z_label = normalize(z_label)
            z_label = manifold.projx(z_label)
            check_uniformity(z_label)

        check_hypersphere_uniformity(z_input)

    z_input = normalize(z_input)
    z_input = manifold.projx(z_input)

    if SQUEEZE:
        z_input, _ = manifold_squeeze(z_input, labels_input, class_means=class_means, alpha=SQUEEZE_ALPHA, reverse=False)

    if CHECK_UNIFORMITY:
        check_uniformity(z_input)

    if generation:
        labels = dummy_labels(generation_samples)
    else:
        labels = labels_input

    z_init = manifold.random_base(generation_samples if generation else len(z_input), z_input.shape[-1]).to(DEVICE)
    z_init = manifold.projx(z_init)

    if generation:
        z_noise = z_init
    else:
        noise = torch.randn_like(z_input) * noise_std
        z_noise = z_input + noise
        z_noise = manifold.projx(z_noise)

    print(f"Refining {z_noise.shape[0]} samples...")
    z_final = integrate_flow(z_noise, labels, steps=STEPS, start_t=start_t)

    print("Clustering report:")
    check_class_clustering(z_input, labels_input, text="Original Class Clustering")
    check_class_clustering(z_noise, labels, text="Noise Class Clustering")
    check_class_clustering(z_final, labels, text="Final Class Clustering")

    if not generation:
        print("Manifold distance stats:")
        measure_manifold_distance(z_init, z_input, text="Full Noise vs Original")
        measure_manifold_distance(z_noise, z_input, text="Noise vs Original")
        measure_manifold_distance(z_noise, z_final, text="Noise vs Improved")
        measure_manifold_distance(z_final, z_input, text="Improved vs Original")

    if SQUEEZE:
        z_final, _ = manifold_squeeze(z_final, labels, class_means=class_means, alpha=SQUEEZE_ALPHA, reverse=True)

    return z_final, labels, split_ids, split_names


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
    target_cos = (1 / total_dim)**0.5  # Expected cos sim for random 1024-D vectors
    print(f"Theoretical Random CosSim: ~{target_cos:.4f}")


@torch.no_grad()
def check_class_clustering(z, labels, text="Class Clustering"):
    """
    Calculates the mean location for each class and measures how
    tightly the samples are clustered around that mean.
    """
    z = z.to(DEVICE)
    labels = labels.to(DEVICE)
    unique_labels = torch.unique(labels)

    print(f"\n=== {text} Report ===")
    print(f"{'Label':<10} | {'Mean Norm':<10} | {'Avg Dist':<10} | {'Std Dist':<10} | {'Max Dist':<10}")
    print("-" * 65)

    all_stats = []

    for label in unique_labels:
        mask = (labels == label)
        z_class = z[mask]

        if z_class.shape[0] == 0:
            continue

        # 1. Calculate the Centroid (Mean) on the sphere
        # We average in Euclidean space then project to the sphere surface
        mean_vec = z_class.mean(dim=0, keepdim=True)
        centroid = manifold.projx(mean_vec)

        # Mean Vector Norm (tells us how 'un-uniform' the class is)
        mean_norm = mean_vec.norm().item()

        # 2. Geodesic Distances from every point in class to its centroid
        # Broadcast centroid to match z_class shape
        centroid_batch = centroid.expand(z_class.shape[0], -1)
        distances = manifold.dist(z_class, centroid_batch)

        avg_d = distances.mean().item()
        std_d = distances.std().item()
        max_d = distances.max().item()

        print(f"{label.item():<10} | {mean_norm:<10.4f} | {avg_d:<10.4f} | {std_d:<10.4f} | {max_d:<10.4f}")

        all_stats.append({
            'label': label.item(),
            'mean_norm': mean_norm,
            'avg_dist': avg_d
        })

    # Global summary
    avg_class_tightness = sum(s['avg_dist'] for s in all_stats) / len(all_stats)
    print("-" * 65)
    print(f"Global Average Class Tightness (Distance to Mean): {avg_class_tightness:.4f}")
    return all_stats


if GENERATION:
    z_final, labels, *_ = improve_encodings(INPUT_PATH, generation=True, generation_samples=N_SAMPLES)
    split_ids = torch.zeros(N_SAMPLES, dtype=torch.long)
    split_names = ["generated"]
else:
    z_final, labels, split_ids, split_names = improve_encodings(INPUT_PATH)

z_final = unnormalize(z_final.cpu())

# torch.save({
#     "encodings": z_final,
#     "labels": labels.cpu(),
#     "split_ids": split_ids.cpu(),
#     "split_names": split_names,
# }, "../../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/output_encodings.pt")

print(f"Done. Saved shape: {z_final.shape}")
