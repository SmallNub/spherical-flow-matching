import tqdm
import torch
import numpy as np
from omegaconf import OmegaConf
from manifm.model_pl import ManifoldFMLitModule
from preprocess_data import manifold_squeeze
from configs.config import PROC_DATA_PATH, OUTPUT_DATA_PATH, SPHERE_DIMS, SQUEEZE_DATA, SQUEEZE_ALPHA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECK_UNIFORMITY = False
RUNTIME_STATS = False

N_SAMPLES = 10000
NUM_CLASSES = 10
STEPS = 100
NOISE_STD = 0.0
START_T = 0.0

GUIDANCE_SCALE = 1.5
# SDE_NOISE = 0.01

SAVE_OUTPUT = True
GENERATION = True
INPUT_PATH = PROC_DATA_PATH
OUTPUT_PATH = OUTPUT_DATA_PATH

RUN_DIR = "outputs/runs/sphere_encodings/fm/2026.05.17/202651"

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
    try:
        N, T, D = z.shape
    except ValueError as e:
        print("[ERROR]", e)
        return z
    N, T, D = z.shape
    assert (T, D) == tuple(SPHERE_DIMS), f"Expected shape [N, {SPHERE_DIMS[0]}, {SPHERE_DIMS[1]}], got {z.shape}"
    z = z.reshape(N, T * D)
    z = z / z.norm(dim=-1, keepdim=True)
    return z


def unnormalize(z_flat):
    N = z_flat.shape[0]
    T, D = SPHERE_DIMS
    z = z_flat.reshape(N, T, D)
    z = z / z.norm(dim=[1, 2], keepdim=True)
    z = z * torch.sqrt(torch.tensor(T * D))
    return z


def get_v(t_val, x, y=None):
    t = torch.full((x.shape[0], 1), t_val, device=DEVICE)
    return model.vecfield(t, x, y=y)


@torch.inference_mode()
def integrate_flow(z_start, labels, steps=100, start_t=0.0, guidance_scale=GUIDANCE_SCALE, null_label=-1):
    """Generic Euler integrator for the manifold flow."""
    z_start = z_start.to(DEVICE)
    labels = labels.to(DEVICE)
    null_labels = torch.full_like(labels, null_label)

    z_prev = z_start.clone().to(DEVICE)
    dt = (1.0 - start_t) / steps

    for i in tqdm.tqdm(range(steps), desc="Integrating flow"):
        current_t = start_t + dt * i
        # High guidance at the start (t=0) to force points into class channels,
        # decaying smoothly to standard guidance as it nears the data destination.
        # This prevents the model from wandering blindly during the t=0 to 0.5 phase.
        init_cfg = 5.0      # Aggressive steering at the start
        target_cfg = 1.1    # Gentle refinement at the end

        # Decays smoothly from init_cfg to target_cfg as current_t goes 0 -> 1
        current_cfg = target_cfg + (init_cfg - target_cfg) * (1.0 - current_t)**2

        if guidance_scale != 1.0 and current_cfg > 1.0:
            z_in = torch.cat([z_prev, z_prev], dim=0).to(DEVICE)
            y_in = torch.cat([null_labels, labels], dim=0).to(DEVICE)

            v_all = get_v(current_t, z_in, y=y_in)
            v_uncond, v_cond = v_all.chunk(2, dim=0)

            v = v_uncond + current_cfg * (v_cond - v_uncond)
        else:
            v = get_v(current_t, z_prev, y=labels)

        u = dt * v
        z_next = manifold.expmap(z_prev, u)

        # if i < (steps - 1):
        #     noise = torch.randn_like(z_next) * SDE_NOISE * torch.sqrt(torch.tensor(dt))
        #     noise = manifold.proju(z_next, noise)
        #     z_next = manifold.expmap(z_next, noise)

        z_next = manifold.projx(z_next)

        if RUNTIME_STATS:
            print(f"Step {i} | Drift: {u.norm().mean():.4f}")
            measure_manifold_distance(z_next, z_prev, "Step Distance")
        z_prev = z_next

    measure_manifold_distance(z_prev, z_start, "Start vs End")
    return z_prev


@torch.inference_mode()
def improve_encodings(
    input_path,
    noise_std=NOISE_STD,
    start_t=START_T,
    generation=False,
    generation_samples=N_SAMPLES,
):
    """Loads existing .pt file and pushes encodings through the flow."""
    print(f"Loading existing encodings from {input_path}")
    data = np.load(input_path, allow_pickle=False)
    z_input = torch.from_numpy(data["encodings"]).float()
    labels_input = torch.from_numpy(data["labels"]).long()
    split_ids = torch.from_numpy(data["split_ids"]).long()
    split_names = data["split_names"].tolist()
    class_means = data.get("class_means", None)

    if class_means is not None:
        class_means = torch.from_numpy(class_means).squeeze(1).float()
        print("Loaded class means.")

    if not generation and N_SAMPLES != -1 and z_input.shape[0] > N_SAMPLES:
        total_samples = z_input.shape[0]
        permutation = torch.randperm(total_samples)

        subsample_indices = permutation[:N_SAMPLES]

        z_input = z_input[subsample_indices]
        labels_input = labels_input[subsample_indices]
        split_ids = split_ids[subsample_indices]

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

    if SQUEEZE_DATA:
        z_input, _ = manifold_squeeze(z_input, labels_input, class_means=class_means, alpha=SQUEEZE_ALPHA, reverse=False)

    if CHECK_UNIFORMITY:
        check_uniformity(z_input)

    if generation:
        labels = dummy_labels(generation_samples)
    else:
        labels = labels_input

    z_init = manifold.random_base(generation_samples if generation else len(z_input), z_input.shape[-1])
    # mean = class_means[labels]
    # noise = torch.randn((generation_samples, 1024)) * 0.5
    # z_init = -mean + noise
    # z_init = manifold.projx(z_init)

    if generation:
        z_noise = z_init
    else:
        noise = torch.randn_like(z_input) * noise_std
        z_noise = z_input + noise
        z_noise = manifold.projx(z_noise)

    print(f"Refining {z_noise.shape[0]} samples...")
    z_final = integrate_flow(z_noise, labels, steps=STEPS, start_t=start_t)

    print("Clustering report:")
    check_class_clustering(z_noise, labels, text="Noise Class Clustering")
    check_class_clustering(z_input, labels_input, text="Original Class Clustering")
    check_class_clustering(z_final, labels, text="Final Class Clustering")

    if not generation:
        print("Manifold distance stats:")
        measure_manifold_distance(z_init, z_input, text="Full Noise vs Original")
        measure_manifold_distance(z_noise, z_input, text="Noise vs Original")
        measure_manifold_distance(z_noise, z_final, text="Noise vs Improved")
        measure_manifold_distance(z_final, z_input, text="Improved vs Original")

    if SQUEEZE_DATA:
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

if SAVE_OUTPUT:
    np.savez_compressed(
        OUTPUT_PATH,
        allow_pickle=False,
        encodings=z_final.cpu().numpy(),
        labels=labels.cpu().numpy(),
        split_ids=split_ids.cpu().numpy(),
        split_names=np.array(split_names, dtype=str),
    )

print(f"Done. Saved shape: {z_final.shape}")
