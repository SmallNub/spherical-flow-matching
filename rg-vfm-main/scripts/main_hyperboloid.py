import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

from data.toys import CheckerboardHyperbolic

from rvf.solver.wrappers_euclidean import VelocityInference, VelocityWrapper
from rvf.solver.wrappers_hyperboloid import VelocityInferenceHyperboloid
from rvf.models.dynamics_models_euclidean import VectorDynamics, PositionDynamics
from rvf.models.dynamics_models_hyperboloid import VectorDynamicsHyperboloid, PositionDynamicsHyperboloid
from rvf.visualisations.hyperbolic_mesh import data_on_hyperboloid, mesh_checkerboard_on_hyperboloid, data_on_hyperboloid_top, mesh_checkerboard_on_hyperboloid_top
from rvf.visualisations.base_mesh import add_x1_background
from rvf.utils.utils import torch2npy, save_model, load_model
from rvf.solver.ode_solver import ODESolver
from rvf.manifolds.hyperboloid import HyperboloidManifold
from scripts.mmd_metrics import compute_checkerboard_coverage, compute_c2st_score

from rvf.losses.loss_hyperboloid import VariationalLossHyperboloid, VanillaLossHyperboloid

import torch.utils.data as data

from tqdm import tqdm
import warnings

import copy
import os
import time

import argparse

import random

def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=3000)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--noise_scale", type=float, default=1e-1)
    # model types
    parser.add_argument("--flow", type=str, default="variational", choices=["vanilla", "variational"])
    parser.add_argument("--geometry", type=str, default="riemannian", choices=["euclidean", "riemannian"])
    parser.add_argument("--support", type=str, default="intrinsic", choices=["extrinsic", "intrinsic"])
    parser.add_argument("--p0_distribution", type=str, default="gaussian", choices=["uniform", "gaussian"])
    # training
    parser.add_argument('--train', action='store_true', default=True, help="Enable training")
    parser.add_argument("--wandb", action='store_true', default=False, help="Log to wandb")
    return parser.parse_args()


def initialize_p0(
    p0_distribution: str,
    support: str,
    num_samples: int
    ):
    """
    Sample initial points x0 and return a log_p0 function for those distributions.

    Args:
        p0_distribution: 'uniform' or 'gaussian'
        support: 'extrinsic' or 'intrinsic'
        num_samples: number of samples to draw

    Returns:
        x0: Tensor of shape [num_samples, dim]
        log_p0: Callable mapping an input tensor of shape [batch, dim] to log-density [batch]
    """
    if support == "extrinsic":
        dim    = 3
        center = torch.tensor([1.0, 0.0, 0.0])   # shift mean to [1,0,0]
    elif support == "intrinsic":
        dim    = 2
        center = torch.zeros(2)                  # mean stays at [0,0]

    if p0_distribution == "uniform":
        x0 = 2*torch.rand(num_samples, dim) - 1
    elif p0_distribution == "gaussian":
        x0 = torch.randn(num_samples, dim)
    else:
        raise ValueError(f"Unknown p0_distribution={p0_distribution!r}")

    # make sure center is on the same device & dtype, then shift
    center = center.to(x0.device).type_as(x0)     # (dim,)
    x0 = x0 + center                              # broadcast to (num_samples, dim)

    if support == "intrinsic":
        x0 = HyperboloidManifold().wrap(x0)
        norm_xx = HyperboloidManifold()._minkowski_dot(x0, x0)
        tol = 1e-2
        assert torch.allclose(norm_xx, torch.tensor(-1.0, device=norm_xx.device), atol=tol), (
            f"Points do not lie on the hyperboloid: max deviation = "
            f"{(norm_xx + 1.0).abs().max().item():.2e}"
        )

    def log_p0(x: torch.Tensor):
        """
        Log-density under the p0_distribution at points x.
        Expects x on the manifold if intrinsic, or Euclidean shifted by center if extrinsic.
        """
        if support == "intrinsic":
            manifold = HyperboloidManifold()
            origin = manifold._get_origin(x.shape[0], dim, x.device)
            v = manifold.log_map(origin, x)
            x_loc = v[..., 1:]
        else:
            x_loc = x - center

        if p0_distribution == "uniform":
            in_support = (x_loc >= -1).all(dim=-1) & (x_loc <= 1).all(dim=-1)
            log_density = torch.full(x_loc.shape[:-1], -dim * torch.log(torch.tensor(2.0, device=x_loc.device)))
            log_density = torch.where(in_support, log_density, torch.tensor(float('-inf'), device=x_loc.device))
        else:
            sq_norm = (x_loc ** 2).sum(dim=-1)
            log_norm_const = -0.5 * dim * torch.log(torch.tensor(2 * torch.pi, device=x_loc.device))
            log_density = log_norm_const - 0.5 * sq_norm
        return log_density

    return x0, log_p0

def train_flow(model, loss_fn, trainloader, num_epoch, lr, p0_distribution, support):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model = None
    best_loss = float('inf')
    torch.autograd.set_detect_anomaly(True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(num_epoch), desc="Training")

    for epoch in pbar:
        total_loss = 0.0
        for x1 in trainloader:
            t = torch.rand(x1.shape[0], 1, device=device)
            x0, _ = initialize_p0(p0_distribution, support, x1.shape[0])
            loss = loss_fn(model, x0, x1, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss/len(trainloader)
        wandb.log({"loss": epoch_loss})

        if total_loss < best_loss:
            best_loss = total_loss
            best_model = copy.deepcopy(model)

        pbar.set_description(f"Training (loss={epoch_loss:.6f})")

    return best_model


def initialize_model(flow, geometry, hidden_dim, support, noise_scale):
    if geometry == "euclidean":
        if flow == "vanilla":
            model = VectorDynamics(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
            velocity = VelocityWrapper(model)
            flow_matching = VanillaLossHyperboloid(noise_scale=noise_scale)
            loss = flow_matching.loss_euclidean

        elif flow == "variational":
            model = PositionDynamics(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
            velocity = VelocityInference(model)
            flow_matching = VariationalLossHyperboloid(noise_scale=noise_scale)
            loss = flow_matching.loss_euclidean

    if geometry == "riemannian":
        if flow == "vanilla":
            if support == "extrinsic":
                warnings.warn(f"Extrinsic support for Riemannian Flow Matching might lead to errors")
                model = VectorDynamicsHyperboloid(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
                velocity = VelocityWrapper(model)
                flow_matching = VanillaLossHyperboloid(noise_scale=noise_scale)
                loss = flow_matching.loss_extrinsic

            elif support == "intrinsic":
                model = VectorDynamicsHyperboloid(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
                velocity = VelocityWrapper(model)
                flow_matching = VanillaLossHyperboloid(noise_scale=noise_scale)
                loss = flow_matching.loss_intrinsic

        elif flow == "variational":
            if support == "extrinsic":
                model = PositionDynamicsHyperboloid(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
                velocity = VelocityInference(model)
                flow_matching = VariationalLossHyperboloid(noise_scale=noise_scale)
                loss = flow_matching.loss_extrinsic

            elif support == "intrinsic":
                model = PositionDynamicsHyperboloid(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
                velocity = VelocityInferenceHyperboloid(model)
                flow_matching = VariationalLossHyperboloid(noise_scale=noise_scale)
                loss = flow_matching.loss_intrinsic

    return model, velocity, loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opts = parse_args()

    # Fix the seed for reproducibility
    seed = 42  # Replace with any integer seed you prefer
    torch.manual_seed(seed)  # PyTorch CPU and CUDA
    np.random.seed(seed)  # NumPy
    random.seed(seed)  # Python's random module

    # If you're using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set the seed for CUDA
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Optional: Make deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load config
    batch_size = opts.batch_size
    hidden_dim = opts.hidden_dim
    num_epoch = opts.num_epoch
    num_samples = opts.num_samples
    lr = opts.lr
    flow = opts.flow
    geometry = opts.geometry
    p0_distribution = opts.p0_distribution
    support = opts.support   
    noise_scale = opts.noise_scale
    train = opts.train
    wand_active = opts.wandb

    # create folder and save config
    if geometry == "euclidean": support = "extrinsic"
    print(f"Running experiment with {flow} {geometry} {support} and {p0_distribution}")
    print(f"Training: {train}")
    
    folder = f"experiments/hyperboloid/{flow}/{geometry}/{support}/{p0_distribution}/"    
    os.makedirs(folder, exist_ok=True)
    print(f"Saving results to {folder}")

    with open(folder + 'config.txt', 'w') as f:
        for key, value in opts.__dict__.items():
            f.write(f"{key}: {value}\n")

    # Initialize wandb
    if wand_active == True and train == True:
        wandb.init(
            project="VRFM",
            config=opts,
        )
    else:
        wandb.init(mode="disabled")


    # prepare data
    dataset = CheckerboardHyperbolic(num_samples)
    trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize model
    model, velocity, loss_fn = initialize_model(flow, geometry, hidden_dim, support, noise_scale)

    # Train the flow model
    if train == True:
        training_start_time = time.time()
        model = train_flow(model, loss_fn, trainloader, num_epoch, lr, p0_distribution, support)
        training_time = time.time() - training_start_time
        save_model(model, folder)
        print(f"Saved model to {folder}")
    
    else:
        training_time = 0.0
        model = load_model(model, folder)
        print(f"Loaded model from {folder}")

    # compute probability paths from t=0 to t=1
    x0, log_p0 = initialize_p0(p0_distribution, support, 10000) # 2500 points to plot the density
    t_span = torch.linspace(0.0, 0.99, steps=101)  # Uniform time grid
    times_to_show = [0.00, 0.25, 0.5, 0.75, 0.99] # t=0 (uniform) to t=1 (checkerboard)

    print("Computing probability paths...")
    x0 = x0.to(device)

    # Time the generation process
    generation_start_time = time.time()
    solver = ODESolver(velocity_model=velocity, manifold=HyperboloidManifold())   # defaults to Euler
    sols   = solver.sample(x0=x0, t_span=t_span, support=support)
    x1_gen = sols[-1]
    generation_time = time.time() - generation_start_time

    # get 5000 points from trainloader
    dataset = CheckerboardHyperbolic(num_samples)
    trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x1 = trainloader.dataset[:10000]
    x1_flat = HyperboloidManifold().unwrap(x1)

    print("Plotting probability paths...")
    # plot the density unwrapped in 2d from the sphere to the plane
    x1_gen_flat = torch2npy(HyperboloidManifold().unwrap(x1_gen))
    fig, ax = plt.subplots()
    ax = add_x1_background(ax, x1_flat, alpha=0.15)
    ax.scatter(x1_gen_flat[:, 0], x1_gen_flat[:, 1], s=1, color="red")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(folder, "density_unwrapped.png"), dpi=300)    
    wandb.log({"density_unwrapped": wandb.Image(plt)})

    # Plot histogram of hyperboloid constraint violations
    print("Analyzing hyperboloid constraint violation...")
    manifold = HyperboloidManifold()
    
    # Compute Minkowski dot product: <x,x> should equal -1 for hyperboloid points
    minkowski_dots = manifold._minkowski_dot(x1_gen, x1_gen).detach().cpu().numpy()
    constraint_violations = np.abs(minkowski_dots + 1.0)  # Distance from -1
    
    # Plot histogram of constraint violations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Minkowski dot products
    ax1.hist(minkowski_dots, bins=100, range=(-1.1, -0.9), alpha=0.7, edgecolor='black')
    ax1.axvline(x=-1.0, color='red', linestyle='--', linewidth=2, label='Perfect hyperboloid (<x,x>=-1)')
    ax1.set_xlabel('Minkowski Dot Product <x,x>')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of <x,x> (Should be ~-1.0)')
    ax1.legend()
    
    # Right plot: Constraint violation distances
    ax2.hist(constraint_violations, bins=100, range=(0, 0.1), alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='Perfect constraint (|<x,x>+1|=0)')
    ax2.set_xlabel('Constraint Violation |<x,x> + 1|')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distance from Hyperboloid Surface')
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(folder, 'hyperboloid_constraint_histogram.svg'), dpi=300)
    fig.savefig(os.path.join(folder, 'hyperboloid_constraint_histogram.png'), dpi=300)
    wandb.log({"hyperboloid_constraint_histogram": wandb.Image(plt)})
    plt.close(fig)
    
    # Compute statistics
    average_minkowski = np.mean(minkowski_dots)
    std_minkowski = np.std(minkowski_dots)
    average_violation = np.mean(constraint_violations)
    max_violation = np.max(constraint_violations)
    percentage_within_tol = np.mean(constraint_violations < 1e-2) * 100
    
    # Save statistics to file
    with open(os.path.join(folder, 'hyperboloid_constraint_stats.txt'), 'w') as f:
        f.write(f"Hyperboloid Constraint Analysis\n")
        f.write(f"===============================\n")
        f.write(f"Average <x,x>: {average_minkowski:.6f}\n")
        f.write(f"Std <x,x>: {std_minkowski:.6f}\n")
        f.write(f"Average Constraint Violation: {average_violation:.6f}\n")
        f.write(f"Max Constraint Violation: {max_violation:.6f}\n")
        f.write(f"Points within 1e-2 tolerance: {percentage_within_tol:.2f}%\n")
    
    print(f"Hyperboloid constraint stats: avg_<x,x>={average_minkowski:.4f}, avg_violation={average_violation:.4f}, within_tol={percentage_within_tol:.1f}%")

    # Compute C2ST and Coverage metrics only
    print("\nEvaluating C2ST and coverage metrics...")
    
    # Ensure tensors are on CPU
    x1_real_cpu = x1.cpu() if x1.is_cuda else x1
    x1_gen_cpu = x1_gen.detach().cpu() if x1_gen.is_cuda else x1_gen.detach()
    
    # 1. Compute coverage metrics
    print("Computing coverage metrics...")
    coverage_results = compute_checkerboard_coverage(x1_gen_cpu, HyperboloidManifold())
    
    # 2. Compute C2ST metrics
    print("Computing C2ST metrics...")
    c2st_results = compute_c2st_score(
        x1_real_cpu, 
        x1_gen_cpu, 
        use_2d_unwrapped=True,
        sphere_manifold=HyperboloidManifold(),  # Pass hyperboloid manifold
        num_runs=3,
        num_epochs=50
    )
    
    # Save results to file
    with open(os.path.join(folder, 'coverage_c2st_metrics.txt'), 'w') as f:
        f.write("COVERAGE AND C2ST METRICS (HYPERBOLOID)\n")
        f.write("=" * 45 + "\n\n")
        
        f.write("COVERAGE METRICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Coverage (% in correct regions): {coverage_results['coverage']*100:.2f}%\n")
        f.write(f"Modes covered: {coverage_results['modes_covered']}/{coverage_results['total_modes']}\n")
        f.write(f"Uniformity CV: {coverage_results['uniformity_cv']:.4f} (lower is better)\n")
        f.write(f"Samples per mode: {coverage_results['samples_per_mode']}\n\n")
        
        f.write("C2ST METRICS\n")
        f.write("-" * 15 + "\n")
        f.write(f"C2ST Accuracy: {c2st_results['c2st_accuracy']*100:.2f}% ± {c2st_results['c2st_std']*100:.2f}%\n")
        f.write(f"C2ST Score: {c2st_results['c2st_score']:.4f}\n")
        f.write(f"Interpretation: ")
        if c2st_results['c2st_accuracy'] < 0.55:
            f.write("Excellent - distributions are nearly indistinguishable\n")
        elif c2st_results['c2st_accuracy'] < 0.65:
            f.write("Good - minor differences detected\n")
        elif c2st_results['c2st_accuracy'] < 0.75:
            f.write("Moderate - noticeable differences\n")
        else:
            f.write("Poor - significant differences detected\n")
        
        f.write("\nPERFORMANCE METRICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
        f.write(f"Generation Time: {generation_time:.2f} seconds for {len(x1_gen)} samples\n")
        if generation_time > 0:
            f.write(f"Generation Rate: {len(x1_gen)/generation_time:.1f} samples/second\n")
    
    print(f"Coverage and C2ST results saved to {folder}coverage_c2st_metrics.txt")
    print(f"Coverage: {coverage_results['coverage']*100:.1f}%, C2ST Accuracy: {c2st_results['c2st_accuracy']*100:.1f}%")

    n_snapshots = len(times_to_show)
    n_cols      = n_snapshots + 1   # <-- one extra for the real x1
    fig, axs = plt.subplots(
        1, n_cols,
        figsize=(6 * n_cols, 5),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    # draw your time‐snapshots in cols 0…n_snapshots-1
    for i, t_end in enumerate(times_to_show):
        t_idx = torch.searchsorted(t_span, torch.tensor(t_end)).item()
        x_t   = sols[t_idx]
        axs[i] = data_on_hyperboloid(x_t, axs[i])
        axs[i] = mesh_checkerboard_on_hyperboloid(axs[i])
        title = "t=1.00" if t_end == 0.99 else f"t={t_end:.2f}"
        axs[i].set_title(title)

    # now in the last column, plot your real x1
    # NOTE: x1 should be in ambient coords on the hyperboloid (shape [N,3])
    real_x1 = trainloader.dataset[:10000].to(device)
    axs[-1] = data_on_hyperboloid(real_x1, axs[-1])
    axs[-1] = mesh_checkerboard_on_hyperboloid(axs[-1])
    axs[-1].set_title("x₁")

    fig.subplots_adjust(wspace=0.01, left=0.02, right=0.98)
    fig.savefig(os.path.join(folder, "probability_paths_hyperboloid_with_x1.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    wandb.log({"probability_paths_with_x1": wandb.Image(plt)})


    n_snapshots = len(times_to_show)
    n_cols      = n_snapshots + 1

    fig, axs = plt.subplots(
        1, n_cols,
        figsize=(6 * n_cols, 5),
        subplot_kw={"projection": "3d"},
        gridspec_kw={
            "left": 0.02,
            "right": 0.98,
            "top": 0.90,      # leave room for titles
            "wspace": 0.01,   # very minimal horizontal gap
            "hspace": 0.0,
        },
    )

    for i, t_end in enumerate(times_to_show):
        t_idx = torch.searchsorted(t_span, torch.tensor(t_end)).item()
        x_t   = sols[t_idx]
        axs[i] = data_on_hyperboloid_top(x_t, axs[i])
        axs[i] = mesh_checkerboard_on_hyperboloid_top(axs[i])
        title = "t=1.00" if t_end == 0.99 else f"t={t_end:.2f}"
        axs[i].set_title(title)

    # final real‐data panel
    axs[-1] = data_on_hyperboloid_top(real_x1, axs[-1])
    axs[-1] = mesh_checkerboard_on_hyperboloid_top(axs[-1])
    axs[-1].set_title("x₁")

    fig.savefig(
        os.path.join(folder, "probability_paths_hyperboloid_top_with_x1.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    wandb.log({"probability_paths_top_with_x1": wandb.Image(plt)})

    plt.show()


    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    import wandb
    main()
