import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import warnings
from data.toys import CheckerboardSphere

from rvf.solver.wrappers_euclidean import VelocityInference, VelocityWrapper
from rvf.solver.wrappers_sphere import VelocityInferenceSphere
from rvf.models.dynamics_models_euclidean import VectorDynamics, PositionDynamics
from rvf.models.dynamics_models_sphere import VectorDynamicsSphere, PositionDynamicsSphere
from rvf.visualisations.sphere_mesh import data_on_sphere, mesh_checkerboard_on_sphere
from rvf.visualisations.base_mesh import add_x1_background
from rvf.utils.utils import torch2npy, save_model, load_model

from rvf.manifolds.sphere import SphereManifold
from rvf.solver.ode_solver import ODESolver
from scripts.mmd_metrics import compute_checkerboard_coverage, compute_c2st_score

from rvf.losses.loss_sphere import VariationalLossSphere, VanillaLossSphere

import torch.utils.data as data

from tqdm import tqdm

import copy
import os
import time

import argparse

import random

def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=3000)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--noise_scale", type=float, default=1.0e-1)
    # model types
    parser.add_argument("--flow", type=str, default="variational", choices=["vanilla", "variational"])
    parser.add_argument("--geometry", type=str, default="riemannian", choices=["euclidean", "riemannian"])
    parser.add_argument("--support", type=str, default="intrinsic", choices=["extrinsic", "intrinsic"])
    parser.add_argument("--p0_distribution", type=str, default="gaussian", choices=["uniform", "gaussian"])
    # training
    parser.add_argument('--train', action='store_true', default=True, help="Enable training")
    parser.add_argument("--wandb", action='store_true', default=False, help="Log to wandb")
    return parser.parse_args()


def initialize_p0(p0_distribution, support, num_samples):
    if support == "extrinsic":
        dim = 3
    elif support == "intrinsic":
        dim = 2

    if p0_distribution == "uniform": # uniform distribution
        x0 = 2*torch.rand(num_samples, dim) - 1

    elif p0_distribution == "gaussian":
        x0 = torch.randn(num_samples, dim)

    if support == "intrinsic":
        x0 = SphereManifold().wrap(x0)

    return x0

def train_flow(model, loss_fn, trainloader, num_epoch, lr, p0_distribution, support):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model = None
    best_loss = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    pbar = tqdm(range(num_epoch), desc="Training")

    for epoch in pbar:
        total_loss = 0.0
        for x1 in trainloader:
            t = torch.rand(x1.shape[0], 1, device=device)
            x0 = initialize_p0(p0_distribution, support, x1.shape[0])
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
            flow_matching = VanillaLossSphere(noise_scale=noise_scale)
            loss = flow_matching.loss_euclidean

        elif flow == "variational":
            model = PositionDynamics(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
            velocity = VelocityInference(model)
            flow_matching = VariationalLossSphere(noise_scale=noise_scale)
            loss = flow_matching.loss_euclidean

    if geometry == "riemannian":
        if flow == "vanilla":
            if support == "extrinsic":
                warnings.warn(f"Extrinsic support for Riemannian Flow Matching might lead to errors")
                model = VectorDynamicsSphere(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
                velocity = VelocityWrapper(model)
                flow_matching = VanillaLossSphere(noise_scale=noise_scale)
                loss = flow_matching.loss_extrinsic

            elif support == "intrinsic":
                model = VectorDynamicsSphere(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
                velocity = VelocityWrapper(model)
                flow_matching = VanillaLossSphere(noise_scale=noise_scale)
                loss = flow_matching.loss_intrinsic

        elif flow == "variational":
            if support == "extrinsic":
                model = PositionDynamicsSphere(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
                velocity = VelocityInference(model)
                flow_matching = VariationalLossSphere(noise_scale=noise_scale)
                loss = flow_matching.loss_extrinsic

            elif support == "intrinsic":
                model = PositionDynamicsSphere(input_dim=3, time_dim=1, hidden_dim=hidden_dim)
                velocity = VelocityInferenceSphere(model)
                flow_matching = VariationalLossSphere(noise_scale=noise_scale)
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
    
    folder = f"experiments/sphere/{flow}/{geometry}/{support}/{p0_distribution}/"    
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
    dataset = CheckerboardSphere(num_samples)
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
    x0 = initialize_p0(p0_distribution, support, 10000) # 2500 points to plot the density
    t_span = torch.linspace(0.0, 0.99, steps=101)  # Uniform time grid
    times_to_show = [0.00, 0.25, 0.5, 0.75, 0.99] # t=0 (uniform) to t=1 (checkerboard)

    print("Computing probability paths...")
    
    # Determine the appropriate support based on geometry and support settings
    if geometry == "riemannian" and support == "intrinsic":
        solver_support = "intrinsic"
        manifold = SphereManifold()
    else:
        solver_support = "extrinsic"
        manifold = SphereManifold()  # Still pass manifold for consistency
    
    # Time the generation process
    generation_start_time = time.time()
    solver = ODESolver(velocity_model=velocity, manifold=manifold)
    sols = solver.sample(x0=x0.to(device), t_span=t_span, support=solver_support)
    x1_gen = sols[-1]
    generation_time = time.time() - generation_start_time

    x1 = trainloader.dataset[:10000]
    x1_flat = SphereManifold().unwrap(x1)

    print("Plotting probability paths...")
    # plot the density unwrapped in 2d from the sphere to the plane
    x1_gen_flat = torch2npy(SphereManifold().unwrap(x1_gen))
    fig, ax = plt.subplots()
    ax = add_x1_background(ax, x1_flat, alpha=0.15)
    ax.scatter(x1_gen_flat[:, 0], x1_gen_flat[:, 1], s=1, color="red")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(folder, "density_unwrapped.png"), dpi=300)
    wandb.log({"density_unwrapped": wandb.Image(plt)})
    plt.close(fig)

    # Plot histogram of norms to check how well points lie on the sphere
    print("Analyzing sphere constraint violation...")
    norms = torch.norm(x1_gen, dim=1).detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.hist(norms, bins=100, range=(0.9, 1.1), alpha=0.7, edgecolor='black')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect sphere (norm=1)')
    ax.set_xlabel('Norm ||x||')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Point Norms (Should be ~1.0 for Sphere)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(folder, 'norms_histogram.svg'), dpi=300)
    fig.savefig(os.path.join(folder, 'norms_histogram.png'), dpi=300)
    wandb.log({"norms_histogram": wandb.Image(plt)})
    plt.close(fig)
    
    # Compute statistics
    average_norm = np.mean(norms)
    std_norm = np.std(norms)
    deviation_from_unit = np.abs(norms - 1.0)
    max_deviation = np.max(deviation_from_unit)
    percentage_within_tol = np.mean(deviation_from_unit < 1e-2) * 100
    
    # Save statistics to file
    with open(os.path.join(folder, 'sphere_constraint_stats.txt'), 'w') as f:
        f.write(f"Sphere Constraint Analysis\n")
        f.write(f"========================\n")
        f.write(f"Average Norm: {average_norm:.6f}\n")
        f.write(f"Std Norm: {std_norm:.6f}\n")
        f.write(f"Max Deviation from 1.0: {max_deviation:.6f}\n")
        f.write(f"Points within 1e-2 tolerance: {percentage_within_tol:.2f}%\n")
    
    print(f"Sphere constraint stats: avg_norm={average_norm:.4f}, std={std_norm:.4f}, within_tol={percentage_within_tol:.1f}%")

    # Compute C2ST and Coverage metrics only
    print("\nEvaluating C2ST and coverage metrics...")
    
    # Ensure tensors are on CPU
    x1_real_cpu = x1.cpu() if x1.is_cuda else x1
    x1_gen_cpu = x1_gen.detach().cpu() if x1_gen.is_cuda else x1_gen.detach()
    
    # 1. Compute coverage metrics
    print("Computing coverage metrics...")
    coverage_results = compute_checkerboard_coverage(x1_gen_cpu, SphereManifold())
    
    # 2. Compute C2ST metrics
    print("Computing C2ST metrics...")
    c2st_results = compute_c2st_score(
        x1_real_cpu, 
        x1_gen_cpu, 
        use_2d_unwrapped=True,
        sphere_manifold=SphereManifold(),  # Pass sphere manifold
        num_runs=3,
        num_epochs=50
    )
    
    # Save results to file
    with open(os.path.join(folder, 'coverage_c2st_metrics.txt'), 'w') as f:
        f.write("COVERAGE AND C2ST METRICS (SPHERE)\n")
        f.write("=" * 40 + "\n\n")
        
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

    # plot the paths with ground truth x1
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
        x_t = torch2npy(sols[t_idx])
        axs[i] = data_on_sphere(x_t, axs[i])
        axs[i] = mesh_checkerboard_on_sphere(axs[i])
        title = "t=1.00" if t_end == 0.99 else f"t={t_end:.2f}"
        axs[i].set_title(title)

    # now in the last column, plot your real x1
    # NOTE: x1 should be in ambient coords on the sphere (shape [N,3])
    real_x1 = trainloader.dataset[:10000].to(device)
    axs[-1] = data_on_sphere(torch2npy(real_x1), axs[-1])
    axs[-1] = mesh_checkerboard_on_sphere(axs[-1])
    axs[-1].set_title("x₁")

    fig.subplots_adjust(wspace=0.01, left=0.02, right=0.98)
    fig.savefig(os.path.join(folder, "probability_paths.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    wandb.log({"probability_paths": wandb.Image(plt)})
    plt.close(fig)

    # Save individual sphere images without titles
    print("Saving individual sphere images...")
    individual_folder = os.path.join(folder, "individual_spheres")
    os.makedirs(individual_folder, exist_ok=True)

    # Save each timestep as an individual image
    for i, t_end in enumerate(times_to_show):
        t_idx = torch.searchsorted(t_span, torch.tensor(t_end)).item()
        x_t = torch2npy(sols[t_idx])

        fig_single = plt.figure(figsize=(6, 5))
        ax_single = fig_single.add_subplot(111, projection='3d')
        ax_single = data_on_sphere(x_t, ax_single, color="magenta")
        ax_single = mesh_checkerboard_on_sphere(ax_single)

        # Save with timestep in filename
        filename = f"sphere_t_{t_end:.2f}.png" if t_end != 0.99 else "sphere_t_1.00.png"
        fig_single.savefig(os.path.join(individual_folder, filename),
                          dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig_single)

    # Save the real x1 as a separate image
    fig_real = plt.figure(figsize=(6, 5))
    ax_real = fig_real.add_subplot(111, projection='3d')
    ax_real = data_on_sphere(torch2npy(real_x1), ax_real, color="magenta")
    ax_real = mesh_checkerboard_on_sphere(ax_real)
    fig_real.savefig(os.path.join(individual_folder, "sphere_x1_real.png"),
                     dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_real)

    print(f"Individual sphere images saved to {individual_folder}")

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    import wandb
    main()
