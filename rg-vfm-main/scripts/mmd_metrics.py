import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple, Dict


def get_checkerboard_black_squares_correct():
    """
    Define the correct black squares based on the actual data generation in toys.py.
    
    The pattern creates points where (cell_x + cell_y) % 2 == 1 (black squares)
    After normalization to [-1, 1], this corresponds to:
    """
    squares = []
    num_squares = 4
    
    for row in range(num_squares):
        for col in range(num_squares):
            # Check if this is a black square (where points should be)
            if (col + row) % 2 == 1:  # Black squares
                # Convert to [-1, 1] coordinates
                x_min = -1.0 + col * (2.0 / num_squares)
                x_max = -1.0 + (col + 1) * (2.0 / num_squares)
                y_min = -1.0 + row * (2.0 / num_squares)
                y_max = -1.0 + (row + 1) * (2.0 / num_squares)
                
                squares.append((x_min, x_max, y_min, y_max))
    
    return squares


def compute_boundary_aware_metrics(samples_2d: np.ndarray, 
                                  real_samples_2d: np.ndarray = None) -> Dict[str, float]:
    """
    Compute metrics that reward sharp boundaries and mode concentration.
    
    Args:
        samples_2d: Generated samples in 2D (after unwrapping)
        real_samples_2d: Real samples for comparison (optional)
    
    Returns:
        Dictionary of boundary-aware metrics
    """
    metrics = {}
    
    # 1. Precision-based Coverage
    # What fraction of generated points are in valid squares?
    precision = compute_precision_coverage(samples_2d)
    metrics['precision_coverage'] = precision
    
    # 2. Mode Separation Score
    # How well separated are the modes?
    separation = compute_mode_separation(samples_2d)
    metrics['mode_separation'] = separation
    
    # 3. Boundary Sharpness
    # How sharp are the transitions between modes?
    sharpness = compute_boundary_sharpness(samples_2d)
    metrics['boundary_sharpness'] = sharpness
    
    # 4. Concentration Score
    # How concentrated are points within each mode?
    concentration = compute_concentration_score(samples_2d)
    metrics['concentration_score'] = concentration
    
    # 5. Inter-mode Contamination
    # What fraction of points fall between modes?
    contamination = compute_inter_mode_contamination(samples_2d)
    metrics['inter_mode_contamination'] = contamination
    
    # 6. F1-style Score
    if real_samples_2d is not None:
        f1_score = compute_checkerboard_f1(samples_2d, real_samples_2d)
        metrics['checkerboard_f1'] = f1_score
    
    return metrics


def compute_precision_coverage(samples_2d: np.ndarray) -> float:
    """
    CORRECTED: Precision: fraction of generated samples that fall within valid BLACK squares.
    This uses the correct checkerboard pattern from toys.py.
    """
    black_squares = get_checkerboard_black_squares_correct()
    
    total_in_squares = 0
    for x_min, x_max, y_min, y_max in black_squares:
        in_square = ((samples_2d[:, 0] >= x_min) & (samples_2d[:, 0] < x_max) & 
                    (samples_2d[:, 1] >= y_min) & (samples_2d[:, 1] < y_max))
        total_in_squares += in_square.sum()
    
    return total_in_squares / len(samples_2d)


def compute_mode_separation(samples_2d: np.ndarray) -> float:
    """
    CORRECTED: Compute how well separated the modes are using correct BLACK square centers.
    Higher score = better separation between modes.
    """
    # Define mode centers based on correct black squares
    black_squares = get_checkerboard_black_squares_correct()
    mode_centers = []
    
    for x_min, x_max, y_min, y_max in black_squares:
        # Center of each black square
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        mode_centers.append((x_center, y_center))
    
    mode_centers = np.array(mode_centers)
    
    # Assign each point to nearest mode
    distances = np.zeros((len(samples_2d), len(mode_centers)))
    for i, center in enumerate(mode_centers):
        distances[:, i] = np.sqrt((samples_2d[:, 0] - center[0])**2 + 
                                 (samples_2d[:, 1] - center[1])**2)
    
    nearest_mode = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    
    # Compute second nearest distance
    distances_copy = distances.copy()
    distances_copy[np.arange(len(samples_2d)), nearest_mode] = np.inf
    second_min_distances = np.min(distances_copy, axis=1)
    
    # Separation score: ratio of second nearest to nearest
    # Higher ratio = better separation
    separation_ratios = second_min_distances / (min_distances + 1e-8)
    
    # Use median to be robust to outliers
    return np.median(separation_ratios)


def compute_boundary_sharpness(samples_2d: np.ndarray, n_bins: int = 50) -> float:
    """
    Compute how sharp the boundaries are between modes and empty space.
    Uses gradient of density estimate.
    """
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        samples_2d[:, 0], samples_2d[:, 1], 
        bins=n_bins, range=[[-1.1, 1.1], [-1.1, 1.1]]
    )
    
    # Smooth slightly to reduce noise
    from scipy.ndimage import gaussian_filter
    hist_smooth = gaussian_filter(hist, sigma=0.5)
    
    # Compute gradients
    grad_x, grad_y = np.gradient(hist_smooth)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Boundary sharpness = mean gradient magnitude at boundaries
    # Normalize by maximum possible gradient
    sharpness = np.mean(gradient_magnitude) / (np.max(hist_smooth) + 1e-8)
    
    return sharpness


def compute_concentration_score(samples_2d: np.ndarray) -> float:
    """
    CORRECTED: Compute how concentrated points are within their assigned modes.
    Lower dispersion within modes = higher score.
    """
    # Use correct black square centers
    black_squares = get_checkerboard_black_squares_correct()
    mode_centers = []
    
    for x_min, x_max, y_min, y_max in black_squares:
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        mode_centers.append((x_center, y_center))
    
    mode_centers = np.array(mode_centers)
    
    # Assign each point to nearest mode
    distances = np.zeros((len(samples_2d), len(mode_centers)))
    for i, center in enumerate(mode_centers):
        distances[:, i] = np.sqrt((samples_2d[:, 0] - center[0])**2 + 
                                 (samples_2d[:, 1] - center[1])**2)
    
    nearest_mode = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    
    # Compute average distance to assigned mode center
    # Normalize by expected distance for uniform distribution in square (≈0.29 for 0.5×0.5 square)
    avg_distance = np.mean(min_distances)
    concentration = 1.0 - min(avg_distance / 0.29, 1.0)
    
    return concentration


def compute_inter_mode_contamination(samples_2d: np.ndarray) -> float:
    """
    CORRECTED: Compute fraction of points that fall in 'forbidden' regions (WHITE squares).
    Lower is better. This now correctly identifies contamination as points NOT in black squares.
    """
    # Points are contamination if they're not in any valid BLACK square
    in_any_black_square = np.zeros(len(samples_2d), dtype=bool)
    
    # Check all black squares
    black_squares = get_checkerboard_black_squares_correct()
    for x_min, x_max, y_min, y_max in black_squares:
        in_square = ((samples_2d[:, 0] >= x_min) & (samples_2d[:, 0] < x_max) & 
                    (samples_2d[:, 1] >= y_min) & (samples_2d[:, 1] < y_max))
        in_any_black_square |= in_square
    
    # Contamination = points NOT in any black square
    contamination = 1.0 - (in_any_black_square.sum() / len(samples_2d))
    return contamination


def compute_checkerboard_f1(gen_samples: np.ndarray, real_samples: np.ndarray, 
                           threshold: float = 0.1) -> float:
    """
    Compute F1 score based on nearest neighbor distances.
    
    Precision: fraction of generated samples that have a real sample within threshold
    Recall: fraction of real samples that have a generated sample within threshold
    """
    from scipy.spatial import KDTree
    
    # Build KD trees for efficient nearest neighbor search
    tree_real = KDTree(real_samples)
    tree_gen = KDTree(gen_samples)
    
    # Precision: for each generated point, is there a real point nearby?
    distances_to_real, _ = tree_real.query(gen_samples, k=1)
    precision = np.mean(distances_to_real < threshold)
    
    # Recall: for each real point, is there a generated point nearby?
    distances_to_gen, _ = tree_gen.query(real_samples, k=1)
    recall = np.mean(distances_to_gen < threshold)
    
    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return f1


def create_boundary_aware_summary(metrics: Dict[str, float]) -> str:
    """
    Create a summary score that emphasizes boundary-aware performance.
    """
    # Combine metrics with weights that emphasize what you care about
    weights = {
        'precision_coverage': 0.3,        # Important: points in correct regions
        'mode_separation': 0.2,          # Important: well-separated modes
        'boundary_sharpness': 0.15,      # Sharp transitions
        'concentration_score': 0.15,     # Concentrated within modes
        'inter_mode_contamination': -0.2 # Penalty for points between modes
    }
    
    # Compute weighted score
    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            if metric == 'inter_mode_contamination':
                # This is a penalty (lower is better)
                score += weight * metrics[metric]
            else:
                score += weight * metrics[metric]
    
    return score


# Integration with your existing code:
def evaluate_with_boundary_metrics(x1_gen, x1_real, sphere_manifold, folder):
    """
    Add boundary-aware metrics to your evaluation.
    """
    # Unwrap samples
    x1_gen_2d = sphere_manifold.unwrap(x1_gen.detach().cpu())
    x1_real_2d = sphere_manifold.unwrap(x1_real.cpu())
    
    if isinstance(x1_gen_2d, torch.Tensor):
        x1_gen_2d = x1_gen_2d.numpy()
    if isinstance(x1_real_2d, torch.Tensor):
        x1_real_2d = x1_real_2d.numpy()
    
    # Compute boundary-aware metrics
    boundary_metrics = compute_boundary_aware_metrics(x1_gen_2d, x1_real_2d)
    
    # Compute summary score
    summary_score = create_boundary_aware_summary(boundary_metrics)
    boundary_metrics['boundary_aware_score'] = summary_score
    
    # Save to file
    with open(folder + 'boundary_metrics.txt', 'w') as f:
        f.write("BOUNDARY-AWARE METRICS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Precision Coverage: {boundary_metrics['precision_coverage']*100:.2f}%\n")
        f.write(f"Mode Separation: {boundary_metrics['mode_separation']:.3f}\n")
        f.write(f"Boundary Sharpness: {boundary_metrics['boundary_sharpness']:.3f}\n")
        f.write(f"Concentration Score: {boundary_metrics['concentration_score']:.3f}\n")
        f.write(f"Inter-mode Contamination: {boundary_metrics['inter_mode_contamination']*100:.2f}%\n")
        if 'checkerboard_f1' in boundary_metrics:
            f.write(f"Checkerboard F1: {boundary_metrics['checkerboard_f1']:.3f}\n")
        f.write(f"\nBoundary-Aware Score: {boundary_metrics['boundary_aware_score']:.3f}\n")
    
    return boundary_metrics



def evaluate_spherical_checkerboard(real_samples: torch.Tensor, 
                                  generated_samples: torch.Tensor,
                                  use_spherical_kernel: bool = True,
                                  bandwidths: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation for spherical checkerboard distribution.
    
    Args:
        real_samples: Real data samples (N x 3) on sphere
        generated_samples: Generated samples (M x 3) on sphere
        use_spherical_kernel: Whether to use spherical geodesic kernel
        bandwidths: List of bandwidths for multi-scale kernel
    
    Returns:
        Dictionary with various metrics
    """
    if bandwidths is None:
        # Default bandwidths chosen for checkerboard scale
        bandwidths = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
    
    # Compute MMD for each bandwidth separately
    mmd_per_bandwidth = {}
    for bw in bandwidths:
        if use_spherical_kernel:
            kernel_fn = lambda x, y: MMDMetrics.spherical_kernel(x, y, bandwidth=bw)
        else:
            kernel_fn = lambda x, y: MMDMetrics.gaussian_kernel(x, y, bandwidth=bw)
        
        # Compute MMD for this bandwidth
        mmd_squared = MMDMetrics.compute_mmd_squared(real_samples, generated_samples, kernel_fn).item()
        mmd_per_bandwidth[f'bw_{bw}'] = {
            'mmd_squared': mmd_squared,
            'mmd': np.sqrt(max(0, mmd_squared))
        }
    
    # Also compute with multi-scale kernel (average)
    if use_spherical_kernel:
        # Multi-scale spherical kernel
        def kernel_fn(x, y):
            kernels = []
            for bw in bandwidths:
                kernels.append(MMDMetrics.spherical_kernel(x, y, bandwidth=bw))
            return torch.stack(kernels).mean(dim=0)
    else:
        # Multi-scale Gaussian kernel in R^3
        def kernel_fn(x, y):
            return MMDMetrics.multi_scale_gaussian_kernel(x, y, bandwidths=bandwidths)
    
    # Compute MMD with confidence intervals
    mmd_results = MMDMetrics.compute_mmd_with_bootstrap(
        real_samples, generated_samples, kernel_fn, n_bootstrap=100
    )
    
    # Permutation test
    mmd_observed, p_value, significant = MMDMetrics.permutation_test(
        real_samples, generated_samples, kernel_fn, n_permutations=100
    )
    
    # For large datasets, also compute linear-time approximation
    if len(real_samples) > 5000 or len(generated_samples) > 5000:
        mmd_linear = MMDMetrics.linear_time_mmd(
            real_samples, generated_samples, kernel_fn, n_permutations=50
        )
    else:
        mmd_linear = mmd_results['mmd']
    
    results = {
        'mmd_squared': mmd_results['mmd'],
        'mmd': mmd_results['mmd_sqrt'],  # Square root for interpretability
        'mmd_lower_ci': mmd_results['lower_ci'],
        'mmd_upper_ci': mmd_results['upper_ci'],
        'mmd_std': mmd_results['std'],
        'mmd_linear_approx': mmd_linear,
        'p_value': p_value,
        'distributions_different': significant,
        'mmd_per_bandwidth': mmd_per_bandwidth,
        'bandwidths': bandwidths
    }
    
    return results

def compute_checkerboard_coverage(samples: torch.Tensor, 
                                 sphere_manifold) -> Dict[str, float]:
    """
    CORRECTED: Compute coverage metrics specific to checkerboard distribution.
    Now uses the correct BLACK squares where points should actually be.
    
    Args:
        samples: Generated samples on sphere
        sphere_manifold: SphereManifold instance for unwrapping
    
    Returns:
        Dictionary with coverage metrics
    """
    # Ensure samples is a tensor
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples)
    
    # Unwrap to 2D for easier analysis
    samples_2d = sphere_manifold.unwrap(samples)
    
    # Convert to numpy for easier processing
    if isinstance(samples_2d, torch.Tensor):
        samples_2d_np = samples_2d.numpy()
    else:
        samples_2d_np = samples_2d
    
    # Get correct black squares
    black_squares = get_checkerboard_black_squares_correct()
    
    samples_per_square = []
    total_in_squares = 0
    
    # Check each black square
    for x_min, x_max, y_min, y_max in black_squares:
        in_square = ((samples_2d_np[:, 0] >= x_min) & (samples_2d_np[:, 0] < x_max) & 
                    (samples_2d_np[:, 1] >= y_min) & (samples_2d_np[:, 1] < y_max))
        
        count = in_square.sum()
        samples_per_square.append(count)
        total_in_squares += count
    
    
    n_samples = len(samples_2d_np)
    coverage = total_in_squares / n_samples
    
    # Count how many squares have any points
    modes_covered = sum(1 for count in samples_per_square if count > 0)
    total_modes = len(black_squares)  # Should be 8 for 4x4 checkerboard
    
    # Compute uniformity (coefficient of variation of samples per mode)
    if modes_covered > 0:
        samples_in_covered_modes = [count for count in samples_per_square if count > 0]
        if len(samples_in_covered_modes) > 1:
            uniformity_cv = np.std(samples_in_covered_modes) / np.mean(samples_in_covered_modes)
        else:
            uniformity_cv = 0.0
    else:
        uniformity_cv = float('inf')
    
    return {
        'coverage': coverage,
        'modes_covered': modes_covered,
        'total_modes': total_modes,
        'uniformity_cv': uniformity_cv,
        'samples_per_mode': samples_per_square
    }


class MMDMetrics:
    """
    CORRECTED MMD computation methods with proper unbiased estimator.
    """
    
    @staticmethod
    def rbf_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
        """Standard RBF/Gaussian kernel."""
        return torch.exp(-torch.cdist(x, y)**2 / (2 * bandwidth**2))
    
    @staticmethod
    def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
        """Alias for RBF kernel."""
        return MMDMetrics.rbf_kernel(x, y, bandwidth)
    
    @staticmethod
    def multi_scale_gaussian_kernel(x: torch.Tensor, y: torch.Tensor, 
                                   bandwidths: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]) -> torch.Tensor:
        """Multi-scale Gaussian kernel - average of multiple bandwidths."""
        kernels = []
        for bandwidth in bandwidths:
            dist = torch.cdist(x, y, p=2.0)
            kernels.append(torch.exp(-dist**2 / (2 * bandwidth**2)))
        return torch.stack(kernels).mean(dim=0)
    
    @staticmethod
    def spherical_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
        """Spherical geodesic kernel for points on unit sphere."""
        # Ensure inputs are normalized
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        y = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)
        
        # Compute great circle distances
        dot_products = torch.mm(x, y.t()).clamp(-1, 1)
        geodesic_distances = torch.acos(dot_products)
        return torch.exp(-geodesic_distances**2 / (2 * bandwidth**2))
    
    @staticmethod
    def compute_mmd_squared(x: torch.Tensor, y: torch.Tensor, 
                           kernel_fn, unbiased: bool = True) -> torch.Tensor:
        """
        CORRECTED: Compute MMD² between two sets of samples using unbiased estimator.
        
        MMD²(P, Q) = E[k(X, X')] + E[k(Y, Y')] - 2E[k(X, Y)]
        where X, X' ~ P and Y, Y' ~ Q
        
        This computes MMD between real samples (x) and generated samples (y).
        """
        # Ensure tensors are float and on same device
        x = x.float()
        y = y.float()
        
        # Compute kernel matrices
        K_xx = kernel_fn(x, x)  # Real vs Real
        K_yy = kernel_fn(y, y)  # Generated vs Generated  
        K_xy = kernel_fn(x, y)  # Real vs Generated
        
        # MMD² computation
        n, m = x.shape[0], y.shape[0]
        
        if unbiased:
            # Remove diagonal from K_xx and K_yy to get unbiased estimator
            K_xx_sum = K_xx.sum() - K_xx.diag().sum()  # Remove diagonal
            K_yy_sum = K_yy.sum() - K_yy.diag().sum()  # Remove diagonal
            K_xy_sum = K_xy.sum()
            
            mmd_squared = (K_xx_sum / (n * (n - 1)) + 
                          K_yy_sum / (m * (m - 1)) - 
                          2 * K_xy_sum / (n * m))
        else:
            # Biased estimator
            mmd_squared = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        
        return mmd_squared.clamp(min=0)  # Ensure non-negative due to numerical errors
    
    @staticmethod
    def compute_mmd_with_bootstrap(x: torch.Tensor, y: torch.Tensor, 
                                  kernel_fn, 
                                  n_bootstrap: int = 100,
                                  confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Compute MMD with bootstrap confidence intervals.
        
        Returns:
            Dictionary with 'mmd', 'lower_ci', 'upper_ci'
        """
        mmd_values = []
        n, m = x.shape[0], y.shape[0]
        
        for _ in range(n_bootstrap):
            # Bootstrap sampling
            idx_x = torch.randint(0, n, (n,))
            idx_y = torch.randint(0, m, (m,))
            x_boot = x[idx_x]
            y_boot = y[idx_y]
            
            mmd = MMDMetrics.compute_mmd_squared(x_boot, y_boot, kernel_fn)
            mmd_values.append(mmd.item())
        
        mmd_values = np.array(mmd_values)
        alpha = 1 - confidence_level
        lower = np.percentile(mmd_values, 100 * alpha/2)
        upper = np.percentile(mmd_values, 100 * (1 - alpha/2))
        
        # Compute MMD on original data
        mmd_original = MMDMetrics.compute_mmd_squared(x, y, kernel_fn).item()
        
        return {
            'mmd': mmd_original,
            'mmd_sqrt': np.sqrt(max(0, mmd_original)),  # Square root for interpretability
            'lower_ci': lower,
            'upper_ci': upper,
            'std': np.std(mmd_values)
        }
    
    @staticmethod
    def linear_time_mmd(x: torch.Tensor, y: torch.Tensor, 
                       kernel_fn, 
                       n_permutations: int = 100) -> float:
        """
        Linear-time MMD approximation for large datasets.
        """
        n = min(len(x), len(y))
        mmd_values = []
        
        for _ in range(n_permutations):
            # Random permutation
            perm_x = torch.randperm(len(x))[:n]
            perm_y = torch.randperm(len(y))[:n]
            
            x_perm = x[perm_x]
            y_perm = y[perm_y]
            
            # Linear-time estimate (paired samples)
            value = 0.0
            for i in range(0, n-1, 2):
                value += kernel_fn(x_perm[i:i+1], x_perm[i+1:i+2]).item()
                value += kernel_fn(y_perm[i:i+1], y_perm[i+1:i+2]).item()
                value -= kernel_fn(x_perm[i:i+1], y_perm[i:i+1]).item()
                value -= kernel_fn(x_perm[i+1:i+2], y_perm[i+1:i+2]).item()
            
            mmd_values.append(value / (n // 2))
        
        return np.mean(mmd_values)
    
    @staticmethod
    def permutation_test(x: torch.Tensor, y: torch.Tensor, 
                        kernel_fn,
                        n_permutations: int = 1000,
                        alpha: float = 0.05) -> Tuple[float, float, bool]:
        """
        Permutation test for MMD to test if distributions are significantly different.
        
        Returns:
            (mmd_observed, p_value, reject_null)
        """
        # Observed MMD
        mmd_observed = MMDMetrics.compute_mmd_squared(x, y, kernel_fn).item()
        
        # Combine datasets
        combined = torch.cat([x, y], dim=0)
        n, m = len(x), len(y)
        
        # Null distribution
        null_distribution = []
        for _ in range(n_permutations):
            perm = torch.randperm(n + m)
            x_perm = combined[perm[:n]]
            y_perm = combined[perm[n:]]
            mmd_null = MMDMetrics.compute_mmd_squared(x_perm, y_perm, kernel_fn).item()
            null_distribution.append(mmd_null)
        
        # p-value
        p_value = np.mean(np.array(null_distribution) >= mmd_observed)
        reject_null = p_value < alpha
        
        return mmd_observed, p_value, reject_null


class C2STClassifier(nn.Module):
    """Simple MLP classifier for C2ST metric."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def compute_c2st_score(real_samples: torch.Tensor, 
                       generated_samples: torch.Tensor,
                       use_2d_unwrapped: bool = True,
                       sphere_manifold = None,
                       hidden_dim: int = 64,
                       num_epochs: int = 50,
                       batch_size: int = 256,
                       lr: float = 1e-3,
                       num_runs: int = 3,
                       device: str = None) -> Dict[str, float]:
    """
    Compute Classifier 2-Sample Test (C2ST) score.
    
    Args:
        real_samples: Real data samples 
        generated_samples: Generated samples
        use_2d_unwrapped: Whether to use 2D unwrapped coordinates (vs 3D sphere coords)
        sphere_manifold: SphereManifold instance for unwrapping (required if use_2d_unwrapped=True)
        hidden_dim: Hidden dimension of classifier
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        num_runs: Number of classifier training runs (for averaging)
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Dictionary with C2ST metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure tensors are on CPU for data preprocessing
    real_samples = real_samples.detach().cpu()
    generated_samples = generated_samples.detach().cpu()
    
    # Prepare data
    if use_2d_unwrapped:
        if sphere_manifold is None:
            raise ValueError("sphere_manifold is required when use_2d_unwrapped=True")
        
        # Unwrap to 2D coordinates
        real_data = sphere_manifold.unwrap(real_samples)
        gen_data = sphere_manifold.unwrap(generated_samples)
        
        # Convert to tensors if numpy arrays
        if isinstance(real_data, np.ndarray):
            real_data = torch.from_numpy(real_data).float()
        if isinstance(gen_data, np.ndarray):
            gen_data = torch.from_numpy(gen_data).float()
        
        input_dim = 2
    else:
        # Use 3D sphere coordinates
        real_data = real_samples.float()
        gen_data = generated_samples.float()
        input_dim = 3
    
    # Combine data and create labels
    X = torch.cat([real_data, gen_data], dim=0)
    y = torch.cat([
        torch.zeros(len(real_data), 1),  # Real = 0
        torch.ones(len(gen_data), 1)     # Generated = 1
    ], dim=0)
    
    # Multiple runs for robustness
    accuracies = []
    
    for run in range(num_runs):
        # Create dataset and shuffle
        dataset = torch.utils.data.TensorDataset(X, y)
        
        # Split into train/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Initialize classifier
        classifier = C2STClassifier(input_dim, hidden_dim).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Training
        classifier.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = classifier(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        # Evaluation
        classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = classifier(batch_x)
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        accuracies.append(accuracy)
    
    # Compute statistics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    return {
        'c2st_accuracy': mean_accuracy,
        'c2st_std': std_accuracy,
        'c2st_runs': accuracies,
        'c2st_score': mean_accuracy  # For consistency with other metrics
    }