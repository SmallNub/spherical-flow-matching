import numpy as np
from rvf.utils.utils import torch2npy


def add_x1_background(ax, x1, alpha=0.1):
    """
    Add x1 as points in background
    """
    if x1.shape[1] == 3:
        x1 = torch2npy(x1)
        ax.scatter(x1[:,0], x1[:,1], x1[:,2], color="grey", s=5, alpha=alpha, zorder=1)
    else:
        x1 = torch2npy(x1)
        ax.scatter(x1[:,0], x1[:,1], color="grey", s=5, alpha=alpha, zorder=1)
    return ax


def create_checkerboard_pattern(num_squares=4, grid_size=100):
    """
    Create a checkerboard pattern with the specified number of squares.

    Args:
        num_squares: Number of checkerboard squares in one dimension (e.g., 4 for a 4x4 checkerboard)
        grid_size: Number of grid points per dimension for the visualization (higher = smoother)

    Returns:
        X, Y: Meshgrid coordinates normalized to [-1,1]
        checker: Binary pattern (0 for white, 1 for black)
    """
    eps = 1e-6
    x = np.linspace(-1+eps, 1-eps, grid_size)
    y = np.linspace(-1+eps, 1-eps, grid_size)
    X, Y = np.meshgrid(x, y)

    # Scale coordinates to have num_squares in [-1, 1]
    X_scaled = X * (num_squares / 2)
    Y_scaled = Y * (num_squares / 2)

    # Determine row and column indices
    column_index = np.floor(X_scaled + num_squares/2).astype(int)
    row_index = np.floor(Y_scaled + num_squares/2).astype(int)

    # Create checkerboard: squares are colored when row+column is even
    checker = (column_index + row_index) % 2 == 0
    checker = checker.astype(float)

    return X, Y, checker
