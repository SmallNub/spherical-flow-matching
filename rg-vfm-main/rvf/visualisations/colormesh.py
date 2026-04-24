import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def pcolormesh_3d(ax, X, Y, Z, C, cmap='gray', alpha=0.3):
    """
    Plot a colored mesh on a 3D surface.

    Args:
        ax: matplotlib 3D axis
        X, Y, Z: 2D arrays of coordinates (shape [n, m])
        C: 2D array of color values (shape [n-1, m-1])
        cmap: colormap name
        alpha: transparency
    """
    cmap = plt.get_cmap(cmap)

    # Normalize C to [0, 1] for colormap
    C_normalized = (C - C.min()) / (C.max() - C.min() + 1e-10)

    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1] - 1):
            # Get the four corners of each cell
            verts = [
                [X[i, j], Y[i, j], Z[i, j]],
                [X[i+1, j], Y[i+1, j], Z[i+1, j]],
                [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]],
                [X[i, j+1], Y[i, j+1], Z[i, j+1]]
            ]

            # Get color from colormap
            color = cmap(C_normalized[i, j])

            # Create polygon and add to axis
            poly = Poly3DCollection([verts], alpha=alpha)
            poly.set_facecolor(color)
            poly.set_edgecolor('none')
            ax.add_collection3d(poly)
