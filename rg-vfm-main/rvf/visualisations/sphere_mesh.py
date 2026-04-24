import numpy as np
import torch
from rvf.visualisations.base_mesh import create_checkerboard_pattern
from rvf.manifolds.sphere import SphereManifold
from rvf.visualisations.colormesh import pcolormesh_3d
from rvf.utils.utils import torch2npy


def mesh_sphere(sdim=100):
    """
    Create a meshgrid on the sphere S^2.
    """
    uu = np.linspace(0, 2 * np.pi, sdim)
    vv = np.linspace(0, np.pi, sdim)

    mesh = np.zeros((sdim, sdim, 3))
    mesh[:, :, 0] = np.outer(np.cos(uu), np.sin(vv))
    mesh[:, :, 1] = np.outer(np.sin(uu), np.sin(vv))
    mesh[:, :, 2] = np.outer(np.ones(np.size(uu)), np.cos(vv))
    return mesh


def data_on_sphere(points_3d, ax, color="r"):
    """
    Plot points on the surface of a sphere, with the sphere mesh and the points.
    """
    points_3d = torch2npy(points_3d)
    smesh = mesh_sphere(sdim=100)

    r = 1.0 - 1e-2
    ax.plot_surface(r*smesh[:,:,0], r*smesh[:,:,1], r*smesh[:,:,2], color="c", alpha=0.2, rstride=5, cstride=5)
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], color=color, s=1, alpha=0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.view_init(elev=90, azim=0)
    ax.axis("off")
    return ax


def mesh_checkerboard_on_sphere(ax, grid_size=51, num_squares=4):
    """
    Create a checkerboard pattern and map it onto the sphere using SphereManifold().wrap.

    Args:
        grid_size: Resolution of the grid (small offset if even)
        num_squares: Number of squares in the checkerboard pattern

    Returns:
        ax: Updated matplotlib axis with the checkerboard pattern on the sphere
    """
    U, V, checker = create_checkerboard_pattern(num_squares, grid_size)

    points_2d = np.stack([U.flatten(), V.flatten()], axis=1)
    points_2d = torch.from_numpy(points_2d).float()

    sphere = SphereManifold()
    points_3d = sphere.wrap(points_2d)
    points_3d = points_3d.reshape(grid_size, grid_size, 3)

    X = points_3d[:,:,0].numpy()
    Y = points_3d[:,:,1].numpy()
    Z = points_3d[:,:,2].numpy()

    pattern = checker.copy()
    pattern[Z < 0] = 0

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.view_init(elev=90, azim=270)
    ax.axis("off")

    pcolormesh_3d(ax, X, Y, Z, pattern[:-1, :-1], cmap='gray', alpha=0.2)

    return ax
