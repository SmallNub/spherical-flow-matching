import numpy as np
import torch

from rvf.visualisations.base_mesh import create_checkerboard_pattern
from rvf.manifolds.hyperboloid import HyperboloidManifold
from rvf.visualisations.colormesh import pcolormesh_3d
from rvf.utils.utils import torch2npy


def mesh_upper_sheet(rho_dim=100, phi_dim=100, max_rho=1.5):
    """
    Create a mesh array for the upper sheet of the hyperboloid H^2:
    returns an array of shape (rho_dim, phi_dim, 3) with columns [x1, x2, x0]
    Parametrized by (rho, phi):
      x0 = cosh(rho), x1 = sinh(rho)*cos(phi), x2 = sinh(rho)*sin(phi)
    """
    rhos = np.linspace(0, max_rho, rho_dim)
    phis = np.linspace(0, 2 * np.pi, phi_dim)
    mesh = np.zeros((rho_dim, phi_dim, 3))
    mesh[:, :, 1] = np.outer(np.sinh(rhos), np.cos(phis))
    mesh[:, :, 2] = np.outer(np.sinh(rhos), np.sin(phis))
    mesh[:, :, 0] = np.outer(np.cosh(rhos), np.ones_like(phis))
    return mesh


def mesh_hyperboloid(rho_dim=100, phi_dim=100, max_rho=1.5):
    """
    Alias of mesh_upper_sheet: returns three 2D arrays X, Y, Z corresponding to x1, x2, x0 slices of the upper sheet mesh.
    """
    mesh = mesh_upper_sheet(rho_dim, phi_dim, max_rho)
    return mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2]


def data_on_hyperboloid(points_3d, ax, color="r", rho_dim=100, phi_dim=100, max_rho=2.0):
    """
    Plot points on the hyperboloid surface together with a mesh.

    Args:
        points_3d: torch tensor of shape (N, 3) in Minkowski coords (x0, x1, x2)
        ax: matplotlib 3D axis
    """
    pts = torch2npy(points_3d)
    X_mesh, Y_mesh, Z_mesh = mesh_hyperboloid(rho_dim, phi_dim, max_rho)
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, color="c", alpha=0.2, rstride=5, cstride=5)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], color=color, s=1, alpha=0.3)
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("X2")
    return ax


def mesh_checkerboard_on_hyperboloid(ax, grid_size=51, num_squares=4, max_rho=2.0):
    """
    Map a 2D checkerboard pattern onto the hyperboloid via exp map at origin.
    """
    U, V, checker = create_checkerboard_pattern(num_squares, grid_size)
    pts2d = np.stack([U.flatten(), V.flatten()], axis=1)
    pts2d_t = torch.from_numpy(pts2d).float()
    hyper = HyperboloidManifold()
    pts3d = hyper.wrap(pts2d_t)
    pts3d = pts3d.reshape(grid_size, grid_size, 3).cpu().numpy()

    X = pts3d[:,:,0]
    Y = pts3d[:,:,1]
    Z = pts3d[:,:,2]

    ax.set_xlim([0.5, 2.5])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_box_aspect([0.7,1,1])

    ax.view_init(elev=-60, azim=0)
    ax.axis("off")
    pcolormesh_3d(ax, X, Y, Z, checker[:-1, :-1], cmap='gray', alpha=0.3)
    return ax


def data_on_hyperboloid_top(points_3d, ax, color="r", rho_dim=100, phi_dim=100, max_rho=2.0):
    """
    Plot points on the hyperboloid surface together with a mesh (top view).

    Args:
        points_3d: torch tensor of shape (N, 3) in Minkowski coords (x0, x1, x2)
        ax: matplotlib 3D axis
    """
    pts = torch2npy(points_3d)
    X_mesh, Y_mesh, Z_mesh = mesh_hyperboloid(rho_dim, phi_dim, max_rho)
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, color="c", alpha=0.2, rstride=5, cstride=5)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], color=color, s=1, alpha=0.3)
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("X2")
    return ax


def mesh_checkerboard_on_hyperboloid_top(ax, grid_size=51, num_squares=4, max_rho=2.0):
    """
    Map a 2D checkerboard pattern onto the hyperboloid via exp map at origin (top view).
    """
    U, V, checker = create_checkerboard_pattern(num_squares, grid_size)
    pts2d = np.stack([U.flatten(), V.flatten()], axis=1)
    pts2d_t = torch.from_numpy(pts2d).float()
    hyper = HyperboloidManifold()
    pts3d = hyper.wrap(pts2d_t)
    pts3d = pts3d.reshape(grid_size, grid_size, 3).cpu().numpy()

    X = pts3d[:,:,0]
    Y = pts3d[:,:,1]
    Z = pts3d[:,:,2]

    ax.set_xlim([0.5, 2.5])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_box_aspect([0.7,1,1])

    ax.view_init(elev=-0, azim=0, roll=0)
    ax.axis("off")
    pcolormesh_3d(ax, X, Y, Z, checker[:-1, :-1], cmap='gray', alpha=0.3)
    return ax
