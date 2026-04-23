import torch
import torch.utils.data as data
from rvf.manifolds.sphere import SphereManifold
from rvf.manifolds.hyperboloid import HyperboloidManifold
import matplotlib.pyplot as plt


def checkboard_2d(batch_size: int = 200, device: str = "cpu", num_square: int = 4):
    """
    Returns a batch of points in R^2 forming a checkerboard distribution.
    Points are generated only within black squares of the checkerboard.

    Args:
        batch_size: Number of points to generate.
        device: Device to generate the points on.
        num_square: Number of squares in the checkerboard pattern

    Returns:
        data: A tensor of shape (batch_size, 2) containing the checkerboard points.
    """
    # Generate random points in [0, num_square]²
    x = torch.rand(batch_size, 2, device=device) * num_square
    
    # Get cell coordinates and determine if they're in white squares
    cell_x = torch.floor(x[:, 0]).to(torch.int)
    cell_y = torch.floor(x[:, 1]).to(torch.int)
    is_white = ((cell_x + cell_y) % 2 == 0)
    
    # For points in white squares, shift them to the nearest black square
    # First try shifting right
    can_shift_right = (cell_x < num_square - 1)
    x[:, 0] = x[:, 0] + (is_white & can_shift_right).float()
    
    # If can't shift right, try shifting down
    can_shift_down = (cell_y < num_square - 1)
    x[:, 1] = x[:, 1] + (is_white & ~can_shift_right & can_shift_down).float()
    
    # If can't shift right or down, shift to first black square in first column
    # First column black squares are at even y coordinates
    is_corner = is_white & ~can_shift_right & ~can_shift_down
    x[:, 0] = x[:,0] - (num_square-1) * is_corner.float()
    
    # Normalize to [-1, 1]
    return (x / num_square * 2 - 1).float()



class CheckerboardEuclidean(data.Dataset):
    """
    Checkerboard distribution in R^2.
    """
    def __init__(self, dataset_size, device: str = "cpu"):
        super().__init__()
        self.dataset_size = dataset_size
        self.data = checkboard_2d(dataset_size, device)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx]
    

class CheckerboardSphere(CheckerboardEuclidean):
    """
    Checkerboard distribution on the sphere in R^3.
    """
    def __init__(self, dataset_size, device: str = "cpu"):
        super().__init__(dataset_size, device)  
        self.data = SphereManifold().wrap(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CheckerboardHyperbolic(CheckerboardEuclidean):
    """
    Checkerboard distribution on the hyperbolic plane in R^2.
    # symmetric in X, Y. Central axis in Z.
    """
    def __init__(self, dataset_size, device: str = "cpu"):
        super().__init__(dataset_size, device)
        self.data = HyperboloidManifold().wrap(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def plot_checkerboard_euclidean(num_points: int = 1000):
    from rvf.visualisations.sphere_mesh import create_checkerboard_pattern
    dataset = CheckerboardEuclidean(num_points)
    dataloader = data.DataLoader(dataset, batch_size=num_points, shuffle=True)
    X, Y, checker = create_checkerboard_pattern(num_squares=4, grid_size=100)
    for batch in dataloader:
        x1 = batch
        break
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.scatter(x1[:, 0], x1[:, 1], s=1, color="r", alpha=0.5)
    ax.pcolormesh(X, Y, checker, cmap="gray", alpha=0.5)
    plt.show()

def plot_checkerboard_sphere(num_points: int = 1000):
    from rvf.visualisations.sphere_mesh import mesh_checkerboard_on_sphere
    dataset = CheckerboardSphere(num_points)
    dataloader = data.DataLoader(dataset, batch_size=num_points, shuffle=True)
    for batch in dataloader:
        x1 = batch
        break
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], s=1, color="r", alpha=0.5)
    ax = mesh_checkerboard_on_sphere(ax)
    plt.show()

def plot_checkerboard_hyperbolic(num_points: int = 1000):
    from rvf.visualisations.hyperbolic_mesh import mesh_checkerboard_on_hyperboloid
    dataset = CheckerboardHyperbolic(num_points)
    dataloader = data.DataLoader(dataset, batch_size=num_points, shuffle=True)
    for batch in dataloader:
        x1 = batch
        break
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], s=1, color="r", alpha=0.5)
    ax = mesh_checkerboard_on_hyperboloid(ax)
    plt.show()

if __name__ == "__main__":
    num_points = 5000
    # plot_checkerboard_euclidean(num_points)
    plot_checkerboard_sphere(num_points)
    # plot_checkerboard_hyperbolic(num_points)
