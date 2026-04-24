import yaml
import torch
from rvf.manifolds.sphere import SphereManifold


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def load_model(model, folder):
    model.load_state_dict(torch.load(folder + 'model.pth'))
    return model


def save_model(model, folder):
    torch.save(model.state_dict(), folder + 'model.pth')


def initialize_p0(p0_distribution, p0_support, num_samples):
    if p0_support == "sphere":
        dim = 2
    else:
        dim = 3

    if p0_distribution == "uniform":
        x0 = 2*torch.rand(num_samples, dim) - 1
    elif p0_distribution == "gaussian":
        x0 = torch.randn(num_samples, dim)

    if p0_support == "sphere":
        x0 = SphereManifold().wrap(x0)
    return x0


def torch2npy_3d(points_3d):
    if isinstance(points_3d, torch.Tensor):
        points_3d = points_3d.cpu().detach().numpy()
    assert points_3d.shape[1] == 3, "Expecting 3D points on the sphere."
    return points_3d


def torch2npy_2d(points_2d):
    if isinstance(points_2d, torch.Tensor):
        points_2d = points_2d.cpu().detach().numpy()
    assert points_2d.shape[1] == 2, "Expecting 2D points."
    return points_2d


def torch2npy(points):
    if isinstance(points, torch.Tensor):
        points = points.cpu().detach().numpy()
    return points
