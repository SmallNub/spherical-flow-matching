# Description: Wrappers defined to be used in pair with the odeint solver
from rvf.manifolds.sphere import SphereManifold
from rvf.solver.wrappers_euclidean import VelocityInference
import torch


class VelocityInferenceSphere(VelocityInference):
    """
    A wrapper that takes a model and compute the velocity at a point x(t) on the sphere.
    To be used with variational Flow Matching when p0 is on the sphere
    """
    def __init__(self, model):
        super().__init__(model)
        self.manifold = SphereManifold()

    def forward(self, t, x):
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        scale = torch.clip(torch.ones_like(t) / (1-t), 0, 10)
        velocity = self.manifold.log_map(x, self.model(t, x)) * scale
        return velocity

