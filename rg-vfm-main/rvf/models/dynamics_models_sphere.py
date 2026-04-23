import torch
import torch.nn as nn
from rvf.models.dynamics_models_euclidean import VectorDynamics, PositionDynamics
from rvf.manifolds.sphere import SphereManifold


class VectorDynamicsSphere(VectorDynamics):
    """
    A subclass of VectorDynamics that ensures the output is
    in the tangent space of S^2.
    """
    def __init__(self,
        time_dim: int = 1,
        input_dim: int = 3,
        hidden_dim: int = 128):
        super().__init__(time_dim, input_dim, hidden_dim)

    def forward(self, t, x):
        """
        x: (batch_size, 3) on S^2
        t: (batch_size, 1) in [0,1]
        return: (batch_size, 3) tangent vector
        """
        assert x.shape[-1] == 3, f"Expected x to have shape (batch_size, 3), got {x.shape}"
        assert t.shape[-1] == 1, f"Expected t to have shape (batch_size, 1), got {t.shape}"
        x = x / (x.norm(dim=1, keepdim=True) + 1e-7)
        inp = torch.cat([t, x], dim=-1)
        inp = self.input_layer(inp)
        return SphereManifold().project_to_tangent(x, self.net(inp))


class PositionDynamicsSphere(PositionDynamics):
    """
    A subclass of PositionDynamics that ensures the output is
    in the tangent space of S^2.
    """
    def __init__(self,
        time_dim: int = 1,
        input_dim: int = 3,
        hidden_dim: int = 128):
        super().__init__(time_dim, input_dim, hidden_dim)

    def forward(self, t, x):
        """
        x: (batch_size, 3) on S^2
        t: (batch_size, 1) in [0,1]
        return: (batch_size, 3) tangent vector
        """
        assert x.shape[-1] == 3, f"Expected x to have shape (batch_size, 3), got {x.shape}"
        assert t.shape[-1] == 1, f"Expected t to have shape (batch_size, 1), got {t.shape}"
        mu_t = super().forward(t, x)
        return mu_t / torch.norm(mu_t, dim=-1, keepdim=True)
