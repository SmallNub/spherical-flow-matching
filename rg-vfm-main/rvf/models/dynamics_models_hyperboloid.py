import torch
import torch.nn as nn
from rvf.models.dynamics_models_euclidean import VectorDynamics, PositionDynamics
from rvf.manifolds.hyperboloid import HyperboloidManifold


class VectorDynamicsHyperboloid(VectorDynamics):
    """
    A subclass of VectorDynamics that ensures the output is
    in the tangent space of the hyperboloid model H^n.
    """
    def __init__(self,
                 time_dim: int = 1,
                 input_dim: int = 3,
                 hidden_dim: int = 128):
        super().__init__(time_dim, input_dim, hidden_dim)

    def forward(self, t, x):
        """
        x: (batch_size, n+1) on H^n
        t: (batch_size, 1)
        return: (batch_size, n+1) tangent vector at x
        """
        assert x.shape[-1] == self.input_dim, f"Expected x shape (...,{self.input_dim}), got {x.shape}"
        assert t.shape[-1] == self.time_dim, f"Expected t shape (...,{self.time_dim}), got {t.shape}"

        norm_xx = HyperboloidManifold()._minkowski_dot(x, x)
        tol = 1e-3
        assert torch.allclose(norm_xx, torch.tensor(-1.0, device=norm_xx.device), atol=tol), (
            f"Points do not lie on the hyperboloid: max deviation = "
            f"{(norm_xx + 1.0).abs().max().item():.2e}"
        )

        inp = torch.cat([t, x], dim=-1)
        inp = self.input_layer(inp)
        v = self.net(inp)
        return HyperboloidManifold().project_to_tangent(x, v)


class PositionDynamicsHyperboloid(PositionDynamics):
    """
    A subclass of PositionDynamics that ensures the output is
    on the hyperboloid H^n via Minkowski retraction.
    """
    def __init__(self,
                 time_dim: int = 1,
                 input_dim: int = 3,
                 hidden_dim: int = 128):
        super().__init__(time_dim, input_dim, hidden_dim)

    def forward(self, t, x):
        """
        x: (batch_size, n+1) on H^n
        t: (batch_size, 1)
        return: (batch_size, n+1) point on H^n
        """
        assert x.shape[-1] == self.input_dim, f"Expected x shape (...,{self.input_dim}), got {x.shape}"
        assert t.shape[-1] == 1, f"Expected t shape (...,{self.time_dim}), got {t.shape}"
        mu_t = super().forward(t, x)
        return HyperboloidManifold().project_to_hyperboloid(mu_t)
