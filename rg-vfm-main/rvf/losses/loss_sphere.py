import torch
from rvf.losses.loss_euclidean import VanillaLoss, VariationalLoss
from rvf.manifolds.sphere import SphereManifold


class VanillaLossSphere(VanillaLoss):
    """Losses for (vanilla) flow matching on sphere.

    Args:
        model: model to be trained
        x0: initial point
        x1: target point
        t: time in [0,1]
        noise_scale: noise scale
        eps: epsilon

    Returns:
        loss: loss value (mse between true and predicted velocity)
    """
    def __init__(self, noise_scale, eps=1e-7):
        super().__init__(noise_scale, eps)

    def loss_intrinsic(self, model, x0, x1, t):
        """
        Compute the MSE between the true geodesic velocity and the predicted velocity.
        The velocity is projected onto the tangent space of the sphere at x0.
        """
        x_pred = SphereManifold().geodesic(x0, x1, t)
        if self.noise_scale > 0:
            x_pred = x_pred + torch.randn_like(x_pred) * self.noise_scale
            x_pred = x_pred / (x_pred.norm(dim=1, keepdim=True) + self.eps)
        v_true = SphereManifold().geodesic_velocity(x0, x1, t)
        v_pred = model(t, x_pred)
        return (v_true - v_pred).pow(2).mean()

    def loss_extrinsic(self, model, x0, x1, t):
        """
        Compute the MSE between the true geodesic velocity and the predicted velocity.
        The velocity is computed in the Euclidean space, and is not constrained to the sphere.
        """
        x_pred = SphereManifold().geodesic(x0, x1, t)
        if self.noise_scale > 0:
            x_pred = x_pred + torch.randn_like(x_pred) * self.noise_scale
        v_true = SphereManifold().geodesic_velocity(x0, x1, t)
        v_pred = model(t, x_pred)
        return (v_true - v_pred).pow(2).mean()


class VariationalLossSphere(VariationalLoss):
    """Losses for variational flow matching on sphere."""

    def __init__(self, noise_scale, eps=1e-7):
        super().__init__(noise_scale, eps)

    def loss_intrinsic(self, model, x0, x1, t):
        """
        Compute the MSE between the true position and the predicted position.
        The position is projected onto the tangent space of the sphere at x0.
        """
        v_0 = SphereManifold().log_map(x0, x1)
        x_t = SphereManifold().exp_map(x0, t * v_0)
        if self.noise_scale > 0:
            x_t = x_t + torch.randn_like(x_t) * self.noise_scale
            x_t = x_t / (x_t.norm(dim=1, keepdim=True) + self.eps)
        mu_t = model(t, x_t)
        distance = SphereManifold().distance(mu_t, x1)
        return distance.pow(2).mean()

    def loss_extrinsic(self, model, x0, x1, t):
        """
        Compute the MSE between the true position and the predicted position.
        The position is computed in the Euclidean space, and is not constrained to the sphere.
        """
        x_t = x0 + t * (x1 - x0)
        if self.noise_scale > 0:
            x_t = x_t + torch.randn_like(x_t) * self.noise_scale
        mu_t = model(t, x_t)
        distance = SphereManifold().distance(mu_t, x1)
        return distance.pow(2).mean()
