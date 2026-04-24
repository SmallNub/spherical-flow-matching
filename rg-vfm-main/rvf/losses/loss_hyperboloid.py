import torch
from rvf.losses.loss_euclidean import VanillaLoss, VariationalLoss
from rvf.manifolds.hyperboloid import HyperboloidManifold


class VanillaLossHyperboloid(VanillaLoss):
    """Losses for (vanilla) flow matching on hyperboloid.

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
        The velocity is projected onto the tangent space of the hyperboloid at x0.
        """
        x_pred = HyperboloidManifold().geodesic(x0, x1, t)
        v_true = HyperboloidManifold().log_map(x_pred, x1) / (1 - t + self.eps)
        v_true = HyperboloidManifold().project_to_tangent(x_pred, v_true)
        if self.noise_scale > 0:
            x_pred = HyperboloidManifold().unwrap(x_pred)
            x_pred = x_pred + torch.randn_like(x_pred) * self.noise_scale
            x_pred = HyperboloidManifold().wrap(x_pred)
        v_pred = model(t, x_pred)
        return (v_true - v_pred).pow(2).mean()

    def loss_extrinsic(self, model, x0, x1, t):
        """
        Compute the MSE between the true geodesic velocity and the predicted velocity.
        The velocity is computed in the ambient space, and is not constrained to the hyperboloid.
        """
        x_pred = HyperboloidManifold().geodesic(x0, x1, t)
        if self.noise_scale > 0:
            x_pred = x_pred + torch.randn_like(x_pred) * self.noise_scale
        v_true = HyperboloidManifold().geodesic_velocity(x0, x1, t)
        v_pred = model(t, x_pred)
        return (v_true - v_pred).pow(2).mean()


class VariationalLossHyperboloid(VariationalLoss):
    """Losses for variational flow matching on hyperboloid."""

    def __init__(self, noise_scale, eps=1e-7):
        super().__init__(noise_scale, eps)

    def loss_intrinsic(self, model, x0, x1, t):
        """
        Compute the MSE between the true position and the predicted position.
        The position is projected onto the tangent space of the hyperboloid at x0.
        """
        v_0 = HyperboloidManifold().log_map(x0, x1)
        x_t = HyperboloidManifold().exp_map(x0, t * v_0)
        if self.noise_scale > 0:
            x_t = HyperboloidManifold().unwrap(x_t)
            x_t = x_t + torch.randn_like(x_t) * self.noise_scale
            x_t = HyperboloidManifold().wrap(x_t)
        mu_t = model(t, x_t)
        distance = HyperboloidManifold().distance(mu_t, x1)
        return distance.pow(2).mean()

    def loss_extrinsic(self, model, x0, x1, t):
        """
        Compute the MSE between the true position and the predicted position.
        The position is computed in the Euclidean space, and is not constrained to the hyperboloid.
        """
        x_t = x0 + t * (x1 - x0)
        if self.noise_scale > 0:
            x_t = x_t + torch.randn_like(x_t) * self.noise_scale
        mu_t = model(t, x_t)
        distance = HyperboloidManifold().distance(mu_t, x1)
        return distance.pow(2).mean()
