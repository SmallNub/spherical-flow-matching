import torch


class VanillaLoss:
    """Losses for (vanilla) flow matching.

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
        self.noise_scale = noise_scale
        self.eps = eps

    def loss_euclidean(self, model, x0, x1, t):
        """Compare model's velocity with true geodesic velocity in Euclidean space."""
        x_pred = x0 + t * (x1 - x0)  # linear interpolation
        v_true = (x1 - x_pred) / (1 - t + self.eps)
        if self.noise_scale > 0:
            x_pred = x_pred + torch.randn_like(x_pred) * self.noise_scale
        v_pred = model(t, x_pred)
        return (v_true - v_pred).pow(2).mean()


class VariationalLoss:
    """Losses for variational flow matching.

    Args:
        model: model to be trained
        x0: initial point
        x1: target point
        t: time in [0,1]
        noise_scale: noise scale
        eps: epsilon

    Returns:
        loss: loss value (mse between true and predicted positions)
    """
    def __init__(self, noise_scale, eps=1e-7):
        self.noise_scale = noise_scale
        self.eps = eps

    def loss_euclidean(self, model, x0, x1, t):
        """Compare model's velocity at x(t) with true geodesic velocity."""
        x_t = x0 + t * (x1 - x0)  # linear interpolation
        if self.noise_scale > 0:
            x_t = x_t + torch.randn_like(x_t) * self.noise_scale
        mu_t = model(t, x_t)
        dist_g = x1 - mu_t
        return dist_g.pow(2).mean()
