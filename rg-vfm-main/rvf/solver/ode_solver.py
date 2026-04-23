import torch
from typing import Callable, Union, Tuple, Sequence
from rvf.manifolds.hyperboloid import HyperboloidManifold
from rvf.manifolds.base import BaseManifold


class ODESolver:
    """
    A simple ODE solver supporting Euler integration on Euclidean and Riemannian manifolds.

    Args:
        velocity_model: Callable f(t, x) -> dx/dt
        method: Integration method, currently only 'euler' is supported
    """
    def __init__(self, velocity_model, method: str = 'euler', manifold: BaseManifold = None):
        self.velocity_model = velocity_model
        self.method = method.lower()
        self.manifold = manifold if manifold is not None else HyperboloidManifold()
        if self.method not in ('euler',):
            raise ValueError(f"Unknown integration method '{method}'. Supported: 'euler'.")

    def sample(self, x0: torch.Tensor, t_span: torch.Tensor, support: str = 'extrinsic') -> torch.Tensor:
        """
        Integrate the ODE defined by velocity_model over the times in t_span.

        Args:
            x0: Initial state, shape [...]
            t_span: 1D tensor of times, shape [T]
            support: 'intrinsic' to use Riemannian (Hyperboloid) integration, 'extrinsic' for Euclidean
        Returns:
            sols: Tensor of shape [T, ...] with the solution at each time in t_span
        """
        support = support.lower()
        if support == 'intrinsic':
            return self._euler_integrate_riem(self.velocity_model, x0, t_span, self.manifold)
        elif support == 'extrinsic':
            return self._euler_integrate(self.velocity_model, x0, t_span)
        else:
            raise ValueError(f"Unknown support '{support}'. Use 'intrinsic' or 'extrinsic'.")

    def compute_likelihood(
        self,
        x1: torch.Tensor,
        log_p0: Callable[[torch.Tensor], torch.Tensor],
        t_span: torch.Tensor,
        support: str = 'extrinsic',
        exact_divergence: bool = True,
        return_intermediates: bool = False,
        enable_grad: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Sequence[torch.Tensor], torch.Tensor]]:
        """
        Compute log-likelihood of samples x1 at t=1 by integrating the ODE backward to t=0
        and accounting for volume change via divergence.

        Args:
            x1: Samples at t=1, shape [batch, ...]
            log_p0: Function giving log p0(x) for x at t=0
            t_span: 1D tensor of times descending from 1.0 to 0.0 (shape [T])
            support: 'intrinsic' or 'extrinsic'
            exact_divergence: if True, compute exact divergence; else use Hutchinson estimator
            return_intermediates: if True, return full path of x and final log-prob
            enable_grad: whether to enable gradients through velocity_model
        Returns:
            (x0, log_prob) or (xs, log_prob)
        """
        t_span = t_span.to(x1.device)
        assert t_span[0] > t_span[-1], f"t_span must descend from t=1 to t=0, got {t_span}"
        support = support.lower()

        batch_size = x1.shape[0]

        # draw random +1/−1 for Hutchinson (if needed)
        if not exact_divergence:
            z = (torch.randn_like(x1) < 0).float().to(x1) * 2.0 - 1.0

        # state and log‐density change accumulator
        x = x1
        logp_delta = torch.zeros(batch_size, device=x.device)

        # to store intermediate x's if desired
        if return_intermediates:
            xs = [x]

        manifold = self.manifold

        # ——— backward Euler loop ———
        for i in range(len(t_span) - 1):
            t_i = t_span[i]
            dt  = t_span[i+1] - t_i   # negative, since t_span descends

            # 1) compute velocity for stepping WITHOUT building a graph
            with torch.no_grad():
                dx = self.velocity_model(t_i, x)

            # 2) compute divergence under its own grad context
            with torch.enable_grad():
                # force x to require grad for autodiff
                x_grad = x.detach().requires_grad_(True)
                dx_grad = self.velocity_model(t_i, x_grad)

                if exact_divergence:
                    # exact trace-divergence
                    flat = dx_grad.flatten(start_dim=1)
                    div = torch.zeros(batch_size, device=x.device)
                    for j in range(flat.shape[1]):
                        # ∂(flat[:,j].sum()) / ∂x_grad → grads w.r.t x
                        g = torch.autograd.grad(flat[:, j].sum(), x_grad, create_graph=True)[0]
                        div = div + g.flatten(start_dim=1)[:, j]
                else:
                    # Hutchinson estimator
                    flat = dx_grad.flatten(start_dim=1)
                    inner = (flat * z.flatten(start_dim=1)).sum(dim=1)
                    # grad(inner.sum()) w.r.t. x_grad
                    g = torch.autograd.grad(inner.sum(), x_grad, retain_graph=False)[0]
                    div = (g.flatten(start_dim=1) * z.flatten(start_dim=1)).sum(dim=1)

            # 3) backward‐Euler step on the state
            if support == 'intrinsic':
                x = manifold.exp_map(x, dt * dx)
            else:
                x = x + dt * dx

            # 4) accumulate change in log‐density:  d log p = - div · dt
            logp_delta = logp_delta - div * dt

            if return_intermediates:
                xs.append(x)

        # At t=0, add base density
        x0 = x
        logp0 = log_p0(x0)
        log_prob = logp0 + logp_delta

        if return_intermediates:
            return xs, log_prob
        else:
            return x0, log_prob

    def _euler_integrate(self, func, x0, t_span):
        device = x0.device
        t_span = t_span.to(device)
        x = x0
        sols = [x]

        for i in range(len(t_span) - 1):
            t_i = t_span[i]
            dt = t_span[i + 1] - t_i
            dx = func(t_i, x)
            x = x + dt * dx
            sols.append(x)

        return torch.stack(sols, dim=0)

    def _euler_integrate_riem(self, func, x0, t_span, manifold):
        device = x0.device
        t_span = t_span.to(device)
        x = x0
        sols = [x]

        for i in range(len(t_span) - 1):
            t_i = t_span[i]
            dt = t_span[i + 1] - t_i
            dx = func(t_i, x)
            x = manifold.exp_map(x, dt * dx)
            sols.append(x)

        return torch.stack(sols, dim=0)
