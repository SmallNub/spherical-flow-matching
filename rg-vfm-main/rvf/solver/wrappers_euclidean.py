# Description: Wrappers defined to be used in pair with the odeint solver
from torch import nn
import torch


class VelocityInference(nn.Module):
    """
    A wrapper that takes a model and compute the velocity at a point x(t).
    To be used with variational Flow Matching when p0 is in R^3
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        scale = torch.clip(torch.ones_like(t) / (1-t), 0, 20)
        velocity = (self.model(t, x) - x) * scale
        return velocity


class VelocityWrapper(nn.Module):
    """
    A wrapper that rewrap the velocity at a point x(t).
    To be used with regular Flow Matching whenever p0 is in R^3 or Sphere
    (velocity already projected on the sphere in the Riemannian case)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        return self.model(t, x)
