import torch
import torch.nn as nn


class VectorDynamics(nn.Module):
    """
    A small MLP that outputs a vector in R^3, 
    """
    def __init__(self, 
        time_dim: int = 1,       
        input_dim: int = 3,
        hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(time_dim + input_dim, 64)

        self.net = nn.Sequential(
            nn.SELU(),
            nn.Linear(64, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 64),
            nn.SELU(),
            nn.Linear(64, input_dim),
        )
    
    def forward(self, t, x): # always (t, x) to be used with odeint
        """
        x: (batch_size, 3) on manifold
        t: (batch_size, 1) in [0,1]
        return: (batch_size, 3) tangent vector
        """
        # assert x.shape[-1] == self.input_dim, f"Expected x to have shape (batch_size, {self.input_dim}), got {x.shape}"
        # assert t.shape[-1] == 1, f"Expected t to have shape (batch_size, 1), got {t.shape}"
        inp = torch.cat([t, x], dim=-1)  # shape (batch_size, 3+1)
        inp = self.input_layer(inp)
        return self.net(inp)


class PositionDynamics(VectorDynamics):
    """ 
    PositionDynamics is structurally the same as VectorDynamics,
    but is defined separately for readability and future extensions.
    """
    pass  # For now, it inherits everything from VectorDynamics

