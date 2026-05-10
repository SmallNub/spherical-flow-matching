import torch
import torch.nn as nn
from manifm.model.arch import ACTFNS
import manifm.model.diffeq_layers as diffeq_layers


NormLayer = nn.Identity


class FlowModelOverride(nn.Module):
    """A wrapper to override the forward method of the original model with conditional inputs."""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        actfn,
        fourier=None,
        dropout=0.3,
        num_classes=None,
        null_chance=0.0,
    ):
        super().__init__()
        if fourier:
            raise NotImplementedError("Fourier features not implemented in override model.")

        self.is_cond = num_classes is not None and num_classes > 0
        actfn = ACTFNS[actfn]

        self.stem = Stem(
            in_dim,
            hidden_dim,
            actfn,
            num_classes=num_classes,
            null_chance=null_chance,
        )

        layers = []
        for _ in range(num_layers - 2):
            layers.append(LinearBlock(hidden_dim, hidden_dim, actfn, dropout=dropout))
        self.core = nn.ModuleList(layers)

        self.head = Head(hidden_dim, in_dim)

    def forward(self, t, x, y=None):
        x = self.stem(t, x, y=y)

        for layer in self.core:
            x = layer(t, x)

        x = self.head(t, x)
        return x


class ConditionalModel(nn.Module):
    def __init__(self, num_classes, in_dim, embed_dim, out_dim, actfn, null_chance=0.0):
        super().__init__()
        self.null_chance = null_chance

        self.embedding = nn.Embedding(num_classes + 1 if null_chance > 0 else num_classes, embed_dim)
        self.linear = diffeq_layers.ConcatLinear_v2(embed_dim + in_dim, out_dim)
        self.norm = NormLayer(out_dim)
        self.actfn = actfn(out_dim)

    def forward(self, t, x, y):
        if self.null_chance > 0:
            if torch.rand(1).item() < self.null_chance:
                y = y * 0  # Label 0 is reserved for the null class
            else:
                y = y + 1  # Shift labels to account for null class

        y = self.embedding(y)
        x = torch.cat([x, y], dim=1)
        x = self.linear(t, x)
        x = self.norm(x)
        x = self.actfn(t, x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, actfn, dropout=0.1):
        super().__init__()
        self.norm = NormLayer(in_dim)
        self.linear = diffeq_layers.ConcatLinear_v2(in_dim, out_dim)
        self.actfn = actfn(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, x):
        residual = x
        x = self.norm(x)
        x = self.linear(t, x)
        x = self.actfn(t, x)
        x = self.dropout(x)
        x = x + residual
        return x


class Stem(nn.Module):
    def __init__(self, in_dim, out_dim, actfn, num_classes=None, null_chance=0.0):
        super().__init__()
        self.actfn = actfn(out_dim)
        self.cond_model = None

        if num_classes is not None and num_classes > 0:
            self.cond_model = ConditionalModel(
                num_classes,
                in_dim,
                in_dim,
                out_dim,
                actfn,
                null_chance
            )

        self.linear = diffeq_layers.ConcatLinear_v2(out_dim if self.cond_model else in_dim, out_dim)
        self.norm = NormLayer(out_dim)

    def forward(self, t, x, y=None):
        if self.cond_model:
            if y is None:
                raise ValueError("Conditional inputs are required.")
            x = self.cond_model(t, x, y)

        x = self.linear(t, x)
        x = self.norm(x)
        x = self.actfn(t, x)
        return x


class Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = NormLayer(in_dim)
        self.linear = diffeq_layers.ConcatLinear_v2(in_dim, out_dim)

    def forward(self, t, x):
        x = self.norm(x)
        x = self.linear(t, x)
        return x
