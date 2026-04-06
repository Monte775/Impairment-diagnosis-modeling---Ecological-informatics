"""MLP model for river impairment diagnosis (binary classification).

Architecture:
    - Narrowing hidden layers: each subsequent layer's width is reduced by `ratio`.
    - Dropout (p=0.2) after the first layer.
    - Softmax output for 2-class probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPImpairment(nn.Module):
    """Multi-Layer Perceptron for binary river-impairment classification.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output classes (default 2).
    hidden_dim : int
        Width of the first hidden layer.
    num_layer : int
        Number of hidden layers.
    act : str
        Activation function for hidden layers.
        One of ``"leaky_relu"``, ``"elu"``, ``"linear"``.
    ratio : float
        Shrinkage ratio applied to successive hidden-layer widths (0 < ratio <= 1).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 64,
        num_layer: int = 3,
        act: str = "leaky_relu",
        ratio: float = 0.5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layer = num_layer
        self.act = act
        self.dropout = nn.Dropout(0.2)

        hdim = self.hidden_dim
        current_dim = self.input_dim
        self.layers = nn.ModuleList()

        for layer_idx in range(num_layer):
            if layer_idx == 0:
                self.layers.append(nn.Linear(current_dim, hdim))
                current_dim = hdim
            else:
                next_dim = max(int(hdim * ratio), 2)
                self.layers.append(nn.Linear(current_dim, next_dim))
                current_dim = next_dim
                hdim = next_dim

        self.layers.append(nn.Linear(current_dim, self.output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            if i == 0:
                x = self.dropout(layer(x))
            else:
                if self.act == "leaky_relu":
                    x = F.leaky_relu(layer(x))
                elif self.act == "elu":
                    x = F.elu(layer(x))
                elif self.act == "linear":
                    x = layer(x)

        out = F.softmax(self.layers[-1](x), dim=-1)
        return out
