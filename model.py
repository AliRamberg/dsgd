"""Model definitions for distributed training experiments."""

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """Simple linear model for binary classification.

    Args:
        d: Input feature dimension
    """

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Forward pass: y = x @ w + b"""
        return x @ self.w + self.b
