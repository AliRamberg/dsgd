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
        self.w = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        """Forward pass: y = x @ w"""
        return x @ self.w
