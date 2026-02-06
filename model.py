"""Model definitions for distributed training experiments."""

import math

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """Simple linear model for binary classification.

    Args:
        d: Input feature dimension
    """

    def __init__(self, d: int):
        super().__init__()
        # Use nn.Linear for better compatibility with quantization
        self.linear = nn.Linear(d, 1, bias=True)
        # Initialize with same scheme as before
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0 / math.sqrt(d))
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """Forward pass: y = linear(x)"""
        # Return shape [batch_size] instead of [batch_size, 1] for BCE loss
        return self.linear(x).squeeze(-1)
