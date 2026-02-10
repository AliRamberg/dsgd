import torch
import torch.nn as nn


class LinearBinaryClassifier(nn.Module):
    """Simple linear binary classifier"""

    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 1, bias=True)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class LinearBinaryClassifierWithPadding(nn.Module):
    """
    Linear classifier with gradient padding for network bottleneck testing.

    Args:
        d: Feature dimension
        gradient_padding_mb: Dummy parameter size in MB (0 = no padding)
    """

    def __init__(self, d: int, gradient_padding_mb: int = 0):
        super().__init__()
        self.fc = nn.Linear(d, 1, bias=True)
        self.has_padding = gradient_padding_mb > 0

        if gradient_padding_mb > 0:
            padding_params = (gradient_padding_mb * 1024 * 1024) // 4
            self.dummy = nn.Parameter(
                torch.zeros(padding_params, dtype=torch.float32), requires_grad=True
            )
            print(
                f"[Gradient Padding] Added {padding_params:,} params ({gradient_padding_mb} MB)"
            )

    def forward(self, x):
        logits = self.fc(x).squeeze(-1)
        
        # Include dummy parameter in computation to satisfy DDP
        # Add 0.0 * dummy.sum() so it participates in backward pass but doesn't affect output
        if self.has_padding:
            logits = logits + 0.0 * self.dummy.sum()
        
        return logits
