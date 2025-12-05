from torch.utils.data import Dataset
import torch


class SyntheticDataset(Dataset):
    """PyTorch Dataset for synthetic XOR classification data with polynomial features

    This Dataset can be used with PyTorch DataLoader and Ray Train's data loading utilities.
    All data is generated upfront for reproducibility and performance.
    """

    def __init__(self, n: int, d: int, noise: float = 0.1, seed: int = 0):
        """
        Args:
            n: Number of samples
            d: Base feature dimension (will be expanded to d+3 with polynomial features)
            noise: Label noise probability (default: 0.1)
            seed: Random seed for reproducibility
        """
        self.n = n
        self.d = d

        # Generate all data upfront
        gen = torch.Generator().manual_seed(seed)
        X_raw = torch.randn(n, d, generator=gen)

        # XOR pattern on first 2 dimensions
        y = ((X_raw[:, 0] > 0).long() ^ (X_raw[:, 1] > 0).long()).float()

        # Add label noise
        flip = torch.rand(n, generator=gen) < noise
        y[flip] = 1 - y[flip]

        # Add polynomial features: [x, x^2, x1*x2] to make XOR linearly separable
        self.X = torch.cat(
            [X_raw, X_raw[:, :2] ** 2, (X_raw[:, 0:1] * X_raw[:, 1:2])], dim=1  # quadratic terms  # interaction term (key for XOR)
        )
        self.y = y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DatasetShard:
    """DatasetShard for custom algorithms (ASGD/SSP/LocalSGD)"""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, device: str):
        self.X = X.to(device)
        self.y = y.to(device)
        self.N = self.X.shape[0]
        self.device = device

    def sample_batch(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.N, (batch,), device=self.device)
        return self.X[idx], self.y[idx]
