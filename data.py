from torch.utils.data import Dataset
import torch


class SyntheticDataset(Dataset):
    """PyTorch Dataset for synthetic linear classification using ALL features.

    This Dataset can be used with PyTorch DataLoader and Ray Train's data loading utilities.
    All data is generated upfront for reproducibility and performance.

    The problem is linearly separable: y = sign(X @ w_true) where w_true uses all dimensions.
    """

    def __init__(
        self,
        n: int,
        d: int,
        noise: float = 0.1,
        seed: int = 0,
        w_true: torch.Tensor = None,
    ):
        """
        Args:
            n: Number of samples
            d: Feature dimension
            noise: Label noise probability (default: 0.1)
            seed: Random seed for reproducibility
            w_true: Optional pre-defined true weights (for consistent labels across chunks)
        """
        self.n = n
        self.d = d

        gen = torch.Generator().manual_seed(seed)

        # Generate features
        X_raw = torch.randn(n, d, generator=gen)

        # Use provided w_true or generate new one
        if w_true is None:
            w_true = torch.randn(d, generator=gen) / (d**0.5)

        # Linear decision boundary: y = sign(X @ w_true)
        logits = X_raw @ w_true
        y = (logits > 0).float()

        # Add label noise
        flip = torch.rand(n, generator=gen) < noise
        y[flip] = 1 - y[flip]

        self.X = X_raw
        self.y = y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DatasetShard:
    """DatasetShard for custom algorithms (ASGD/SSP/LocalSGD).

    Created on CPU by the driver (head node), then moved to the target device
    inside the Ray worker actor via `to()`. This avoids CUDA calls on the
    GPU-less head node.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, device: str = "cpu"):
        self.X = X.to(device)
        self.y = y.to(device)
        self.N = self.X.shape[0]
        self.device = device

    def to(self, device: str) -> "DatasetShard":
        """Move shard data to a different device (e.g. 'cuda')."""
        if device != self.device:
            self.X = self.X.to(device)
            self.y = self.y.to(device)
            self.device = device
        return self

    def sample_batch(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.N, (batch,), device=self.device)
        return self.X[idx], self.y[idx]
