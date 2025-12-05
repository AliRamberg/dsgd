from dataclasses import dataclass


@dataclass
class TrainConfig:
    d: int = 200  # feature dimension
    lr: float = 0.05  # learning rate
    batch_size: int = 512  # batch size per worker step
    total_updates: int = 100  # target total parameter updates across all methods
    device: str = "cpu"
    hetero_base: float = 0.0  # baseline worker delay in seconds (0 = no delays)
    hetero_jitter: float = 0.0  # jitter in worker delay
    hetero_straggler_every: int = 0  # straggler frequency (0 = disabled)
    eval_every: int = 5  # evaluate loss every N steps
