from __future__ import annotations
import ray
import torch
import torch.nn.functional as F
import math
from typing import TYPE_CHECKING, Optional

from config import TrainConfig
from model import LinearModel
from utils import sleep_heterogeneity
from metrics import PrometheusMetricCollector

if TYPE_CHECKING:
    from metrics import MetricsCollector
    from data import DatasetShard


@ray.remote
class LocalAverager:
    def __init__(self, d: int, device: str = "cpu"):
        self.d = d
        self.device = device
        self.last_broadcast = None

    def average_and_broadcast(self, state_dicts: list[dict]):
        # Average all parameters across state_dicts
        avg_state = {}
        for key in state_dicts[0].keys():
            tensors = [sd[key] for sd in state_dicts]
            avg_state[key] = torch.stack(tensors).mean(dim=0)
        self.last_broadcast = avg_state
        return avg_state


@ray.remote
class LocalWorker:
    def __init__(
        self,
        worker_id: int,
        shard,
        cfg: TrainConfig,
        initial_w: Optional[dict] = None,
    ):
        self.id = worker_id
        self.ds = shard.to(cfg.device)  # Move shard to GPU (arrived on CPU from head)
        self.cfg = cfg
        self.model = LinearModel(cfg.d).to(cfg.device)
        if initial_w is not None:
            self.model.load_state_dict(initial_w)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # Prometheus metrics for this worker
        self.prom_metrics = PrometheusMetricCollector(
            mode="localsgd",
            worker_id=worker_id,
            run_info=cfg.run_info,
        )
        self.prom_metrics.log_training_start()
        self.total_steps = 0

    def local_steps(self, K: int, eval_every: int = 5) -> dict:
        import time

        for k in range(K):
            t0 = time.time()

            X, y = self.ds.sample_batch(self.cfg.batch_size)
            sleep_heterogeneity(
                self.id,
                self.cfg.hetero_base,
                self.cfg.hetero_jitter,
                self.cfg.hetero_straggler_every,
                step=k,
            )
            self.optimizer.zero_grad()
            logits = self.model(X).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            self.optimizer.step()

            iter_time = time.time() - t0

            # Log to Prometheus
            self.prom_metrics.log_iteration(
                step=self.total_steps,
                latency_ms=iter_time * 1000,
                comm_bytes=0,  # No communication during local steps
                staleness=0,
            )
            self.prom_metrics.log_samples_processed(self.cfg.batch_size)

            self.total_steps += 1

        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, state_dict: dict):
        self.model.load_state_dict(state_dict)
        # Note: optimizer state is preserved (not reset)

    def current_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def finish_training(self):
        """Log training completion"""
        self.prom_metrics.log_training_end()


def run_localsgd(
    cfg: TrainConfig,
    shards: list,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    K: int,
    metrics: Optional[MetricsCollector] = None,
):
    """LocalSGD: K local steps then weight averaging

    Target: ~cfg.total_updates parameter updates
    Each sync round does K local updates per worker = K global updates
    Number of rounds = total_updates // K
    """
    if metrics:
        metrics.start_training()

    num_workers = len(shards)
    param_size_bytes = cfg.d * 4  # float32 = 4 bytes
    # Communication: all workers send weights, then receive averaged weights
    # Each sync round: num_workers * param_size (send) + num_workers * param_size (receive)
    comm_bytes_per_round = 2 * num_workers * param_size_bytes

    avg = LocalAverager.remote(cfg.d, cfg.device)
    workers = [LocalWorker.remote(i, shards[i], cfg) for i in range(len(shards))]

    w_global = {
        k: v for k, v in LinearModel(cfg.d).state_dict().items()
    }  # Initialize with zeros
    num_rounds = math.ceil(cfg.total_updates / K)
    for r in range(num_rounds):
        # local K steps in parallel
        local_ws = ray.get([w.local_steps.remote(K, cfg.eval_every) for w in workers])
        # average and broadcast
        w_global = ray.get(avg.average_and_broadcast.remote(local_ws))
        for w in workers:
            w.set_weights.remote(w_global)

        # Evaluate and record loss
        step = (r + 1) * K  # Approximate global step
        should_eval = (r % 2 == 0) or (r == num_rounds - 1)
        if should_eval:
            eval_model = LinearModel(cfg.d).to(cfg.device)
            eval_model.load_state_dict(w_global)
            eval_model.eval()
            with torch.no_grad():
                loss = float(
                    F.binary_cross_entropy_with_logits(
                        eval_model(X_full).squeeze(), y_full
                    )
                )
            print(f"[LocalSGD K={K}] round={r:3d} updates≈{step:4d} loss={loss:.5f}")

            if metrics:
                # Count communication for this round (and previous rounds if eval_every > 1)
                rounds_since_last_eval = 2 if r > 0 else (r + 1)
                if r == num_rounds - 1 and r % 2 != 0:
                    rounds_since_last_eval = r % 2 + 1
                comm_bytes = comm_bytes_per_round * rounds_since_last_eval
                metrics.record_round(step, loss, comm_bytes=comm_bytes)

    # Finish training for all workers
    ray.get([w.finish_training.remote() for w in workers])

    if metrics:
        metrics.stop_training()
        # Record final if not already recorded
        if num_rounds % 2 != 0:
            final_step = num_rounds * K
            eval_model = LinearModel(cfg.d).to(cfg.device)
            eval_model.load_state_dict(w_global)
            eval_model.eval()
            with torch.no_grad():
                final_loss = float(
                    F.binary_cross_entropy_with_logits(
                        eval_model(X_full).squeeze(), y_full
                    )
                )
            metrics.record_final(final_step, final_loss, cfg.total_updates)
