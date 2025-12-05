from __future__ import annotations
import ray
import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import TYPE_CHECKING, Optional

from config import TrainConfig
from model import LinearModel
from utils import sleep_heterogeneity

if TYPE_CHECKING:
    from metrics import MetricsCollector
    from data import DatasetShard

@ray.remote
class ASGDParameterServer:
    def __init__(self, d: int, lr: float, device: str, eval_every: int = 5):
        self.model = LinearModel(d).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.num_updates = 0
        self.eval_every = eval_every
        self.loss_history: list[tuple[int, float]] = []  # Track (update_count, loss) pairs
        self.X_full = None
        self.y_full = None

    def set_full_dataset(self, X: torch.Tensor, y: torch.Tensor):
        """Set full dataset for periodic evaluation"""
        self.X_full = X.to(self.device)
        self.y_full = y.to(self.device)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_num_updates(self):
        return self.num_updates

    def get_loss_history(self):
        return self.loss_history

    def push_grad(self, grad_dict: dict):
        # apply immediately
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if name in grad_dict:
                param.grad = grad_dict[name].to(self.device)
        self.optimizer.step()
        self.num_updates += 1

        # Periodic evaluation
        if self.X_full is not None and self.y_full is not None:
            if self.num_updates % self.eval_every == 0 or self.num_updates == 1:
                with torch.no_grad():
                    logits = self.model(self.X_full).squeeze()
                    loss = float(F.binary_cross_entropy_with_logits(logits, self.y_full))
                self.loss_history.append((self.num_updates, loss))

        # return current model version index
        return self.num_updates


@ray.remote
class ASGDWorker:
    def __init__(self, worker_id: int, ps, shard, cfg: TrainConfig):
        self.id = worker_id
        self.ps = ps
        self.ds = shard
        self.cfg = cfg
        self.local_step = 0

    def loop(self, iterations: int):
        """Worker loop that returns iteration metrics for aggregation."""
        # Create local model for gradient computation
        local_model = LinearModel(self.cfg.d).to(self.cfg.device)

        metrics_data = []
        for i in range(iterations):
            t0 = time.time()

            # Fetch weights from parameter server
            state_dict = ray.get(self.ps.get_weights.remote())
            local_model.load_state_dict(state_dict)

            # Sample batch
            X, y = self.ds.sample_batch(self.cfg.batch_size)
            sleep_heterogeneity(
                self.id,
                self.cfg.hetero_base,
                self.cfg.hetero_jitter,
                self.cfg.hetero_straggler_every,
                step=self.local_step,
            )

            # Forward + backward pass
            logits = local_model(X).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, y)

            local_model.zero_grad()
            loss.backward()

            # Extract gradients and send to parameter server
            grad_dict = {name: p.grad.cpu() for name, p in local_model.named_parameters() if p.grad is not None}
            ver = ray.get(self.ps.push_grad.remote(grad_dict))

            iter_time = time.time() - t0
            self.local_step += 1
            # how stale was the weight we used? approximate via global version when we fetched vs after push
            s = max(0, ver - 1)

            # Collect metrics data (will be recorded by main loop)
            metrics_data.append(
                {
                    "step": ver,
                    "iter": i,
                    "latency_ms": iter_time * 1000,
                    "staleness": s,
                }
            )

        return {
            "worker": self.id,
            "local_steps": self.local_step,
            "metrics_data": metrics_data,
        }


def run_asgd(
    cfg: TrainConfig,
    shards: list,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    metrics: Optional[MetricsCollector] = None,
):
    """ASGD: Asynchronous SGD with parameter server

    Target: ~cfg.total_updates parameter updates
    Each worker does total_updates // num_workers iterations
    """
    if metrics:
        metrics.start_training()

    num_workers = len(shards)
    grad_size_bytes = cfg.d * 4  # float32 = 4 bytes
    # Communication: each worker push is grad_size_bytes
    comm_bytes_per_update = grad_size_bytes

    ps = ASGDParameterServer.remote(cfg.d, cfg.lr, cfg.device, eval_every=cfg.eval_every)
    # Set full dataset for periodic evaluation (keep on same device as model)
    ray.get(ps.set_full_dataset.remote(X_full, y_full))
    workers = [ASGDWorker.remote(i, ps, shards[i], cfg) for i in range(len(shards))]

    steps_per_worker = cfg.total_updates // num_workers

    # Run workers and collect stats
    stats = ray.get([w.loop.remote(steps_per_worker) for w in workers])

    # Record iteration metrics
    if metrics:
        for worker_stat in stats:
            worker_id = worker_stat["worker"]
            for m in worker_stat["metrics_data"]:
                metrics.record_iteration(
                    worker_id=worker_id,
                    step=m["step"],
                    latency_ms=m["latency_ms"],
                    staleness=m["staleness"],
                    comm_bytes=comm_bytes_per_update,
                )

        # Periodic loss evaluation - evaluate on full dataset periodically
        final_updates = ray.get(ps.get_num_updates.remote())
        state_dict = ray.get(ps.get_weights.remote())

        # Evaluate loss periodically during training (evaluate from parameter server's history if available)
        loss_history = ray.get(ps.get_loss_history.remote())
        if loss_history:
            for update_count, loss in loss_history:
                metrics.record_loss(update_count, loss, comm_bytes=0)

        # Final evaluation
        eval_model = LinearModel(cfg.d).to(cfg.device)
        eval_model.load_state_dict(state_dict)
        eval_model.eval()
        with torch.no_grad():
            final_loss = float(F.binary_cross_entropy_with_logits(eval_model(X_full).squeeze(), y_full))
        metrics.record_final(final_updates, final_loss, final_updates)
        metrics.stop_training()

        print(f"[ASGD] updates={final_updates} loss={final_loss:.5f}")
    else:
        state_dict = ray.get(ps.get_weights.remote())
        eval_model = LinearModel(cfg.d).to(cfg.device)
        eval_model.load_state_dict(state_dict)
        eval_model.eval()
        with torch.no_grad():
            loss = float(F.binary_cross_entropy_with_logits(eval_model(X_full).squeeze(), y_full))
        final_updates = ray.get(ps.get_num_updates.remote())
        print(f"[ASGD] updates={final_updates} loss={loss:.5f}")

    for s in stats:
        avg_staleness = float(np.mean([m["staleness"] for m in s["metrics_data"]]))
        print(f"  worker {s['worker']}: steps={s['local_steps']} avg_staleness≈{avg_staleness:.2f}")
