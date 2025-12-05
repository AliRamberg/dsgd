from __future__ import annotations
import ray
import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import TYPE_CHECKING, Optional

from config import TrainConfig
from model import LinearModel
from utils import sleep_heterogeneity, setup_actor_logging
from logger import logger

if TYPE_CHECKING:
    from metrics import MetricsCollector
    from data import DatasetShard


@ray.remote
class SSPController:

    def __init__(self, d: int, lr: float, device: str, staleness: int, eval_every: int = 5) -> SSPController:
        # Configure logging for this Ray actor
        self.logger = setup_actor_logging()

        self.model = LinearModel(d).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.staleness = staleness
        self.eval_every = eval_every
        self.worker_steps: dict[int, int] = {}
        self.global_updates = 0
        self.loss_history: list[tuple[int, float]] = []  # Track (update_count, loss) pairs
        self.X_full = None
        self.y_full = None

    def set_full_dataset(self, X: torch.Tensor, y: torch.Tensor):
        """Set full dataset for periodic evaluation"""
        self.X_full = X.to(self.device)
        self.y_full = y.to(self.device)

    def register_worker(self, wid: int):
        self.worker_steps[wid] = -1  # -1 means no steps completed yet
        self.logger.debug(f"SSP: Registered worker {wid}")

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}, self.global_updates

    def get_loss_history(self):
        return self.loss_history

    def can_proceed(self, wid: int, step_i: int):
        # SSP rule: fast workers can be at most s ahead of slowest worker
        # IMPORTANT: Non-blocking - returns immediately to avoid actor deadlock

        # Get min of ALL workers' COMPLETED steps
        min_step = min(self.worker_steps.values(), default=0)
        staleness = step_i - min_step

        # Can proceed if this step is at most s ahead of the slowest worker
        can_go = staleness <= self.staleness

        if can_go:
            self.logger.debug(f"SSP: Worker {wid} step {step_i} CAN PROCEED (min={min_step}, staleness={staleness})")
        else:
            self.logger.debug(f"SSP: Worker {wid} step {step_i} BLOCKED (min={min_step}, staleness={staleness} > bound={self.staleness})")

        return can_go, staleness if can_go else 0

    def push_grad(self, wid: int, step_i: int, grad_dict: dict):
        # Apply gradient immediately (like ASGD)
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if name in grad_dict:
                param.grad = grad_dict[name].to(self.device)
        self.optimizer.step()
        self.global_updates += 1

        # Periodic evaluation
        if self.X_full is not None and self.y_full is not None:
            if self.global_updates % self.eval_every == 0 or self.global_updates == 1:
                with torch.no_grad():
                    logits = self.model(self.X_full).squeeze()
                    loss = float(F.binary_cross_entropy_with_logits(logits, self.y_full))
                self.loss_history.append((self.global_updates, loss))

        # Mark this step as COMPLETED after gradient is applied
        self.worker_steps[wid] = step_i
        self.logger.debug(f"SSP: Worker {wid} completed step {step_i}, global_updates now {self.global_updates}")
        return self.global_updates


@ray.remote
class SSPWorker:

    def __init__(self, worker_id: int, ctrl, shard, cfg: TrainConfig):
        # Configure logging for this Ray actor
        self.logger = setup_actor_logging()

        self.id = worker_id
        self.ctrl = ctrl
        self.ds = shard
        self.cfg = cfg
        self.local_step = 0

    def loop(self, iterations: int):
        """Worker loop that returns iteration metrics for aggregation."""
        # Create local model for gradient computation
        local_model = LinearModel(self.cfg.d).to(self.cfg.device)

        metrics_data = []
        self.logger.debug(f"SSP: Worker {self.id} starting loop for {iterations} iterations")
        for i in range(iterations):
            t0 = time.time()
            # Poll until we get permission (non-blocking on actor side)
            self.logger.debug(f"SSP: Worker {self.id} requesting permission for step {self.local_step}")
            while True:
                can_go, staleness = ray.get(self.ctrl.can_proceed.remote(self.id, self.local_step))
                if can_go:
                    break
                # Backoff before retrying
                time.sleep(0.001)

            # Fetch current weights and compute gradient
            (state_dict, global_step) = ray.get(self.ctrl.get_weights.remote())
            local_model.load_state_dict(state_dict)

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

            # Extract gradients and push (marks step as completed)
            grad_dict = {name: p.grad.cpu() for name, p in local_model.named_parameters() if p.grad is not None}
            ver = ray.get(self.ctrl.push_grad.remote(self.id, self.local_step, grad_dict))

            iter_time = time.time() - t0
            self.logger.debug(f"SSP: Worker {self.id} completed step {self.local_step}, staleness={staleness}")
            self.local_step += 1

            # Collect metrics data (will be recorded by main loop)
            metrics_data.append(
                {
                    "step": ver,
                    "iter": i,
                    "latency_ms": iter_time * 1000,
                    "staleness": staleness,
                }
            )

        return {
            "worker": self.id,
            "local_steps": self.local_step,
            "metrics_data": metrics_data,
        }


def run_ssp(
    cfg: TrainConfig,
    shards: list,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    ssp_s: int,
    metrics: Optional[MetricsCollector] = None,
):
    """SSP: Stale Synchronous Parallel with bounded staleness

    Target: ~cfg.total_updates parameter updates
    Each worker does total_updates // num_workers iterations (with SSP blocking)
    """
    if metrics:
        metrics.start_training()

    num_workers = len(shards)
    grad_size_bytes = cfg.d * 4  # float32 = 4 bytes
    # Communication: each worker push is grad_size_bytes
    comm_bytes_per_update = grad_size_bytes

    logger.debug(f"SSP: Starting training with staleness bound s={ssp_s}, {num_workers} workers")
    ctrl = SSPController.remote(cfg.d, cfg.lr, cfg.device, staleness=ssp_s, eval_every=cfg.eval_every)
    # Set full dataset for periodic evaluation (keep on same device as model)
    ray.get(ctrl.set_full_dataset.remote(X_full, y_full))
    workers = [SSPWorker.remote(i, ctrl, shards[i], cfg) for i in range(num_workers)]

    ray.get([ctrl.register_worker.remote(i) for i in range(num_workers)])
    logger.debug(f"SSP: All {num_workers} workers registered")

    steps_per_worker = cfg.total_updates // num_workers
    logger.debug(f"SSP: Each worker will run {steps_per_worker} iterations")
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

        # Periodic loss evaluation using controller's loss_history
        loss_history = ray.get(ctrl.get_loss_history.remote())
        if loss_history:
            for update_count, loss in loss_history:
                metrics.record_loss(update_count, loss, comm_bytes=0)

        # Final evaluation
        state_dict, ver = ray.get(ctrl.get_weights.remote())
        eval_model = LinearModel(cfg.d).to(cfg.device)
        eval_model.load_state_dict(state_dict)
        eval_model.eval()
        with torch.no_grad():
            final_loss = float(F.binary_cross_entropy_with_logits(eval_model(X_full).squeeze(), y_full))
        metrics.record_final(ver, final_loss, ver)
        metrics.stop_training()

        logger.debug(f"SSP: Training complete, final version={ver}, loss={final_loss:.5f}")
        print(f"[SSP s={ssp_s}] updates={ver} loss={final_loss:.5f}")
    else:
        state_dict, ver = ray.get(ctrl.get_weights.remote())
        eval_model = LinearModel(cfg.d).to(cfg.device)
        eval_model.load_state_dict(state_dict)
        eval_model.eval()
        with torch.no_grad():
            loss = float(F.binary_cross_entropy_with_logits(eval_model(X_full).squeeze(), y_full))
        logger.debug(f"SSP: Training complete, final version={ver}, loss={loss:.5f}")
        print(f"[SSP s={ssp_s}] updates={ver} loss={loss:.5f}")

    for s in stats:
        avg_staleness = float(np.mean([m["staleness"] for m in s["metrics_data"]]))
        print(f"  worker {s['worker']}: steps={s['local_steps']} avg_staleness≈{avg_staleness:.2f}")
