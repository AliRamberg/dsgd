# distributed_sgd_showcase.py
"""
Distributed SGD showcase for four training modes on a single machine (CPU or a single GPU):
  1) SSGD (Synchronous SGD with per-step averaging)
  2) ASGD (Async SGD via parameter server)
  3) SSP  (Stale Synchronous Parallel with staleness bound s)
  4) LocalSGD (K local steps then synchronize parameters)

Why this file?
- Let you *see* differences in utilization, staleness, and convergence without a GPU cluster.
- Uses Ray Actors to emulate parameter servers / coordinators and workers.
- Simulates heterogeneity (stragglers) with sleep delays (disabled by default).

Prereqs (Python 3.9+):
    pip install ray torch

Run examples:
    python try1.py --mode ssgd --num-workers 4 --total-updates 500
    python try1.py --mode asgd --num-workers 4 --total-updates 500
    python try1.py --mode ssp  --num-workers 4 --total-updates 500 --ssp-staleness 2
    python try1.py --mode localsgd --num-workers 4 --total-updates 500 --local-k 5


Notes:
- Uses XOR classification with polynomial features (non-convex, learnable)
- Adam optimizer for better convergence than plain SGD
- Heterogeneity disabled by default; enable with --hetero-straggler-every N
- Works on CPU by default; use --device cuda for GPU
"""

from __future__ import annotations
import argparse
import time
import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import ray

from logger import logger
from metrics import MetricsCollector

# --------------------------- Utils ---------------------------


def get_collective_backend(device: str) -> str:
    """Determine Ray collective backend: 'nccl' for CUDA GPUs, 'gloo' for CPU"""
    if device == "cuda" and torch.cuda.is_available():
        return "nccl"
    return "gloo"


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_data(n: int, d: int, noise: float = 0.1, seed: int = 0):
    """Non-convex classification with polynomial features to enable XOR learning"""
    gen = torch.Generator().manual_seed(seed)
    X_raw = torch.randn(n, d, generator=gen)

    # XOR pattern on first 2 dimensions
    y = ((X_raw[:, 0] > 0).long() ^ (X_raw[:, 1] > 0).long()).float()

    # Add label noise
    flip = torch.rand(n, generator=gen) < noise
    y[flip] = 1 - y[flip]

    # Add polynomial features: [x, x^2, x1*x2] to make XOR linearly separable
    X_poly = torch.cat(
        [X_raw, X_raw[:, :2] ** 2, (X_raw[:, 0:1] * X_raw[:, 1:2])], dim=1  # quadratic terms  # interaction term (key for XOR)
    )

    return X_poly, y


def grad_bce(X: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gradient of BCE: X^T (sigmoid(Xw) - y) / N"""
    logits = X @ w
    probs = torch.sigmoid(logits)
    N = X.shape[0]
    return X.t() @ (probs - y) / N


# Simulate worker heterogeneity by sleeping (in seconds)
def sleep_heterogeneity(worker_id: int, base: float, jitter: float, straggler_every: int = 0, step: int = 0):
    """Simulate worker delays

    - base/jitter: adds small random delay to every step (if base > 0)
    - straggler_every: worker 0 becomes 5× slower every N steps (if > 0)
    """
    delay = 0.0

    # Base delay with jitter (applies every step if base > 0)
    if base > 0:
        delay = max(0.0, np.random.normal(loc=base, scale=jitter))

    # Periodic straggler: worker 0 gets 5× slower every straggler_every steps
    if worker_id == 0 and straggler_every != 0 and step != 0 and step % straggler_every == 0:
        delay += base * 5.0 if base > 0 else 0.1  # at least 0.1s if no base delay

    if delay > 0:
        print(f"Worker {worker_id} sleeping for {delay*1000:.2f}ms")
        time.sleep(delay)


# --------------------------- Ray Actors ---------------------------


@dataclass
class TrainConfig:
    d: int = 200  # feature dimension
    lr: float = 0.05  # learning rate
    batch: int = 512  # batch size per worker step
    total_updates: int = 100  # target total parameter updates across all methods
    device: str = "cpu"
    hetero_base: float = 0.0  # baseline worker delay in seconds (0 = no delays)
    hetero_jitter: float = 0.0  # jitter in worker delay
    hetero_straggler_every: int = 0  # straggler frequency (0 = disabled)
    eval_every: int = 5  # evaluate loss every N steps


@ray.remote
class DatasetShard:
    def __init__(self, X: torch.Tensor, y: torch.Tensor, device: str):
        self.X = X.to(device)
        self.y = y.to(device)
        self.N = self.X.shape[0]
        self.device = device

    def sample_batch(self, batch: int):
        idx = torch.randint(0, self.N, (batch,), device=self.device)
        return self.X[idx], self.y[idx]


# --------------- SSGD: synchronous, per-step gradient average ---------------


@ray.remote
class SSGDCoordinator:
    def __init__(self, d: int, lr: float, device: str):
        self.w = torch.zeros(d, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.w], lr=lr)
        self.device = device
        self.step = 0

    def get_weights(self):
        return self.w.detach().cpu()

    def aggregate_and_step(self, grads: list[torch.Tensor]):
        g = torch.stack(grads, dim=0).mean(dim=0).to(self.device)
        self.optimizer.zero_grad()
        self.w.grad = g
        self.optimizer.step()
        self.step += 1
        return self.w.detach().cpu(), self.step


@ray.remote
class SSGDWorker:
    def __init__(self, worker_id: int, coord: SSGDCoordinator, shard: DatasetShard, cfg: TrainConfig):
        self.id = worker_id
        self.coord = coord
        self.ds = shard
        self.cfg = cfg

    def step_once(self):
        # pull weights
        w = ray.get(self.coord.get_weights.remote()).to(self.cfg.device)
        X, y = ray.get(self.ds.sample_batch.remote(self.cfg.batch))
        sleep_heterogeneity(
            self.id,
            self.cfg.hetero_base,
            self.cfg.hetero_jitter,
            self.cfg.hetero_straggler_every,
        )
        g = grad_bce(X, w, y).cpu()
        return g


# --------------- SSGD AllReduce: decentralized gradient averaging ---------------


@ray.remote
class AllReduceOrchestrator:
    """Orchestrates AllReduce communication (gather-reduce-broadcast pattern)"""

    def allreduce_grads(self, grads: list[torch.Tensor]) -> torch.Tensor:
        """Average gradients across all workers"""
        return torch.stack(grads, dim=0).mean(dim=0)


@ray.remote
class SSGDAllReduceWorker:
    """SSGD worker that maintains local weights and applies AllReduce gradients"""

    def __init__(self, worker_id: int, shard: DatasetShard, cfg: TrainConfig):
        self.id = worker_id
        self.ds = shard
        self.cfg = cfg
        # Each worker has its own copy of weights
        self.w = torch.zeros(cfg.d, device=cfg.device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.w], lr=cfg.lr)

    def get_weights(self):
        """Return current weights (for evaluation)"""
        return self.w.detach().cpu()

    def compute_grad(self):
        """Compute gradient on local batch"""
        X, y = ray.get(self.ds.sample_batch.remote(self.cfg.batch))
        sleep_heterogeneity(
            self.id,
            self.cfg.hetero_base,
            self.cfg.hetero_jitter,
            self.cfg.hetero_straggler_every,
        )
        g = grad_bce(X, self.w, y).cpu()
        return g

    def apply_grad(self, grad_avg: torch.Tensor):
        """Apply averaged gradient to local weights"""
        self.optimizer.zero_grad()
        self.w.grad = grad_avg.to(self.cfg.device)
        self.optimizer.step()


# --------------- ASGD: parameter server applies immediately ---------------


@ray.remote
class ASGDParameterServer:
    def __init__(self, d: int, lr: float, device: str, eval_every: int = 5):
        self.w = torch.zeros(d, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.w], lr=lr)
        self.device = device
        self.num_updates = 0
        self.eval_every = eval_every
        self.loss_history: list[tuple[int, float]] = []  # Track (update_count, loss) pairs
        self.X_full = None
        self.y_full = None

    def set_full_dataset(self, X: torch.Tensor, y: torch.Tensor):
        """Set full dataset for periodic evaluation"""
        self.X_full = X
        self.y_full = y

    def get_weights(self):
        return self.w.detach().cpu()

    def get_num_updates(self):
        return self.num_updates

    def get_loss_history(self):
        return self.loss_history

    def push_grad(self, grad: torch.Tensor):
        # apply immediately
        self.optimizer.zero_grad()
        self.w.grad = grad.to(self.device)
        self.optimizer.step()
        self.num_updates += 1

        # Periodic evaluation
        if self.X_full is not None and self.y_full is not None:
            if self.num_updates % self.eval_every == 0 or self.num_updates == 1:
                with torch.no_grad():
                    loss = float(F.binary_cross_entropy_with_logits(self.X_full @ self.w, self.y_full))
                self.loss_history.append((self.num_updates, loss))

        # return current model version index
        return self.num_updates

    def evaluate_and_store(self, X: torch.Tensor, y: torch.Tensor):
        """Evaluate loss on full dataset and store in history"""
        with torch.no_grad():
            loss = float(F.binary_cross_entropy_with_logits(X @ self.w, y))
        self.loss_history.append((self.num_updates, loss))
        return loss


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
        metrics_data = []
        for i in range(iterations):
            t0 = time.time()
            w = ray.get(self.ps.get_weights.remote()).to(self.cfg.device)
            X, y = ray.get(self.ds.sample_batch.remote(self.cfg.batch))
            sleep_heterogeneity(
                self.id,
                self.cfg.hetero_base,
                self.cfg.hetero_jitter,
                self.cfg.hetero_straggler_every,
                step=self.local_step,
            )
            g = grad_bce(X, w, y).cpu()
            # submit grad and get global update index
            ver = ray.get(self.ps.push_grad.remote(g))
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


# --------------- SSP: staleness-bounded async via permits ---------------


@ray.remote
class SSPController:
    def __init__(self, d: int, lr: float, device: str, staleness: int, eval_every: int = 5):
        self.w = torch.zeros(d, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.w], lr=lr)
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
        self.X_full = X
        self.y_full = y

    def register_worker(self, wid: int):
        self.worker_steps[wid] = -1  # -1 means no steps completed yet
        logger.debug(f"SSP: Registered worker {wid}")

    def get_weights(self):
        return self.w.detach().cpu(), self.global_updates

    def get_loss_history(self):
        return self.loss_history

    def evaluate_and_store(self, X: torch.Tensor, y: torch.Tensor):
        """Evaluate loss on full dataset and store in history"""
        with torch.no_grad():
            loss = float(F.binary_cross_entropy_with_logits(X @ self.w, y))
        self.loss_history.append((self.global_updates, loss))
        return loss

    def can_proceed(self, wid: int, step_i: int):
        # SSP rule: fast workers can be at most s ahead of slowest worker
        # IMPORTANT: Non-blocking - returns immediately to avoid actor deadlock

        # Get min of ALL workers' COMPLETED steps
        min_step = min(self.worker_steps.values(), default=0)
        staleness = step_i - min_step

        # Can proceed if this step is at most s ahead of the slowest worker
        can_go = staleness <= self.staleness

        if can_go:
            logger.debug(f"SSP: Worker {wid} step {step_i} CAN PROCEED (min={min_step}, staleness={staleness})")
        else:
            logger.debug(f"SSP: Worker {wid} step {step_i} BLOCKED (min={min_step}, staleness={staleness} > bound={self.staleness})")

        return can_go, staleness if can_go else 0

    def push_grad(self, wid: int, step_i: int, grad: torch.Tensor):
        # Apply gradient immediately (like ASGD)
        self.optimizer.zero_grad()
        self.w.grad = grad.to(self.device)
        self.optimizer.step()
        self.global_updates += 1

        # Periodic evaluation
        if self.X_full is not None and self.y_full is not None:
            if self.global_updates % self.eval_every == 0 or self.global_updates == 1:
                with torch.no_grad():
                    loss = float(F.binary_cross_entropy_with_logits(self.X_full @ self.w, self.y_full))
                self.loss_history.append((self.global_updates, loss))

        # Mark this step as COMPLETED after gradient is applied
        self.worker_steps[wid] = step_i
        logger.debug(f"SSP: Worker {wid} completed step {step_i}, global_updates now {self.global_updates}")
        return self.global_updates


@ray.remote
class SSPWorker:
    def __init__(self, worker_id: int, ctrl, shard, cfg: TrainConfig):
        self.id = worker_id
        self.ctrl = ctrl
        self.ds = shard
        self.cfg = cfg
        self.local_step = 0

    def loop(self, iterations: int):
        """Worker loop that returns iteration metrics for aggregation."""
        metrics_data = []
        logger.debug(f"SSP: Worker {self.id} starting loop for {iterations} iterations")
        for i in range(iterations):
            t0 = time.time()
            # Poll until we get permission (non-blocking on actor side)
            logger.debug(f"SSP: Worker {self.id} requesting permission for step {self.local_step}")
            while True:
                can_go, staleness = ray.get(self.ctrl.can_proceed.remote(self.id, self.local_step))
                if can_go:
                    break
                # Backoff before retrying
                time.sleep(0.001)

            # Fetch current weights and compute gradient
            (w, global_step) = ray.get(self.ctrl.get_weights.remote())
            w = w.to(self.cfg.device)
            X, y = ray.get(self.ds.sample_batch.remote(self.cfg.batch))
            sleep_heterogeneity(
                self.id,
                self.cfg.hetero_base,
                self.cfg.hetero_jitter,
                self.cfg.hetero_straggler_every,
                step=self.local_step,
            )
            g = grad_bce(X, w, y).cpu()

            # Push gradient (marks step as completed)
            ver = ray.get(self.ctrl.push_grad.remote(self.id, self.local_step, g))
            iter_time = time.time() - t0
            logger.debug(f"SSP: Worker {self.id} completed step {self.local_step}, staleness={staleness}")
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


# --------------- LocalSGD: K local steps then average parameters ---------------


@ray.remote
class LocalAverager:
    def __init__(self, d: int):
        self.d = d
        self.last_broadcast = torch.zeros(d)

    def average_and_broadcast(self, weights: list[torch.Tensor]):
        W = torch.stack(weights, dim=0).mean(dim=0)
        self.last_broadcast = W.clone()
        return W


@ray.remote
class LocalWorker:
    def __init__(
        self,
        worker_id: int,
        shard,
        cfg: TrainConfig,
        initial_w: Optional[torch.Tensor] = None,
    ):
        self.id = worker_id
        self.ds = shard
        self.cfg = cfg
        self.w = torch.zeros(cfg.d, device=cfg.device, requires_grad=True) if initial_w is None else initial_w.to(cfg.device).clone()
        self.w.requires_grad = True
        self.optimizer = torch.optim.Adam([self.w], lr=cfg.lr)

    def local_steps(self, K: int) -> torch.Tensor:
        for k in range(K):
            X, y = ray.get(self.ds.sample_batch.remote(self.cfg.batch))
            sleep_heterogeneity(
                self.id,
                self.cfg.hetero_base,
                self.cfg.hetero_jitter,
                self.cfg.hetero_straggler_every,
                step=k,
            )
            g = grad_bce(X, self.w, y)
            self.optimizer.zero_grad()
            self.w.grad = g
            self.optimizer.step()
        return self.w.detach().cpu()

    def set_weights(self, w_new: torch.Tensor):
        self.w.data = w_new.to(self.cfg.device).clone()
        # Reset optimizer state after weight update
        # self.optimizer = torch.optim.Adam([self.w], lr=self.cfg.lr)

    def current_weights(self):
        return self.w.detach().cpu()


# --------------------------- Runner ---------------------------


def run_ssgd_centralized(
    cfg: TrainConfig,
    shards: list,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    metrics: Optional[MetricsCollector] = None,
):
    """SSGD with centralized coordinator (parameter server pattern)

    Each sync round = 1 global update, so cfg.total_updates rounds
    """
    if metrics:
        metrics.start_training()

    num_workers = len(shards)
    grad_size_bytes = cfg.d * 4  # float32 = 4 bytes
    # Communication: workers -> coordinator (gradients) + coordinator -> workers (weights)
    # For centralized: 2 * num_workers * grad_size per update
    comm_bytes_per_update = 2 * num_workers * grad_size_bytes

    coord = SSGDCoordinator.remote(cfg.d, cfg.lr, cfg.device)
    workers = [SSGDWorker.remote(i, coord, shards[i], cfg) for i in range(num_workers)]

    for t in range(cfg.total_updates):
        grads = ray.get([w.step_once.remote() for w in workers])
        w_new, step = ray.get(coord.aggregate_and_step.remote(grads))

        # Evaluate and record loss
        should_eval = (t % cfg.eval_every == 0) or (t == cfg.total_updates - 1)
        if should_eval:
            loss = float(F.binary_cross_entropy_with_logits(X_full @ w_new, y_full))
            print(f"[SSGD-Centralized] step={step:4d} loss={loss:.5f}")

            if metrics:
                # Count communication for all updates since last eval
                steps_since_last_eval = cfg.eval_every if t > 0 else (t + 1)
                if t == cfg.total_updates - 1 and t % cfg.eval_every != 0:
                    steps_since_last_eval = t % cfg.eval_every + 1
                comm_bytes = comm_bytes_per_update * steps_since_last_eval
                metrics.record_round(step, loss, comm_bytes=comm_bytes)

    if metrics:
        metrics.stop_training()
        # Record final if not already recorded
        if cfg.total_updates % cfg.eval_every != 0:
            w_new = ray.get(coord.get_weights.remote())
            loss = float(F.binary_cross_entropy_with_logits(X_full @ w_new, y_full))
            metrics.record_final(cfg.total_updates, loss, cfg.total_updates)


def run_ssgd_allreduce(
    cfg: TrainConfig,
    shards: list,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    metrics: Optional[MetricsCollector] = None,
):
    """SSGD with AllReduce (decentralized gradient averaging)

    Each sync round = 1 global update, so cfg.total_updates rounds
    """
    if metrics:
        metrics.start_training()

    num_workers = len(shards)
    grad_size_bytes = cfg.d * 4  # float32 = 4 bytes
    # AllReduce: Ring algorithm sends 2 * (N-1) / N * grad_size per worker
    # Total communication: 2 * (num_workers - 1) * grad_size_bytes
    comm_bytes_per_update = 2 * (num_workers - 1) * grad_size_bytes

    orch = AllReduceOrchestrator.remote()
    workers = [SSGDAllReduceWorker.remote(i, shards[i], cfg) for i in range(len(shards))]

    for t in range(cfg.total_updates):
        # All workers compute gradients on their local weights
        grads = ray.get([w.compute_grad.remote() for w in workers])

        # AllReduce: average gradients across workers
        avg_grad = ray.get(orch.allreduce_grads.remote(grads))

        # All workers apply the same averaged gradient
        ray.get([w.apply_grad.remote(avg_grad) for w in workers])

        # Evaluate and record loss
        should_eval = (t % cfg.eval_every == 0) or (t == cfg.total_updates - 1)
        if should_eval:
            w = ray.get(workers[0].get_weights.remote())
            loss = float(F.binary_cross_entropy_with_logits(X_full @ w, y_full))
            print(f"[SSGD-AllReduce] step={t+1:4d} loss={loss:.5f}")

            if metrics:
                # Count communication for all updates since last eval
                if t == 0:
                    steps_since_last_eval = 1
                elif t == cfg.total_updates - 1 and t % cfg.eval_every != 0:
                    steps_since_last_eval = (t % cfg.eval_every) + 1
                else:
                    steps_since_last_eval = cfg.eval_every
                comm_bytes = comm_bytes_per_update * steps_since_last_eval
                metrics.record_round(t + 1, loss, comm_bytes=comm_bytes)

    if metrics:
        metrics.stop_training()
        # Record final if not already recorded
        if cfg.total_updates % cfg.eval_every != 0:
            w = ray.get(workers[0].get_weights.remote())
            loss = float(F.binary_cross_entropy_with_logits(X_full @ w, y_full))
            metrics.record_final(cfg.total_updates, loss, cfg.total_updates)


def run_ssgd(
    cfg: TrainConfig,
    shards: list,
    X_full: torch.Tensor,
    y_full: torch.Tensor,
    sync_method: str = "centralized",
    metrics: Optional[MetricsCollector] = None,
):
    """Dispatcher for SSGD sync methods"""
    if sync_method == "centralized":
        run_ssgd_centralized(cfg, shards, X_full, y_full, metrics)
    else:  # allreduce
        run_ssgd_allreduce(cfg, shards, X_full, y_full, metrics)


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
    # Set full dataset for periodic evaluation
    ray.get(ps.set_full_dataset.remote(X_full.to("cpu"), y_full.to("cpu")))
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
        w = ray.get(ps.get_weights.remote())

        # Evaluate loss periodically during training (evaluate from parameter server's history if available)
        loss_history = ray.get(ps.get_loss_history.remote())
        if loss_history:
            for update_count, loss in loss_history:
                metrics.record_loss(update_count, loss, comm_bytes=0)

        # Final evaluation
        final_loss = float(F.binary_cross_entropy_with_logits(X_full @ w, y_full))
        metrics.record_final(final_updates, final_loss, final_updates)
        metrics.stop_training()

        print(f"[ASGD] updates={final_updates} loss={final_loss:.5f}")
    else:
        w = ray.get(ps.get_weights.remote())
        loss = float(F.binary_cross_entropy_with_logits(X_full @ w, y_full))
        final_updates = ray.get(ps.get_num_updates.remote())
        print(f"[ASGD] updates={final_updates} loss={loss:.5f}")

    for s in stats:
        avg_staleness = float(np.mean([m["staleness"] for m in s["metrics_data"]]))
        print(f"  worker {s['worker']}: steps={s['local_steps']} avg_staleness≈{avg_staleness:.2f}")


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
    # Set full dataset for periodic evaluation
    ray.get(ctrl.set_full_dataset.remote(X_full.to("cpu"), y_full.to("cpu")))
    workers = [SSPWorker.remote(i, ctrl, shards[i], cfg) for i in range(num_workers)]

    # Register all workers upfront to avoid race conditions
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
        w, ver = ray.get(ctrl.get_weights.remote())
        final_loss = float(F.binary_cross_entropy_with_logits(X_full @ w, y_full))
        metrics.record_final(ver, final_loss, ver)
        metrics.stop_training()

        logger.debug(f"SSP: Training complete, final version={ver}, loss={final_loss:.5f}")
        print(f"[SSP s={ssp_s}] updates={ver} loss={final_loss:.5f}")
    else:
        w, ver = ray.get(ctrl.get_weights.remote())
        loss = float(F.binary_cross_entropy_with_logits(X_full @ w, y_full))
        logger.debug(f"SSP: Training complete, final version={ver}, loss={loss:.5f}")
        print(f"[SSP s={ssp_s}] updates={ver} loss={loss:.5f}")

    for s in stats:
        avg_staleness = float(np.mean([m["staleness"] for m in s["metrics_data"]]))
        print(f"  worker {s['worker']}: steps={s['local_steps']} avg_staleness≈{avg_staleness:.2f}")


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

    avg = LocalAverager.remote(cfg.d)
    workers = [LocalWorker.remote(i, shards[i], cfg) for i in range(len(shards))]

    w_global = torch.zeros(cfg.d)
    num_rounds = math.ceil(cfg.total_updates / K)
    for r in range(num_rounds):
        # local K steps in parallel
        local_ws = ray.get([w.local_steps.remote(K) for w in workers])
        # average and broadcast
        w_global = ray.get(avg.average_and_broadcast.remote(local_ws))
        for w in workers:
            w.set_weights.remote(w_global)

        # Evaluate and record loss
        step = (r + 1) * K  # Approximate global step
        should_eval = (r % 2 == 0) or (r == num_rounds - 1)
        if should_eval:
            loss = float(F.binary_cross_entropy_with_logits(X_full @ w_global, y_full))
            print(f"[LocalSGD K={K}] round={r:3d} updates≈{step:4d} loss={loss:.5f}")

            if metrics:
                # Count communication for this round (and previous rounds if eval_every > 1)
                rounds_since_last_eval = 2 if r > 0 else (r + 1)
                if r == num_rounds - 1 and r % 2 != 0:
                    rounds_since_last_eval = r % 2 + 1
                comm_bytes = comm_bytes_per_round * rounds_since_last_eval
                metrics.record_round(step, loss, comm_bytes=comm_bytes)

    if metrics:
        metrics.stop_training()
        # Record final if not already recorded
        if num_rounds % 2 != 0:
            final_step = num_rounds * K
            final_loss = float(F.binary_cross_entropy_with_logits(X_full @ w_global, y_full))
            metrics.record_final(final_step, final_loss, cfg.total_updates)


# --------------------------- Main ---------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ssgd", "asgd", "ssp", "localsgd"], required=True)
    parser.add_argument(
        "--sync-method",
        choices=["centralized", "allreduce"],
        default="centralized",
        help="Sync method for SSGD: centralized (coordinator) or allreduce (peer-to-peer)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--total-updates", type=int, default=100, help="Target total parameter updates across all methods")
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ssp-staleness", type=int, default=2)
    parser.add_argument("--local-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hetero-base", type=float, default=0.0, help="Worker delay baseline in seconds (default: 0.0 = no delays)")
    parser.add_argument("--hetero-jitter", type=float, default=0.0, help="Worker delay jitter (default: 0.0)")
    parser.add_argument(
        "--hetero-straggler-every", type=int, default=0, help="Make worker 0 a straggler every N steps (default: 0 = disabled)"
    )
    parser.add_argument("--outdir", type=str, default="runs", help="Output directory for logs and metrics")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name (default: auto-generated)")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate loss every N steps")
    parser.add_argument("--log-every", type=int, default=1, help="Log metrics every N iterations")
    parser.add_argument("--no-logging", action="store_true", help="Disable metrics logging")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Data: generate and shard
    N = 200000
    noise = 0.0  # No label noise - should converge to ~0.1-0.2
    print(f"Creating synthetic data with N={N}, dim={args.dim}, noise={noise}, seed={args.seed}")
    X, y = make_synthetic_data(N, args.dim, noise=noise, seed=args.seed)
    actual_d = X.shape[1]  # Includes polynomial features (d+3)
    print(f"Feature dimension after polynomial expansion: {actual_d} (original: {args.dim})")
    shards = []
    per = N // args.num_workers
    print(f"Sharding data into {args.num_workers} shards, each with {per} samples")
    for i in range(args.num_workers):
        Xi = X[i * per : (i + 1) * per]
        yi = y[i * per : (i + 1) * per]
        shards.append(DatasetShard.remote(Xi, yi, device))

    cfg = TrainConfig(
        d=actual_d,
        lr=args.lr,
        batch=args.batch,
        total_updates=args.total_updates,
        device=device,
        hetero_base=args.hetero_base,
        hetero_jitter=args.hetero_jitter,
        hetero_straggler_every=args.hetero_straggler_every,
        eval_every=args.eval_every,
    )

    base_loss = float(F.binary_cross_entropy_with_logits(X @ torch.zeros(actual_d), y))
    print(f"Baseline loss (w=0): {base_loss:.5f} | Target: minimize via gradient descent")

    # Create metrics collector and run directory
    import json
    from datetime import datetime
    from pathlib import Path

    run_id = args.run_name
    if not run_id:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}-{args.mode}-w{args.num_workers}-d{args.dim}-u{args.total_updates}-s{args.seed}"

    outdir = Path(args.outdir)
    run_dir = outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = None
    if not args.no_logging:
        metrics = MetricsCollector(mode=args.mode, run_id=run_id)

    # Save config
    config_dict = {
        "mode": args.mode,
        "sync_method": args.sync_method,
        "num_workers": args.num_workers,
        "total_updates": args.total_updates,
        "dim": args.dim,
        "lr": args.lr,
        "batch": args.batch,
        "ssp_staleness": args.ssp_staleness,
        "local_k": args.local_k,
        "seed": args.seed,
        "device": args.device,
        "hetero_base": args.hetero_base,
        "hetero_jitter": args.hetero_jitter,
        "hetero_straggler_every": args.hetero_straggler_every,
        "outdir": args.outdir,
        "run_name": args.run_name,
        "eval_every": args.eval_every,
        "log_every": args.log_every,
        "no_logging": args.no_logging,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save metadata
    import socket
    import torch as torch_module

    meta_dict = {
        "run_id": run_id,
        "mode": args.mode,
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch_module.__version__,
        "device": args.device,
        "git_sha": None,  # Could add git info later
    }
    try:
        import ray as ray_module

        meta_dict["ray_version"] = ray_module.__version__
    except:
        pass

    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta_dict, f, indent=2)

    # Run training
    t0 = time.time()
    if args.mode == "ssgd":
        run_ssgd(cfg, shards, X, y, sync_method=args.sync_method, metrics=metrics)
    elif args.mode == "asgd":
        run_asgd(cfg, shards, X, y, metrics=metrics)
    elif args.mode == "ssp":
        run_ssp(cfg, shards, X, y, args.ssp_staleness, metrics=metrics)
    elif args.mode == "localsgd":
        run_localsgd(cfg, shards, X, y, args.local_k, metrics=metrics)
    dt = time.time() - t0
    print(f"Elapsed: {dt:.2f}s")

    # Write metrics
    if metrics:
        metrics.write_jsonl(run_dir / "events.jsonl")
        summary = metrics.get_summary()
        print(f"\nMetrics summary:")
        for key, value in summary.items():
            if key not in ("mode", "run_id"):
                print(f"  {key}: {value}")

    ray.shutdown()


if __name__ == "__main__":
    main()
