"""SSGD (Synchronous SGD) implementation using Ray Train TorchTrainer

This module contains the Ray Train-based SSGD implementation extracted from main.py
for better code organization and modularity.
"""

from __future__ import annotations
from typing import Optional
import time

from logger import logger
import torch
import ray
import ray.train
import ray.train.torch
import ray.data
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from config import TrainConfig
from data import SyntheticDataset
from model import LinearModel
from metrics import MetricsCollector, PrometheusMetricCollector


def train_func_ssgd(config: TrainConfig):
    """Ray Train training function for SSGD"""

    ctx = ray.train.get_context()
    worker_id = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    # Initialize Prometheus metrics collector
    collector = PrometheusMetricCollector(mode="ssgd")

    # Estimate per-worker communication for DDP-style all-reduce (ring all-reduce).
    param_size_bytes = config.d * 4  # float32 params
    comm_bytes_per_step = int(2 * (world_size - 1) * param_size_bytes / world_size) if world_size > 1 else 0

    dataset_shard = ray.train.get_dataset_shard("train")
    model = ray.train.torch.prepare_model(LinearModel(config.d))

    if worker_id == 0:
        logger.debug(f"[SSGD] Using LR: {config.lr:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    step = 0

    # SSGD logic: total_updates here means total epochs
    total_epochs = config.total_updates

    if worker_id == 0:
        logger.debug(f"SSGD: Starting training for {total_epochs} epochs")

    for epoch in range(total_epochs):
        if worker_id == 0:
            logger.debug(f"SSGD: Creating batch iterator for epoch {epoch}")

        # Recreate iterator for each epoch (required for Ray Data)
        batch_iterator = dataset_shard.iter_torch_batches(
            batch_size=config.batch_size,
            drop_last=True,
            device=ray.train.torch.get_device(),
        )

        model.train()
        epoch_steps = 0
        for batch in batch_iterator:
            t0 = time.perf_counter()

            X_batch = batch["X"]
            y_batch = batch["y"]

            logits = model(X_batch).squeeze()
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            latency_ms = (time.perf_counter() - t0) * 1000.0

            should_report = (step % config.eval_every) == 0

            # Log iteration metrics for all workers
            collector.log_iteration(worker_id=worker_id, latency_ms=latency_ms, comm_bytes=comm_bytes_per_step)

            # Log loss metrics for worker 0 (rank 0)
            if worker_id == 0:
                collector.step_gauge.set(step, collector.tags)
                if should_report:
                    collector.log_loss(step=step, loss=loss.item())
                    logger.debug(f"[SSGD] Step {step} loss: {loss.item():.6f}")

            step += 1

        if worker_id == 0:
            logger.debug(f"[SSGD] Completed epoch {epoch}/{total_epochs}, total_steps={step}")


# -----------------------------------------------------------------------------
# Driver function
# -----------------------------------------------------------------------------


def run_ssgd(
    cfg: TrainConfig,
    num_workers: int,
    dataset: SyntheticDataset,
    metrics: Optional[MetricsCollector] = None,
):
    """SSGD using Ray Train TorchTrainer"""
    if metrics:
        metrics.start_training()

    # Prepare Ray dataset
    X_np = dataset.X.detach().cpu().numpy()
    y_np = dataset.y.detach().cpu().numpy()
    ray_dataset = ray.data.from_numpy(X_np).zip(ray.data.from_numpy(y_np)).rename_columns(["X", "y"])

    # Scaling config
    use_gpu = cfg.device == "cuda" and torch.cuda.is_available()
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # Create and run trainer (no callbacks needed - metrics go directly to Prometheus)
    trainer = TorchTrainer(
        train_func_ssgd,
        train_loop_config=cfg,
        scaling_config=scaling_config,
        datasets={"train": ray_dataset},
    )

    result = trainer.fit()

    # Fallback: if we need file-based metrics for local testing, use result.metrics
    if metrics and result.metrics is not None:
        # Extract final metrics from Ray Train result
        # For SSGD, total_steps is roughly epochs * batches_per_epoch
        # Since we don't know exact batches here easily, we'll just use total_updates which now means epochs
        last_step = cfg.total_updates
        last_loss = float(result.metrics.get("eval_loss", result.metrics.get("train_loss", 0.0)))

        metrics.record_final(last_step, last_loss, last_step)
        metrics.stop_training()
        print(f"[SSGD] epochs={last_step} loss={last_loss:.5f}")
    else:
        print("[SSGD] Training completed - metrics exported to Prometheus")
