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

    # Initialize Prometheus metrics collector with experiment configuration
    collector = PrometheusMetricCollector(
        mode="ssgd",
        worker_id=worker_id,
        num_workers=world_size,
        batch_size=config.batch_size,
    )

    # Estimate per-worker communication for DDP-style all-reduce (ring all-reduce).
    param_size_bytes = config.d * 4  # float32 params
    comm_bytes_per_step = (
        int(2 * (world_size - 1) * param_size_bytes / world_size)
        if world_size > 1
        else 0
    )

    dataset_shard = ray.train.get_dataset_shard("train")
    model = ray.train.torch.prepare_model(LinearModel(config.d))

    if worker_id == 0:
        logger.debug(f"[SSGD] Using LR: {config.lr:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    step = 0

    # SSGD logic: total_updates here means total epochs
    total_epochs = config.total_updates

    if worker_id == 0:
        logger.debug(f"SSGD: Starting training for {total_epochs} epochs")

    for epoch in range(total_epochs):
        # Log epoch progression (all workers)
        collector.log_epoch(epoch)

        if worker_id == 0:
            logger.debug(f"SSGD: Creating batch iterator for epoch {epoch}")

        # Recreate iterator for each epoch (required for Ray Data)
        # Use local_shuffle_buffer_size for streaming-compatible shuffling (avoids materializing entire dataset)
        batch_iterator = dataset_shard.iter_torch_batches(
            batch_size=config.batch_size,
            drop_last=True,
            device=ray.train.torch.get_device(),
            local_shuffle_buffer_size=config.batch_size
            * 4,  # Shuffle buffer = 4 batches
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

            # Measure backward pass duration
            # In DDP, this includes the time spent waiting for other workers to sync gradients.
            # Fast workers will have a longer backward time (high idle time),
            # while the slowest worker will have the shortest backward time (pure computation).
            t_back_start: float = time.perf_counter()
            loss.backward()
            backward_time_ms: float = (time.perf_counter() - t_back_start) * 1000.0

            optimizer.step()

            latency_ms = (time.perf_counter() - t0) * 1000.0

            should_report = (step % config.eval_every) == 0

            # Log iteration metrics for all workers
            # We use backward_time_ms as a proxy for "wait time" in the logs
            collector.log_iteration(
                step=step,
                latency_ms=latency_ms,
                comm_bytes=comm_bytes_per_step,
                # Pass backward time as "staleness" or a custom metric to visualize it
                # reusing the staleness field for visualization convenience in this demo
                staleness=backward_time_ms,
            )

            # Log loss metrics and report to Ray Train (all workers must participate)
            if should_report:
                # All workers must call ray.train.report() for synchronization
                loss_val = (
                    loss.item() if worker_id == 0 else 0.0
                )  # Only worker 0 has real loss
                ray.train.report({"loss": loss_val, "step": step})

                if worker_id == 0:
                    collector.log_loss(step=step, loss=loss.item())
                    logger.debug(f"[SSGD] Step {step} loss: {loss.item():.6f}")

            step += 1

        if worker_id == 0:
            logger.debug(
                f"[SSGD] Completed epoch {epoch}/{total_epochs}, total_steps={step}"
            )


# -----------------------------------------------------------------------------
# Driver function
# -----------------------------------------------------------------------------


def run_ssgd(
    cfg: TrainConfig,
    num_workers: int,
    dataset: Optional[SyntheticDataset],
    metrics: Optional[MetricsCollector] = None,
    dataset_path: Optional[str] = None,
):
    """SSGD using Ray Train TorchTrainer

    Args:
        cfg: Training configuration
        num_workers: Number of workers
        dataset: In-memory SyntheticDataset (if dataset_path is None)
        metrics: Optional metrics collector
        dataset_path: Path to pre-saved parquet dataset (memory efficient for large dims)
    """
    if metrics:
        metrics.start_training()

    # Prepare Ray dataset
    if dataset_path:
        # Load from disk or S3 (memory efficient)
        import torch

        # Ray Data automatically discovers all parquet files in directory
        print(f"Loading Ray dataset from parquet: {dataset_path}")
        ray_dataset = ray.data.read_parquet(dataset_path)

        # Convert list columns to proper numpy arrays using pandas format
        # X is stored as list-of-lists in parquet, pandas handles this well
        def convert_to_arrays(batch):
            import numpy as np

            # batch is a pandas DataFrame
            # batch["X"] is a Series of lists, convert to 2D array
            X_array = np.stack(batch["X"].values).astype(np.float32)
            y_array = batch["y"].values.astype(np.float32)
            return {"X": X_array, "y": y_array}

        # Note: Avoid .random_shuffle() as it materializes the entire dataset
        # Instead, use local_shuffle_buffer_size in iter_torch_batches for streaming shuffle
        ray_dataset = ray_dataset.map_batches(convert_to_arrays, batch_format="pandas")
    else:
        # Generate in-memory (original behavior)
        X_np = dataset.X.detach().cpu().numpy()
        y_np = dataset.y.detach().cpu().numpy()
        ray_dataset = (
            ray.data.from_numpy(X_np)
            .zip(ray.data.from_numpy(y_np))
            .rename_columns(["X", "y"])
        )

    # Scaling config
    # Note: Driver may not have GPU, but workers do. Trust the device config.
    use_gpu = cfg.device == "cuda"
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
        last_loss = float(
            result.metrics.get("eval_loss", result.metrics.get("train_loss", 0.0))
        )

        metrics.record_final(last_step, last_loss, last_step)
        metrics.stop_training()
        print(f"[SSGD] epochs={last_step} loss={last_loss:.5f}")
    else:
        print("[SSGD] Training completed - metrics exported to Prometheus")
