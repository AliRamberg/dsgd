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
from ray.train.torch import TorchTrainer, TorchConfig

from config import TrainConfig
from data import SyntheticDataset
from model import LinearModel
from models import LinearBinaryClassifierWithPadding
from metrics import MetricsCollector, PrometheusMetricCollector
import utils


def train_func_ssgd(config: TrainConfig):
    """Ray Train training function for SSGD"""

    ctx = ray.train.get_context()
    worker_id = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    logger.info(f"SSGD: Worker {worker_id} of {world_size} workers")

    # Initialize Prometheus metrics collector with experiment configuration
    collector = PrometheusMetricCollector(
        mode="ssgd",
        worker_id=worker_id,
        num_workers=world_size,
        batch_size=config.batch_size,
        run_info=config.run_info,
    )

    # Estimate per-worker communication for DDP-style all-reduce (ring all-reduce).
    # Include gradient padding if enabled
    base_param_size_bytes = config.d * 4  # float32 params for real model
    padding_param_size_bytes = (
        config.gradient_padding_mb * 1024 * 1024
        if config.gradient_padding_mb > 0
        else 0
    )
    total_param_size_bytes = base_param_size_bytes + padding_param_size_bytes

    # Ring AllReduce: each worker sends/receives 2*(N-1)/N of total params
    comm_bytes_per_step = (
        int(2 * (world_size - 1) * total_param_size_bytes / world_size)
        if world_size > 1
        else 0
    )

    dataset_shard: ray.data.DataIterator = ray.train.get_dataset_shard("train")

    # Use padded model if gradient_padding_mb > 0
    if config.gradient_padding_mb > 0:
        model = ray.train.torch.prepare_model(
            LinearBinaryClassifierWithPadding(config.d, config.gradient_padding_mb),
            parallel_strategy_kwargs={
                "bucket_cap_mb": 1,  # 1MB bucket cap to force synchronous communication
            },
        )
    else:
        model = ray.train.torch.prepare_model(LinearModel(config.d))

    logger.info(f"Using padded model with ~{config.gradient_padding_mb}MB gradients")
    logger.debug(f"[SSGD] Using LR: {config.lr:.4f}")
    if comm_bytes_per_step > 0:
        logger.info(
            f"[SSGD] Expected communication per step: {comm_bytes_per_step / (1024 * 1024):.2f} MB "
            f"(base: {base_param_size_bytes / (1024 * 1024):.2f} MB, "
            f"padding: {padding_param_size_bytes / (1024 * 1024):.2f} MB)"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # SSGD logic: total_updates here means total epochs
    total_epochs = config.total_updates

    if worker_id == 0:
        logger.debug(f"SSGD: Starting training for {total_epochs} epochs")

    step = 0
    for epoch in range(total_epochs):
        # Log epoch progression (all workers)
        collector.log_epoch(epoch)

        if worker_id == 0:
            logger.debug(f"SSGD: Creating batch iterator for epoch {epoch}")

        batch_iterator = dataset_shard.iter_torch_batches(
            batch_size=config.batch_size,
            drop_last=True,
            device=ray.train.torch.get_device(),
            prefetch_batches=2,
        )

        logger.debug(f"SSGD: Batch iterator created for epoch {epoch}")

        model.train()
        batch_idx = 0
        for batch in batch_iterator:
            logger.debug(f"SSGD: Processing batch {batch_idx} for epoch {epoch}")
            t0 = time.perf_counter()

            # iter_torch_batches returns tensors directly
            X_batch = batch["X"]
            y_batch = batch["y"]

            # Ensure correct dtype
            if X_batch.dtype != torch.float32:
                X_batch = X_batch.to(torch.float32)
            if y_batch.dtype != torch.float32:
                y_batch = y_batch.to(torch.float32)

            # Add heterogeneity delay BEFORE computation to create worker speed differences
            # Fast workers finish computation early and wait (idle) during AllReduce
            # This enhances visibility of GPU idleness during synchronization barriers
            utils.sleep_heterogeneity(
                worker_id=worker_id,
                base=config.hetero_base,
                jitter=config.hetero_jitter,
                straggler_every=config.hetero_straggler_every,
                step=step,
            )

            # Forward pass (GPU compute) - measure time
            t_fwd_start = time.perf_counter()
            logits = model(X_batch).squeeze()
            loss = criterion(logits, y_batch)
            forward_time_seconds = time.perf_counter() - t_fwd_start
            collector.log_forward_time(forward_time_seconds)

            optimizer.zero_grad()

            # Backward pass + AllReduce (GPU compute + network)
            # In DDP, backward() synchronizes gradients via AllReduce.
            # With NCCL_BLOCKING_WAIT=1, the AllReduce operations block until complete.
            # With large gradient padding, this AllReduce will saturate the network
            # and cause GPUs to idle waiting for communication to complete.
            t_back_start = time.perf_counter()
            loss.backward()
            backward_time_seconds = time.perf_counter() - t_back_start
            collector.log_backward_time_total(backward_time_seconds)

            optimizer.step()

            latency_ms = (time.perf_counter() - t0) * 1000.0

            should_report = (step % config.eval_every) == 0

            # Log iteration metrics for all workers
            collector.log_iteration(
                step=step,
                latency_ms=latency_ms,
                comm_bytes=comm_bytes_per_step,
            )

            # Log backward pass time to histogram/gauge (for instant queries)
            collector.log_backward_time(backward_time_seconds)

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
            batch_idx += 1

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
        print(f"Loading Ray dataset from parquet: {dataset_path}")

        ray_dataset = ray.data.read_parquet(dataset_path)
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
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
    )

    torch_config = TorchConfig(
        backend=utils.get_collective_backend(cfg.device),
        timeout_s=1800,  # Long timeout to avoid premature failures
    )

    # Create and run trainer (no callbacks needed - metrics go directly to Prometheus)
    trainer = TorchTrainer(
        train_func_ssgd,
        train_loop_config=cfg,
        scaling_config=scaling_config,
        datasets={"train": ray_dataset},
        torch_config=torch_config,
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
