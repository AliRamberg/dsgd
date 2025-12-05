"""SSGD (Synchronous SGD) implementation using Ray Train TorchTrainer

This module contains the Ray Train-based SSGD implementation extracted from main.py
for better code organization and modularity.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from logger import logger
import torch
import torch.nn.functional as F
import ray
import ray.train
import ray.train.torch
import ray.data
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from config import TrainConfig
from data import SyntheticDataset
from utils import sleep_heterogeneity
from model import LinearModel

if TYPE_CHECKING:
    from metrics import MetricsCollector


def train_func_ssgd(config: TrainConfig):
    """Ray Train training function for SSGD"""

    # Get dataset shard
    dataset_shard = ray.train.get_dataset_shard("train")

    model = ray.train.torch.prepare_model(LinearModel(config.d))

    # Scale learning rate by number of workers (linear scaling rule for distributed training)
    # This compensates for the increased effective batch size when using more workers
    num_workers = ray.train.get_context().get_world_size()
    scaled_lr = config.lr * num_workers

    # Log scaled learning rate (only on rank 0 to avoid duplicate prints)
    logger.debug(f"[SSGD] Scaling LR: base_lr={config.lr:.4f} × num_workers={num_workers} = scaled_lr={scaled_lr:.4f}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=scaled_lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop: do total_updates epochs, processing all batches in each epoch
    step = 0

    for epoch in range(config.total_updates):
        # Create iterator for this epoch (Ray Data handles shuffling automatically)
        logger.debug(f"SSGD: Creating batch iterator for epoch {epoch}")
        batch_iterator = dataset_shard.iter_torch_batches(
            batch_size=config.batch_size,
            drop_last=True,
            device=ray.train.torch.get_device(),
        )

        # Accumulate evaluation loss during training (if evaluating this step)
        eval_total_loss = 0.0
        eval_total_samples = 0
        should_eval = (step == 0) or ((step + 1) % config.eval_every == 0)

        # Process all batches in this epoch
        model.train()
        for batch in batch_iterator:
            logger.debug(f"SSGD: Processing batch {step}")
            # Extract X and y from dict format (Ray Data returns dicts when using from_items)
            X_batch = batch["X"]
            y_batch = batch["y"]

            # Simulate heterogeneity
            # sleep_heterogeneity(
            #     ray.train.get_context().get_world_rank(),
            #     config.hetero_base,
            #     config.hetero_jitter,
            #     config.hetero_straggler_every,
            #     step=step,
            # )

            # Forward pass
            logits = model(X_batch).squeeze()
            loss = criterion(logits, y_batch)

            # Accumulate for evaluation (before model update, if evaluating)
            if should_eval:
                eval_total_loss += loss.item() * X_batch.shape[0]
                eval_total_samples += X_batch.shape[0]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ray.train.report(metrics={"train_loss": loss.item(), "step": step})
            step += 1

        # Report evaluation loss at end of epoch (if we were accumulating)
        if should_eval and eval_total_samples > 0:
            eval_loss = eval_total_loss / eval_total_samples
            ray.train.report(metrics={"eval_loss": eval_loss, "step": step})

        logger.debug(f"[SSGD] Completed epoch {epoch}/{config.total_updates}, total steps: {step}")


def run_ssgd(
    cfg: TrainConfig,
    num_workers: int,
    dataset: SyntheticDataset,
    metrics: Optional[MetricsCollector] = None,
):
    """SSGD using Ray Train TorchTrainer

    Args:
        cfg: Training configuration
        dataset: SyntheticDataset instance
        metrics: Optional metrics collector
    """
    if metrics:
        metrics.start_training()

    # Convert in-memory PyTorch tensors to Ray Data format
    # For in-memory data, from_numpy is the most efficient approach
    # Create separate datasets for features and labels, then combine using Dataset.zip()
    # Note: Using ray.data.Dataset.zip() method (not builtin zip function)
    X_np = dataset.X.detach().cpu().numpy()
    y_np = dataset.y.detach().cpu().numpy()

    ray_dataset = ray.data.from_numpy(X_np).zip(ray.data.from_numpy(y_np)).rename_columns(["X", "y"])

    # Configure scaling
    use_gpu = cfg.device == "cuda" and torch.cuda.is_available()
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # Create trainer
    trainer = TorchTrainer(
        train_func_ssgd,
        train_loop_config=cfg,
        scaling_config=scaling_config,
        datasets={"train": ray_dataset},
    )

    # Train
    result = trainer.fit()

    # Extract metrics and update our metrics collector
    if metrics:
        # Calculate communication bytes per step
        param_size_bytes = cfg.d * 4
        comm_bytes_per_step = param_size_bytes * num_workers

        # Get final loss
        final_loss = result.metrics.get("train_loss", 0.0)

        # Calculate initial loss
        initial_loss = float(F.binary_cross_entropy_with_logits(dataset.X @ torch.zeros(cfg.d), dataset.y))

        # Estimate batches per epoch (approximate)
        batches_per_epoch = len(dataset) // cfg.batch_size
        total_batches = cfg.total_updates * batches_per_epoch

        # Record loss at eval_every intervals
        # Since we sync after every batch in SSGD, record every eval_every steps
        for step in range(0, total_batches + 1, cfg.eval_every):
            if step == 0:
                metrics.record_loss(step, initial_loss, comm_bytes=0)
            elif step >= total_batches:
                metrics.record_loss(step, final_loss, comm_bytes=comm_bytes_per_step * cfg.eval_every)
            else:
                # Interpolate loss (approximation)
                progress = step / total_batches
                interpolated_loss = initial_loss * (1 - progress) + final_loss * progress
                metrics.record_loss(step, interpolated_loss, comm_bytes=comm_bytes_per_step * cfg.eval_every)

        metrics.record_final(total_batches, final_loss, total_batches)
        metrics.stop_training()
        print(f"[SSGD] total_batches={total_batches} loss={final_loss:.5f}")
