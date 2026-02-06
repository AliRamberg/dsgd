# distributed_sgd_showcase.py
"""
Distributed SGD showcase for four training modes on a single machine (CPU or a single GPU):
  1) SSGD (Synchronous SGD using Ray Train TorchTrainer)
  2) ASGD (Async SGD via parameter server - custom Ray Actors implementation)
  3) SSP  (Stale Synchronous Parallel with staleness bound s - custom Ray Actors)
  4) LocalSGD (K local steps then synchronize parameters - custom Ray Actors)

Why this file?
- Let you *see* differences in utilization, staleness, and convergence without a GPU cluster.
- SSGD uses Ray Train for production-ready distributed training (handles NCCL/Gloo automatically).
- ASGD/SSP/LocalSGD use custom Ray Actors to demonstrate alternative synchronization patterns.
- Simulates heterogeneity (stragglers) with sleep delays (disabled by default).

Prereqs (Python 3.9+):
    pip install ray torch

Run examples:
    python main.py --mode ssgd --num-workers 4 --total-updates 500
    python main.py --mode asgd --num-workers 4 --total-updates 500
    python main.py --mode ssp  --num-workers 4 --total-updates 500 --ssp-staleness 2
    python main.py --mode localsgd --num-workers 4 --total-updates 500 --local-k 5


Notes:
- Uses XOR classification with polynomial features (non-convex, learnable)
- Adam optimizer for better convergence than plain SGD
- Heterogeneity disabled by default; enable with --hetero-straggler-every N
- Works on CPU by default; use --device cuda for GPU
- SSGD automatically uses NCCL for GPU or Gloo for CPU (handled by Ray Train)
"""

from __future__ import annotations
import argparse
import time
import socket
import json
import boto3

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import torch
import torch.nn.functional as F
import ray

from metrics import MetricsCollector
from config import TrainConfig
from data import SyntheticDataset, DatasetShard
from utils import set_seed

# Import training functions from dedicated modules
from train_ssgd import run_ssgd
from train_asgd import run_asgd
from train_ssp import run_ssp
from train_localsgd import run_localsgd


# --------------------------- Main ---------------------------


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ssgd", "asgd", "ssp", "localsgd"], required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--total-updates", type=int, default=100, help="Target total parameter updates across all methods")
    parser.add_argument("--num-samples", type=int, default=200000, help="Number of samples in synthetic dataset (default: 200000)")
    parser.add_argument("--noise", type=float, default=0.0, help="Label noise probability (default: 0.0)")
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ssp-staleness", type=int, default=2)
    parser.add_argument("--local-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hetero-base", type=float, default=0.0, help="Worker delay baseline in seconds (default: 0.0 = no delays)")
    parser.add_argument("--hetero-jitter", type=float, default=0.0, help="Worker delay jitter (default: 0.0)")
    parser.add_argument("--hetero-straggler-every", type=int, default=0, help="Make worker 0 a straggler every N steps (default: 0 = disabled)")
    parser.add_argument("--gradient-padding-mb", type=int, default=0, help="Add dummy params (MB) for network bottleneck testing (0=disabled, 400=BERT-sized)")
    parser.add_argument("--outdir", type=str, default="runs", help="Output directory for logs and metrics")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name (default: auto-generated)")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate and log metrics every N steps")
    parser.add_argument("--no-logging", action="store_true", help="Disable metrics logging")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to pre-saved parquet dataset (default: generate in-memory)")
    # fmt: on
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    # Driver keeps data on CPU, Ray workers handle GPU placement
    driver_device = "cpu"

    ray.init(ignore_reinit_error=True, _metrics_export_port=8080)
    ray.data.DataContext.get_current().enable_rich_progress_bars = True
    ray.data.DataContext.get_current().use_ray_tqdm = False

    # Data: generate and shard
    if args.dataset_path:
        # Load from pre-saved parquet files (memory efficient for large dims)
        dataset_path = args.dataset_path
        print(f"Loading dataset from: {dataset_path}")

        # Read metadata - handle S3 paths
        if dataset_path.startswith("s3://"):
            parsed = urlparse(dataset_path)
            s3_bucket = parsed.netloc
            s3_key = parsed.path.lstrip("/") + "/metadata.json"

            s3_client = boto3.client("s3")
            obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            metadata = json.loads(obj["Body"].read().decode("utf-8"))
        else:
            # Local path
            local_path = Path(dataset_path)
            with open(local_path / "metadata.json") as f:
                metadata = json.load(f)

        actual_d = metadata["expanded_dim"]
        num_samples = metadata["num_samples"]

        print(f"  Samples: {num_samples:,}")
        print(f"  Base dimension: {metadata['base_dim']:,}")
        print(f"  Expanded dimension: {actual_d:,}")
        print(f"  Dataset size: {metadata['dataset_size_gb']:.2f} GB")
        print(f"  Model parameters: {actual_d + 1:,}")

        # For SSGD mode: use Ray Data to stream from disk
        if args.mode == "ssgd":
            dataset = None  # Will be loaded in run_ssgd with ray.data.read_parquet()
            X = None
            y = None
        else:
            raise NotImplementedError(
                f"--dataset-path only supported for SSGD mode currently (got: {args.mode})"
            )
    else:
        # Generate in-memory (original behavior)
        print(
            f"Creating synthetic data with N={args.num_samples}, dim={args.dim}, noise={args.noise}, seed={args.seed}"
        )
        dataset = SyntheticDataset(
            args.num_samples, args.dim, noise=args.noise, seed=args.seed
        )
        X = dataset.X.to(driver_device)
        y = dataset.y.to(driver_device)
        actual_d = X.shape[1]  # Includes polynomial features (d+3)
        print(
            f"Feature dimension after polynomial expansion: {actual_d} (original: {args.dim})"
        )

    # Only create shards for custom algorithms (ASGD, SSP, LocalSGD)
    # SSGD uses Ray Train which handles sharding automatically.
    # Shards are created on CPU (driver_device) because the head node has no GPU.
    # Workers move shard data to the target device inside their actor __init__.
    shards: list[DatasetShard] = []
    if args.mode != "ssgd":
        per = args.num_samples // args.num_workers
        print(f"Sharding data into {args.num_workers} shards, each with {per} samples")
        for i in range(args.num_workers):
            Xi = X[i * per : (i + 1) * per]
            yi = y[i * per : (i + 1) * per]
            shards.append(DatasetShard(Xi, yi, driver_device))

    print(f"Learning rate: {args.lr}")

    run_id = args.run_name
    if not run_id:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}-{args.mode}-w{args.num_workers}-d{args.dim}-u{args.total_updates}-s{args.seed}"

    # Build run_info dict for Prometheus info metric.
    # This creates a ray_train_run_info gauge=1 with all config as labels,
    # so you can join/filter any metric by run configuration in Grafana.
    run_info = {
        "mode": args.mode,
        "num_workers": str(args.num_workers),
        "lr": str(args.lr),
        "batch_size": str(args.batch),
        "total_updates": str(args.total_updates),
        "dim": str(args.dim),
        "num_samples": str(args.num_samples),
        "device": device,
        "seed": str(args.seed),
        "ssp_staleness": str(args.ssp_staleness),
        "local_k": str(args.local_k),
        "gradient_padding_mb": str(args.gradient_padding_mb),
        "run_id": run_id,
    }

    cfg = TrainConfig(
        d=actual_d,
        lr=args.lr,
        batch_size=args.batch,
        total_updates=args.total_updates,
        device=device,
        hetero_base=args.hetero_base,
        hetero_jitter=args.hetero_jitter,
        hetero_straggler_every=args.hetero_straggler_every,
        eval_every=args.eval_every,
        gradient_padding_mb=args.gradient_padding_mb,
        run_info=run_info,
    )

    if X is not None and y is not None:
        base_loss = float(
            F.binary_cross_entropy_with_logits(
                X @ torch.zeros(actual_d, device=driver_device), y
            )
        )
        print(
            f"Baseline loss (w=0): {base_loss:.5f} | Target: minimize via gradient descent"
        )
    else:
        print("Skipping baseline loss computation (streaming mode)")

    outdir = Path(args.outdir)
    run_dir = outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = None
    if not args.no_logging:
        metrics = MetricsCollector(mode=args.mode, run_id=run_id)

    # Save config
    config_dict = {
        "mode": args.mode,
        "num_workers": args.num_workers,
        "total_updates": args.total_updates,
        "num_samples": args.num_samples,
        "noise": args.noise,
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
        "no_logging": args.no_logging,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    meta_dict = {
        "run_id": run_id,
        "mode": args.mode,
        "hostname": socket.gethostname(),
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "torch_version": torch.__version__,
        "ray_version": ray.__version__,
    }

    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta_dict, f, indent=2)

    # Run training
    try:
        t0 = time.time()
        if args.mode == "ssgd":
            run_ssgd(
                cfg,
                args.num_workers,
                dataset,
                metrics=metrics,
                dataset_path=args.dataset_path,
            )
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
            print("\nMetrics summary:")
            for key, value in summary.items():
                if key not in ("mode", "run_id"):
                    print(f"  {key}: {value}")

            # Update meta.json with training_time_sec if available
            if "training_time_sec" in summary:
                with open(run_dir / "meta.json", "r") as f:
                    meta_dict = json.load(f)
                meta_dict["training_time_sec"] = summary["training_time_sec"]
                with open(run_dir / "meta.json", "w") as f:
                    json.dump(meta_dict, f, indent=2)
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
