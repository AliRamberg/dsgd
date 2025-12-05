#!/usr/bin/env python3
"""Plot comparison graphs from distributed SGD experiment logs.

Usage:
    python plot.py runs/20251027-*                  # Plot all matching runs
    python plot.py runs/*/                          # Plot all runs
    python plot.py runs/run1/ runs/run2/            # Plot specific runs
    python plot.py --outdir comparison_plots runs/* # Custom output directory
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_run(run_path: Path) -> dict[str, Any]:
    """Load a single run's config, meta, and events."""
    run_path = Path(run_path)

    with open(run_path / "config.json") as f:
        config = json.load(f)

    with open(run_path / "meta.json") as f:
        meta = json.load(f)

    # Load events from JSONL
    events = []
    with open(run_path / "events.jsonl") as f:
        for line in f:
            events.append(json.loads(line))

    df = pd.DataFrame(events)

    # Get duration from meta.json (training_time_sec) if available,
    # otherwise fall back to calculating from event timestamps
    duration_sec = meta.get("training_time_sec")
    if duration_sec is None and not df.empty and "timestamp" in df.columns:
        timestamps = df["timestamp"].dropna()
        if len(timestamps) > 0:
            duration_sec = timestamps.max() - timestamps.min()

    return {
        "config": config,
        "meta": meta,
        "events": df,
        "path": run_path,
        "duration_sec": duration_sec,
    }


def plot_loss_vs_updates(runs: list[dict], outdir: Path):
    """Plot loss trajectories for all runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for run in runs:
        df = run["events"]
        config = run["config"]
        mode = config["mode"]

        # Skip empty runs
        if df.empty or "event_type" not in df.columns:
            continue

        # Extract loss events: use "loss", "round", or "final" events
        loss_events = df[df["event_type"].isin(["loss", "round", "final"])].copy()
        if loss_events.empty:
            continue

        # Sort by step
        loss_events = loss_events.sort_values("step")
        x = loss_events["step"].values
        y = loss_events["loss"].values

        # Create label based on mode and config
        if mode == "ssgd":
            sync = config.get("sync_method", "centralized")
            label = f"SSGD-{sync}"
        elif mode == "asgd":
            label = f"ASGD (w={config['num_workers']})"
        elif mode == "ssp":
            label = f"SSP (s={config['ssp_staleness']})"
        elif mode == "localsgd":
            label = f"LocalSGD (K={config['local_k']})"
        else:
            label = mode

        # Add duration to label if available
        if run.get("duration_sec") is not None:
            duration = run["duration_sec"]
            if duration < 60:
                label += f" ({duration:.1f}s)"
            else:
                minutes = int(duration // 60)
                seconds = duration % 60
                label += f" ({minutes}m {seconds:.1f}s)"

        ax.plot(x, y, marker="o", label=label, linewidth=2, markersize=4)

    ax.set_xlabel("Total Parameter Updates", fontsize=12)
    ax.set_ylabel("Loss (BCE)", fontsize=12)
    ax.set_title("Loss vs Updates: SGD Variant Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    outfile = outdir / "loss_vs_updates.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"📈 Saved: {outfile}")
    plt.close()


def plot_worker_throughput(runs: list[dict], outdir: Path):
    """Plot per-worker throughput (iterations/sec) for async methods."""
    async_runs = [r for r in runs if r["config"]["mode"] in ["asgd", "ssp"]]

    if not async_runs:
        print("⚠️  No async runs (ASGD/SSP) found for throughput plot")
        return

    fig, axes = plt.subplots(1, len(async_runs), figsize=(6 * len(async_runs), 5), squeeze=False)
    axes = axes[0]

    for idx, run in enumerate(async_runs):
        df = run["events"]
        config = run["config"]
        mode = config["mode"]

        # Get iteration events (new unified format)
        iter_df = df[df["event_type"] == "iteration"].copy()

        if iter_df.empty:
            print(f"⚠️  No iteration events found for {mode}")
            continue

        if mode == "asgd":
            title = f"ASGD (w={config['num_workers']})"
        else:  # ssp
            title = f"SSP (s={config['ssp_staleness']}, w={config['num_workers']})"

        # Add duration to title if available
        if run.get("duration_sec") is not None:
            duration = run["duration_sec"]
            if duration < 60:
                title += f" - {duration:.1f}s"
            else:
                minutes = int(duration // 60)
                seconds = duration % 60
                title += f" - {minutes}m {seconds:.1f}s"

        # Compute throughput per worker
        throughputs = []
        labels = []
        for worker_id in sorted(iter_df["worker_id"].dropna().unique()):
            worker_df = iter_df[iter_df["worker_id"] == worker_id]
            # Throughput = 1000 / avg_latency_ms
            if "latency_ms" in worker_df.columns and not worker_df["latency_ms"].isna().all():
                avg_latency = worker_df["latency_ms"].mean()
                throughput = 1000 / avg_latency  # iterations per second
                throughputs.append(throughput)
                labels.append(f"W{int(worker_id)}")

        if not throughputs:
            print(f"⚠️  No valid latency data for {mode}")
            continue

        ax = axes[idx]
        bars = ax.bar(labels, throughputs, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(throughputs))))
        ax.set_xlabel("Worker", fontsize=11)
        ax.set_ylabel("Iterations/sec", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, throughputs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    outfile = outdir / "worker_throughput.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"📈 Saved: {outfile}")
    plt.close()


def plot_staleness_analysis(runs: list[dict], outdir: Path):
    """Plot staleness distribution and time series for async methods."""
    async_runs = [r for r in runs if r["config"]["mode"] in ["asgd", "ssp"]]

    if not async_runs:
        print("⚠️  No async runs (ASGD/SSP) found for staleness plot")
        return

    fig, axes = plt.subplots(2, len(async_runs), figsize=(6 * len(async_runs), 10), squeeze=False)

    for idx, run in enumerate(async_runs):
        df = run["events"]
        config = run["config"]
        mode = config["mode"]

        # Get iteration events (new unified format)
        iter_df = df[df["event_type"] == "iteration"].copy()

        if iter_df.empty or "staleness" not in iter_df.columns:
            print(f"⚠️  No staleness data found for {mode}")
            continue

        if mode == "asgd":
            title = f"ASGD"
        else:  # ssp
            title = f"SSP (s={config['ssp_staleness']})"

        # Add duration to title if available
        if run.get("duration_sec") is not None:
            duration = run["duration_sec"]
            if duration < 60:
                title += f" - {duration:.1f}s"
            else:
                minutes = int(duration // 60)
                seconds = duration % 60
                title += f" - {minutes}m {seconds:.1f}s"

        # Filter out NaN staleness values
        iter_df = iter_df[iter_df["staleness"].notna()]

        if iter_df.empty:
            print(f"⚠️  No valid staleness data for {mode}")
            continue

        # Top plot: staleness distribution
        ax_dist = axes[0][idx]
        staleness_vals = iter_df["staleness"].values
        ax_dist.hist(staleness_vals, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
        ax_dist.axvline(staleness_vals.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean={staleness_vals.mean():.1f}")
        ax_dist.set_xlabel("Staleness", fontsize=11)
        ax_dist.set_ylabel("Count", fontsize=11)
        ax_dist.set_title(f"{title}: Staleness Distribution", fontsize=12, fontweight="bold")
        ax_dist.legend(fontsize=9)
        ax_dist.grid(True, alpha=0.3)

        # Bottom plot: overall max staleness at each evaluation step (epoch)
        ax_ts = axes[1][idx]

        # Get evaluation steps (where loss is recorded) - these are the "epochs"
        loss_df = df[df["event_type"].isin(["loss", "round", "final"])].copy()
        if loss_df.empty:
            print(f"⚠️  No loss evaluation events found for {mode}")
            continue

        eval_steps = sorted(loss_df["step"].unique())

        # Calculate maximum staleness at each evaluation step
        max_staleness_values = []
        eval_steps_plot = []

        prev_eval_step = 0
        for eval_step in eval_steps:
            # Get iterations between previous evaluation and current evaluation
            # This gives us the staleness status at this epoch
            iterations_in_epoch = iter_df[(iter_df["step"] > prev_eval_step) & (iter_df["step"] <= eval_step)]

            # If no iterations in this epoch, try to get iterations at or before this step
            if iterations_in_epoch.empty:
                iterations_in_epoch = iter_df[iter_df["step"] <= eval_step]

            if not iterations_in_epoch.empty and iterations_in_epoch["staleness"].notna().any():
                # Calculate maximum staleness across all workers in this epoch
                max_staleness = iterations_in_epoch["staleness"].max()
                max_staleness_values.append(max_staleness)
                eval_steps_plot.append(eval_step)

            prev_eval_step = eval_step

        if eval_steps_plot:
            # Plot only the maximum staleness line
            ax_ts.plot(
                eval_steps_plot, max_staleness_values, marker="o", label="Max Staleness", linewidth=2, markersize=6, color="steelblue"
            )

        ax_ts.set_xlabel("Evaluation Step (Epoch)", fontsize=11)
        ax_ts.set_ylabel("Max Staleness", fontsize=11)
        ax_ts.set_title(f"{title}: Overall Max Staleness Over Time", fontsize=12, fontweight="bold")
        ax_ts.legend(fontsize=9)
        ax_ts.grid(True, alpha=0.3)

    outfile = outdir / "staleness_analysis.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"📈 Saved: {outfile}")
    plt.close()


def plot_communication_bytes(runs: list[dict], outdir: Path):
    """Plot estimated communication bytes per update."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    bytes_per_update = []
    colors = []

    color_map = {
        "ssgd": "steelblue",
        "asgd": "darkorange",
        "ssp": "forestgreen",
        "localsgd": "purple",
    }

    for run in runs:
        df = run["events"]
        if 'comm_bytes' not in df.columns:
            df['comm_bytes'] = 0
        config = run["config"]
        mode = config["mode"]

        # Extract communication bytes from all events with comm_bytes field
        comm_events = df[df["comm_bytes"].notna()].copy()

        if comm_events.empty:
            continue

        # Calculate duration for this run
        duration_sec = run.get("duration_sec")
        duration_str = ""
        if duration_sec is not None:
            if duration_sec < 60:
                duration_str = f" ({duration_sec:.1f}s)"
            else:
                minutes = int(duration_sec // 60)
                seconds = duration_sec % 60
                duration_str = f" ({minutes}m {seconds:.1f}s)"

        if mode == "ssgd":
            sync = config.get("sync_method", "centralized")
            # For SSGD, comm_bytes is per round
            avg_bytes_per_round = comm_events["comm_bytes"].mean()
            # Per update: divide by _every (assuming comm is counted over eval_every steps)
            eval_every = config.get("eval_every", 5)
            avg_bytes_per_update = avg_bytes_per_round / eval_every
            labels.append(f"SSGD-{sync}{duration_str}")
            bytes_per_update.append(avg_bytes_per_update / 1e6)  # MB
            colors.append(color_map.get(mode, "gray"))
        elif mode == "asgd":
            # ASGD: comm_bytes is per iteration
            avg_bytes = comm_events["comm_bytes"].mean()
            labels.append(f"ASGD (w={config['num_workers']}){duration_str}")
            bytes_per_update.append(avg_bytes / 1e6)  # MB
            colors.append(color_map.get(mode, "gray"))
        elif mode == "ssp":
            # SSP: comm_bytes is per iteration
            avg_bytes = comm_events["comm_bytes"].mean()
            labels.append(f"SSP (s={config['ssp_staleness']}){duration_str}")
            bytes_per_update.append(avg_bytes / 1e6)  # MB
            colors.append(color_map.get(mode, "gray"))
        elif mode == "localsgd":
            # LocalSGD: comm_bytes is per round
            avg_bytes_per_round = comm_events["comm_bytes"].mean()
            K = config["local_k"]
            # Bytes per update = total bytes per round / K local steps
            avg_bytes_per_update = avg_bytes_per_round / K
            labels.append(f"LocalSGD (K={K}){duration_str}")
            bytes_per_update.append(avg_bytes_per_update / 1e6)  # MB per update
            colors.append(color_map.get(mode, "gray"))

    if not bytes_per_update:
        print("⚠️  No communication byte data found")
        return

    bars = ax.bar(labels, bytes_per_update, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Communication (MB per update)", fontsize=12)
    ax.set_title("Communication Overhead Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    # Add value labels on bars
    for bar, val in zip(bars, bytes_per_update):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    outfile = outdir / "communication_bytes.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"📈 Saved: {outfile}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot distributed SGD experiment comparisons")
    parser.add_argument("runs", nargs="+", type=str, help="Run directories to plot (supports globs)")
    parser.add_argument("--outdir", type=str, default="comparison_plots", help="Output directory for plots")
    args = parser.parse_args()

    # Expand globs and collect run directories
    run_paths = []
    for pattern in args.runs:
        path = Path(pattern)
        if path.is_dir() and (path / "events.jsonl").exists():
            run_paths.append(path)
        else:
            # Try glob expansion
            for match in Path(".").glob(str(path)):
                if match.is_dir() and (match / "events.jsonl").exists():
                    run_paths.append(match)

    if not run_paths:
        print("❌ No valid run directories found")
        print("   Each run directory must contain events.jsonl, config.json, and meta.json")
        return

    print(f"📂 Found {len(run_paths)} run(s):")
    for p in run_paths:
        print(f"   - {p}")

    # Load all runs
    runs = []
    for path in run_paths:
        try:
            run = load_run(path)
            runs.append(run)
            print(f"✅ Loaded: {run['meta']['run_id']}")
        except Exception as e:
            print(f"⚠️  Failed to load {path}: {e}")

    if not runs:
        print("❌ No runs loaded successfully")
        return

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 Generating plots in: {outdir}")

    # Generate all plots
    plot_loss_vs_updates(runs, outdir)
    plot_worker_throughput(runs, outdir)
    plot_staleness_analysis(runs, outdir)
    plot_communication_bytes(runs, outdir)

    print(f"\n✅ All plots generated successfully!")


if __name__ == "__main__":
    main()
