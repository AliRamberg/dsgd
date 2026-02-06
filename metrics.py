"""
Unified metrics collection layer for distributed SGD variants.

Collects metrics in-memory during training and writes to JSONL at the end
to minimize performance impact.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from enum import IntEnum
from pathlib import Path
from typing import Optional
from ray.util.metrics import Counter, Gauge, Histogram


class TrainingPhase(IntEnum):
    """Training phase indicator for anti-correlation visualization.

    Used with ray_train_phase gauge to show what phase each worker is in.
    """

    IDLE = 0  # Between steps, waiting
    FORWARD = 1  # Forward pass (GPU compute)
    BACKWARD = 2  # Backward pass + AllReduce (GPU compute + network)


@dataclass
class MetricEvent:
    """Unified event schema for all metrics."""

    event_type: str  # "loss", "iteration", "round", "final"
    step: int
    timestamp: float
    loss: Optional[float] = None
    staleness: Optional[float] = None
    comm_bytes: Optional[int] = None
    latency_ms: Optional[float] = None
    worker_id: Optional[int] = None
    mode: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, omitting None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class MetricsCollector:
    """In-memory metrics collector for distributed training."""

    def __init__(self, mode: str, run_id: str):
        self.mode = mode
        self.run_id = run_id
        self.events: list[MetricEvent] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Aggregated stats
        self.total_comm_bytes = 0
        self.loss_history: list[tuple[int, float]] = []  # (step, loss)

    def start_training(self):
        """Mark training start time."""
        self.start_time = time.time()

    def stop_training(self):
        """Mark training end time."""
        self.end_time = time.time()

    def record_loss(self, step: int, loss: float, comm_bytes: int = 0):
        """Record a loss evaluation event.

        Args:
            step: Global step/update count
            loss: Loss value
            comm_bytes: Communication bytes for this step (if applicable)
        """
        event = MetricEvent(
            event_type="loss",
            step=step,
            timestamp=time.time(),
            loss=loss,
            comm_bytes=comm_bytes if comm_bytes > 0 else None,
            mode=self.mode,
        )
        self.events.append(event)
        self.loss_history.append((step, loss))
        self.total_comm_bytes += comm_bytes

    def record_iteration(
        self,
        worker_id: int,
        step: int,
        latency_ms: float,
        staleness: float = 0,
        comm_bytes: int = 0,
    ):
        """Record a worker iteration event.

        Args:
            worker_id: Worker identifier
            step: Local iteration step
            latency_ms: Iteration latency in milliseconds
            staleness: Staleness value (for async methods)
            comm_bytes: Communication bytes for this iteration
        """
        event = MetricEvent(
            event_type="iteration",
            step=step,
            timestamp=time.time(),
            latency_ms=latency_ms,
            staleness=staleness if staleness > 0 else None,
            comm_bytes=comm_bytes if comm_bytes > 0 else None,
            worker_id=worker_id,
            mode=self.mode,
        )
        self.events.append(event)
        self.total_comm_bytes += comm_bytes

    def record_round(self, step: int, loss: float, comm_bytes: int = 0):
        """Record a synchronization round (for SSGD/LocalSGD).

        Args:
            step: Global step/round count
            loss: Loss value after this round
            comm_bytes: Communication bytes for this round
        """
        event = MetricEvent(
            event_type="round",
            step=step,
            timestamp=time.time(),
            loss=loss,
            comm_bytes=comm_bytes if comm_bytes > 0 else None,
            mode=self.mode,
        )
        self.events.append(event)
        self.loss_history.append((step, loss))
        self.total_comm_bytes += comm_bytes

    def record_final(self, step: int, loss: float, total_updates: int):
        """Record final training state.

        Args:
            step: Final step count
            loss: Final loss value
            total_updates: Total parameter updates
        """
        event = MetricEvent(
            event_type="final",
            step=step,
            timestamp=time.time(),
            loss=loss,
            mode=self.mode,
        )
        self.events.append(event)
        self.loss_history.append((step, loss))

    def get_summary(self) -> dict:
        """Return aggregated summary statistics."""
        if not self.events:
            return {}

        # Extract metrics by type
        iterations = [e for e in self.events if e.event_type == "iteration"]
        losses = [e for e in self.events if e.event_type in ("loss", "round")]

        summary = {
            "mode": self.mode,
            "run_id": self.run_id,
            "total_events": len(self.events),
            "total_comm_bytes": self.total_comm_bytes,
        }

        if self.start_time and self.end_time:
            summary["training_time_sec"] = self.end_time - self.start_time

        if losses:
            summary["final_loss"] = losses[-1].loss
            summary["initial_loss"] = losses[0].loss
            summary["num_loss_evaluations"] = len(losses)

        if iterations:
            latencies = [e.latency_ms for e in iterations if e.latency_ms is not None]
            if latencies:
                summary["avg_iteration_latency_ms"] = sum(latencies) / len(latencies)

            staleness_vals = [
                e.staleness for e in iterations if e.staleness is not None
            ]
            if staleness_vals:
                summary["avg_staleness"] = sum(staleness_vals) / len(staleness_vals)
                summary["max_staleness"] = max(staleness_vals)

        return summary

    def write_jsonl(self, path: Path):
        """Write all events to a JSONL file.

        Args:
            path: Path to output JSONL file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for event in self.events:
                # Add run_id to each event
                event_dict = event.to_dict()
                event_dict["run_id"] = self.run_id
                f.write(json.dumps(event_dict) + "\n")


class PrometheusMetricCollector:
    """Prometheus metrics collector using Ray's metrics API.

    Replaces file-based metrics with direct Prometheus export for Kubernetes deployments.
    """

    def __init__(
        self,
        mode: str,
        worker_id: Optional[int] = None,
        num_workers: int = 1,
        batch_size: int = 512,
        run_info: Optional[dict] = None,
    ):
        # Common tags for all metrics (remove high cardinality worker_id)
        self.mode = mode
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.base_tags = {
            "mode": mode,
            "num_workers": str(num_workers),
            "batch_size": str(batch_size),
        }

        # Run info metric - a gauge set to 1 with all run config as labels.
        # Standard Prometheus "info metric" pattern (like node_uname_info).
        # Allows joining run metadata onto any other metric via PromQL:
        #   ray_train_loss * on(mode) group_left(lr, device, ...) ray_train_run_info
        if run_info:
            info_tag_keys = tuple(sorted(run_info.keys()))
            self.run_info_gauge = Gauge(
                "ray_train_run_info",
                description="Run configuration metadata (info metric, always 1)",
                tag_keys=info_tag_keys,
            )
            info_tags = {k: str(v) for k, v in run_info.items()}
            self.run_info_gauge.set(1, info_tags)
        else:
            self.run_info_gauge = None

        # Core metrics - always available
        self.loss_hist = Histogram(
            "ray_train_loss",
            description="Training loss value histogram",
            boundaries=[
                0.01,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                1.0,
            ],
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.step_gauge = Gauge(
            "ray_train_worker_steps_total",
            description="Total steps completed by worker",
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.gradient_updates_counter = Counter(
            "ray_train_gradient_updates_total",
            description="Total gradient updates applied",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Performance metrics
        self.iteration_latency_hist = Histogram(
            "ray_train_iteration_latency_seconds",
            description="Per-iteration latency in seconds",
            boundaries=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Backward pass / AllReduce time metric (SSGD)
        # This captures the time spent in loss.backward() which includes gradient
        # computation AND AllReduce synchronization in DDP mode.
        # With NCCL_BLOCKING_WAIT=1, this is the primary indicator of network bottleneck.
        self.backward_time_hist = Histogram(
            "ray_train_backward_time_seconds",
            description="Time spent in backward pass including AllReduce (seconds)",
            boundaries=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.backward_time_gauge = Gauge(
            "ray_train_backward_time_current",
            description="Most recent backward pass time in seconds",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Communication metrics
        self.comm_bytes_counter = Counter(
            "ray_train_communication_bytes_total",
            description="Total communication bytes (AllReduce)",
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.comm_time_counter = Counter(
            "ray_train_communication_seconds_total",
            description="Total time spent in communication",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Training phase time counters for anti-correlation visualization
        # Use rate() on these to see fraction of time spent in each phase
        self.forward_time_counter = Counter(
            "ray_train_forward_seconds_total",
            description="Total time spent in forward pass (seconds)",
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.backward_time_total_counter = Counter(
            "ray_train_backward_seconds_total",
            description="Total time spent in backward pass including AllReduce (seconds)",
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.data_load_time_counter = Counter(
            "ray_train_data_load_seconds_total",
            description="Total time spent loading data (seconds)",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Staleness metrics (ASGD/SSP)
        self.staleness_gauge = Gauge(
            "ray_train_staleness_current",
            description="Current staleness value for async methods",
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.staleness_avg_gauge = Gauge(
            "ray_train_staleness_avg",
            description="Average staleness over recent window",
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.staleness_max_gauge = Gauge(
            "ray_train_staleness_max",
            description="Maximum staleness observed",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Synchronization metrics (SSGD)
        self.barrier_wait_counter = Counter(
            "ray_train_barrier_wait_seconds_total",
            description="Total time spent waiting at synchronization barriers",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # LocalSGD metrics
        self.localsgd_rounds_counter = Counter(
            "ray_train_localsgd_rounds_total",
            description="Total LocalSGD synchronization rounds completed",
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.localsgd_drift_gauge = Gauge(
            "ray_train_localsgd_drift",
            description="Gradient drift between workers (L2 norm)",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Samples processed
        self.samples_processed_counter = Counter(
            "ray_train_samples_processed_total",
            description="Total training samples processed",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Epoch progression - use Counter instead of Gauge for better Prometheus compatibility
        self.epoch_counter = Counter(
            "ray_train_epochs_completed_total",
            description="Total training epochs completed",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Training state
        self.start_time_gauge = Gauge(
            "ray_train_start_time",
            description="Unix timestamp when training started",
            tag_keys=("mode", "num_workers", "batch_size"),
        )
        self.end_time_gauge = Gauge(
            "ray_train_end_time",
            description="Unix timestamp when training ended",
            tag_keys=("mode", "num_workers", "batch_size"),
        )

        # Initialize core metrics to 0 so they appear in Prometheus immediately
        self.step_gauge.set(0, self.base_tags)

    def log_iteration(
        self, step: int, latency_ms: float, comm_bytes: int = 0, staleness: float = 0
    ):
        """Log iteration metrics from a worker."""
        self.step_gauge.set(step, self.base_tags)
        self.iteration_latency_hist.observe(
            latency_ms / 1000.0, self.base_tags
        )  # Convert to seconds
        self.gradient_updates_counter.inc(1, self.base_tags)

        if comm_bytes > 0:
            self.comm_bytes_counter.inc(comm_bytes, self.base_tags)

        if staleness > 0:
            self.staleness_gauge.set(staleness, self.base_tags)

    def log_loss(self, step: int, loss: float):
        """Log loss evaluation metrics."""
        self.step_gauge.set(step, self.base_tags)
        self.loss_hist.observe(loss, self.base_tags)

    def log_staleness_stats(self, avg_staleness: float, max_staleness: float):
        """Log aggregated staleness statistics (ASGD/SSP)."""
        self.staleness_avg_gauge.set(avg_staleness, self.base_tags)
        self.staleness_max_gauge.set(max_staleness, self.base_tags)

    def log_barrier_wait(self, wait_time_seconds: float):
        """Log time spent waiting at synchronization barrier (SSGD)."""
        self.barrier_wait_counter.inc(wait_time_seconds, self.base_tags)

    def log_backward_time(self, backward_time_seconds: float):
        """Log backward pass time including AllReduce synchronization.

        In DDP mode with NCCL_BLOCKING_WAIT=1, the backward pass includes:
        - Gradient computation on the GPU
        - AllReduce synchronization across all workers

        This metric is key for identifying network bottlenecks - when backward
        time is much larger than expected compute time, the network is saturated.

        Args:
            backward_time_seconds: Time for loss.backward() in seconds
        """
        self.backward_time_hist.observe(backward_time_seconds, self.base_tags)
        self.backward_time_gauge.set(backward_time_seconds, self.base_tags)

    def log_forward_time(self, forward_time_seconds: float):
        """Log time spent in forward pass.

        Args:
            forward_time_seconds: Duration of forward pass in seconds
        """
        self.forward_time_counter.inc(forward_time_seconds, self.base_tags)

    def log_backward_time_total(self, backward_time_seconds: float):
        """Log time spent in backward pass (cumulative counter).

        This complements backward_time_hist/gauge by providing a counter
        that works well with rate() for time-series visualization.

        Args:
            backward_time_seconds: Duration of backward pass in seconds
        """
        self.backward_time_total_counter.inc(backward_time_seconds, self.base_tags)

    def log_data_load_time(self, data_load_time_seconds: float):
        """Log time spent loading/preparing data.

        Args:
            data_load_time_seconds: Duration of data loading in seconds
        """
        self.data_load_time_counter.inc(data_load_time_seconds, self.base_tags)

    def log_localsgd_round(self, round_num: int, drift: float = 0):
        """Log LocalSGD synchronization round."""
        self.localsgd_rounds_counter.inc(1, self.base_tags)

        if drift > 0:
            self.localsgd_drift_gauge.set(drift, self.base_tags)

    def log_samples_processed(self, num_samples: int):
        """Log number of training samples processed."""
        self.samples_processed_counter.inc(num_samples, self.base_tags)

    def log_epoch(self, epoch: int):
        """Log epoch completion (increments counter at end of epoch)."""
        # Only increment when starting a new epoch (not on epoch 0)
        if epoch > 0:
            self.epoch_counter.inc(1, self.base_tags)

    def log_training_start(self):
        """Mark training start time."""
        import time

        self.start_time_gauge.set(time.time(), self.base_tags)

    def log_training_end(self):
        """Mark training end time."""
        import time

        self.end_time_gauge.set(time.time(), self.base_tags)
