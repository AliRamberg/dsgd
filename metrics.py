"""
Unified metrics collection layer for distributed SGD variants.

Collects metrics in-memory during training and writes to JSONL at the end
to minimize performance impact.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


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
            
            staleness_vals = [e.staleness for e in iterations if e.staleness is not None]
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

