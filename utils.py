from __future__ import annotations
import random
import logging
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

def setup_actor_logging():
    """Configure logging for Ray actors (they run in separate processes).

    Ray actors run in separate processes. When we import logger from logger.py
    in an actor, the module-level configuration should execute, but we ensure
    it's properly set up by reconfiguring if needed (to handle any edge cases
    with Ray's process isolation).
    """
    # Import logger - this should execute the module-level config in logger.py
    from logger import logger as actor_logger

    # Ensure logger is configured (in case module-level code didn't run properly)
    # Clear handlers first to avoid duplicates
    if not actor_logger.handlers:

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        actor_logger.addHandler(handler)
        actor_logger.setLevel(logging.DEBUG)

    return actor_logger


def get_collective_backend(device: str) -> str:
    """Determine Ray collective backend: 'nccl' for CUDA GPUs, 'gloo' for CPU"""
    if device == "cuda" and torch.cuda.is_available():
        return "nccl"
    return "gloo"


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
