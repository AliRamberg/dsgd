# CLAUDE.md

Guidance for Claude Code when working with this distributed deep learning repository.

## Project Overview

**Active File**: `main.py` - A comprehensive showcase of 4 distributed SGD communication patterns using Ray.

**Focus**: Compare SSGD, ASGD, SSP, and LocalSGD on a single machine using Ray actors to simulate distributed workers.

**Philosophy**: Educational, self-contained, no external datasets required. ~580 lines.

## Repository Structure

```
ddp/
├── main.py          # Main showcase file (582 lines) - ACTIVE
├── CLAUDE.md        # This file
└── README.md        # Project documentation
```

**Note**: `main.py` is completely standalone with no dependencies on other Python files in this repository.

## main.py - Complete Architecture Explanation

### Purpose
Demonstrates 4 distributed SGD patterns on a **single machine** using Ray actors to simulate parameter servers, coordinators, and workers. Works on CPU or single GPU.

### Why This File Exists
- **Educational**: See differences in utilization, staleness, and convergence without needing a GPU cluster
- **Simulation**: Uses Ray Actors to emulate distributed systems
- **Comparison**: Run same problem with 4 different synchronization strategies
- **Heterogeneity**: Can simulate stragglers with configurable delays

---

## Core Components

### 1. Data Generation (`make_synthetic_data`, lines 60-77)

**Problem**: XOR classification with polynomial features

```python
# XOR pattern: y = (x[0] > 0) XOR (x[1] > 0)
# Raw features: X_raw ~ N(0,1), shape [n, d]
# Polynomial features: [X_raw, X_raw[:,:2]^2, X_raw[0]*X_raw[1]]
# This makes XOR linearly separable
```

**Key insight**: Raw linear model can't solve XOR. Adding quadratic and interaction terms (x₁², x₂², x₁·x₂) makes it learnable with linear classifier.

**Output**:
- `X_poly`: [n, d+3] features
- `y`: binary labels with optional noise

---

### 2. Gradient Computation (`grad_bce`, lines 80-85)

```python
def grad_bce(X, w, y):
    logits = X @ w
    probs = sigmoid(logits)
    return X.T @ (probs - y) / N
```

**Formula**: ∇L = (1/N) X^T (σ(Xw) - y)

This is the gradient of binary cross-entropy loss. All workers compute this on their local data shard.

---

### 3. Heterogeneity Simulation (`sleep_heterogeneity`, lines 89-94)

```python
delay = N(base, jitter)  # Normal distribution
if worker_id == 0 and step % straggler_every == 0:
    delay *= 5  # Make worker 0 a periodic straggler
time.sleep(delay)
```

**Purpose**: Simulate real-world worker speed differences
- **base**: baseline delay (default 0.0 = disabled)
- **jitter**: random variance
- **straggler_every**: make worker 0 slow every N steps

---

## Four Training Modes

### Mode 1: SSGD - Synchronous SGD (lines 125-169)

**Architecture**:
```
Coordinator (SSGDCoordinator)
    ├─ Maintains global weights w
    ├─ Has Adam optimizer
    └─ Waits for ALL workers each step

Workers (SSGDWorker) x N
    ├─ Pull weights from coordinator
    ├─ Compute gradient on local batch
    └─ Send gradient to coordinator
```

**Flow** (one step):
1. All workers pull current weights `w` from coordinator
2. Each worker computes gradient `g_i` on local batch
3. Coordinator waits for all gradients, averages: `g = mean(g_1, ..., g_N)`
4. Coordinator applies `g` with Adam optimizer: `w ← w - η·Adam(g)`
5. Repeat

**Characteristics**:
- ✅ **Deterministic**: Same result as single-GPU SGD with batch size = sum of worker batches
- ✅ **No staleness**: All workers use same weight version
- ❌ **Synchronization barrier**: Slowest worker determines speed
- ❌ **Idle time**: Fast workers wait for stragglers

**Implementation note**: Uses centralized parameter server pattern (coordinator holds weights).

---

### Mode 1b: SSGD AllReduce (lines 170-215)

**Architecture**:
```
AllReduceOrchestrator
    └─ Averages gradients (no weights stored)

Workers (SSGDAllReduceWorker) x N
    ├─ Each maintains own copy of weights w
    ├─ Each has own Adam optimizer
    └─ Compute gradient, participate in AllReduce
```

**Flow** (one step):
1. Each worker computes gradient `g_i` on local batch using **local weights**
2. AllReduce: gather all gradients, average, broadcast result: `g_avg = mean(g_1, ..., g_N)`
3. Each worker applies `g_avg` locally: `w_i ← w_i - η·Adam(g_avg)`
4. All workers now have **identical weights** (since they started identical and applied same gradient)

**Characteristics**:
- ✅ **Decentralized**: No single coordinator bottleneck
- ✅ **Same convergence** as centralized SSGD
- ✅ **Scales better**: Communication pattern is peer-to-peer (Ring AllReduce in practice)
- ❌ **Still synchronous**: Same barrier issues as centralized

**Key difference from centralized**:
- Centralized: workers send gradients to one coordinator
- AllReduce: workers exchange gradients in a ring pattern (simulated here)

**Note**: In real distributed training (PyTorch DDP), AllReduce is the standard because it avoids coordinator bottleneck.

---

### Mode 2: ASGD - Asynchronous SGD (lines 217-273)

**Architecture**:
```
ParameterServer (ASGDParameterServer)
    ├─ Maintains global weights w
    ├─ Has Adam optimizer
    └─ NO waiting - applies gradients immediately

Workers (ASGDWorker) x N
    └─ Run independent loops, no coordination
```

**Flow** (worker loop):
1. Worker pulls current weights `w` (version V)
2. Worker computes gradient `g` on local batch
3. *Meanwhile, other workers may have updated the server*
4. Worker pushes gradient `g` to server
5. Server **immediately** applies `g` (no waiting)
6. Server increments version: V → V+1
7. Worker repeats (no synchronization)

**Characteristics**:
- ✅ **No barriers**: Workers never wait for each other
- ✅ **High throughput**: Fast workers make more progress
- ✅ **Handles stragglers**: Slow workers don't block fast ones
- ❌ **Stale gradients**: Worker computes gradient on version V, but server may be at V+k when gradient arrives
- ❌ **Convergence issues**: High staleness can hurt or even prevent convergence
- ❌ **Non-deterministic**: Race conditions in gradient application order

**Staleness metric**:
```python
staleness = current_server_version - version_worker_used
```
If worker fetched weights at version 100, but server is at version 105 when gradient arrives, staleness = 5.

**When to use**: Large models where staleness impact is small, or when stragglers are severe.

---

### Mode 3: SSP - Stale Synchronous Parallel (lines 275-346)

**Architecture**:
```
SSPController
    ├─ Maintains global weights w
    ├─ Tracks each worker's step count
    └─ Enforces staleness bound s

Workers (SSPWorker) x N
    └─ Request permission before each step
```

**Flow** (worker step):
1. Worker requests permission: `can_proceed(worker_id, step_i)?`
2. Controller checks: `step_i - min(all_worker_steps) ≤ s`
   - If yes: grant permission
   - If no: block until slowest worker catches up
3. Worker pulls weights, computes gradient
4. Worker pushes gradient (applied immediately like ASGD)
5. Controller updates `worker_steps[worker_id] = step_i`
6. Repeat

**Characteristics**:
- ✅ **Bounded staleness**: Maximum staleness = s
- ✅ **Better than SSGD**: Fast workers can run ahead by s steps
- ✅ **Better than ASGD**: Prevents unbounded staleness
- ✅ **Convergence guarantees**: Theoretical convergence proofs exist
- ⚠️ **Partial synchronization**: Fast workers still block eventually
- ⚠️ **Staleness bound s**: Must tune s (higher = more async, lower = more sync)

**Example** (s=2, 4 workers):
```
Worker steps: [10, 10, 10, 8]
- Workers 0,1,2 at step 10, worker 3 at step 8
- Max allowed = min(8) + s = 8 + 2 = 10
- Workers 0,1,2 must WAIT until worker 3 reaches step 9
- Then max allowed = 9 + 2 = 11, workers can proceed
```

**When to use**: Balance between SSGD and ASGD. Common in parameter server systems.

**Tuning s**:
- s=0: equivalent to SSGD (fully synchronous)
- s=∞: equivalent to ASGD (fully asynchronous)
- s=1-5: practical range for most workloads

---

### Mode 4: LocalSGD (lines 349-403)

**Architecture**:
```
LocalAverager
    └─ Averages worker weights (not gradients!)

Workers (LocalWorker) x N
    ├─ Each has independent weights w_i
    ├─ Each has independent optimizer
    └─ Run K steps locally, then average weights
```

**Flow** (one round):
1. Each worker runs **K local steps** independently:
   ```
   for k in range(K):
       g_i = gradient(w_i, local_batch)
       w_i ← w_i - η·Adam(g_i)
   ```
2. After K steps, gather all worker weights: `[w_1, w_2, ..., w_N]`
3. Average weights: `w_global = mean(w_1, w_2, ..., w_N)`
4. Broadcast `w_global` to all workers (reset local weights)
5. Repeat

**Characteristics**:
- ✅ **Reduced communication**: Only sync every K steps (vs every step in SSGD)
- ✅ **Local exploration**: Workers explore different parts of loss surface
- ✅ **Handles heterogeneity**: Slow workers don't block during local steps
- ❌ **Gradient drift**: Workers diverge during K steps
- ❌ **Convergence**: May converge slower than SSGD for same number of gradient computations
- ❌ **Requires larger K**: Too small K = no benefit, too large K = poor convergence

**Key insight**: Instead of averaging **gradients** (SSGD), average **weights** (LocalSGD).

**Mathematics**:
- SSGD: `w ← w - η·mean(g_1, ..., g_N)` every step
- LocalSGD: Each worker `w_i ← w_i - η·g_i` for K steps, then `w ← mean(w_1, ..., w_N)`

**When to use**:
- Communication is expensive (slow network)
- Workers have heterogeneous speeds
- K typical values: 5-20 for CNNs, 50-100 for LLMs

**Typical configuration**:
```bash
python main.py --mode localsgd --num-workers 4 --steps 500 --local-k 10
# 500 steps = 50 rounds of K=10 local steps
```

---

## Configuration and Data Flow

### TrainConfig (lines 100-109)
```python
@dataclass
class TrainConfig:
    d: int = 200              # feature dimension
    lr: float = 0.05          # learning rate
    batch: int = 512          # batch size per worker
    steps: int = 100          # global steps
    device: str = "cpu"       # "cpu" or "cuda"
    hetero_base: float = 0.0  # worker delay baseline
    hetero_jitter: float = 0.0
    hetero_straggler_every: int = 0  # straggler frequency
```

### DatasetShard (lines 112-123)
Ray actor that holds a partition of the dataset in GPU/CPU memory. Each worker gets one shard.

```python
# Main script creates shards:
N = 200000 total samples
per = N // num_workers = 50000 per worker
shards = [DatasetShard(X[i*per:(i+1)*per], y[i*per:(i+1)*per]) for i in range(num_workers)]
```

**Method**: `sample_batch(batch)` - returns random batch from shard

---

## Running Examples

### Basic Usage
```bash
# Prerequisites
pip install ray torch

# SSGD centralized (parameter server)
python main.py --mode ssgd --num-workers 4 --steps 500

# SSGD AllReduce (decentralized)
python main.py --mode ssgd --sync-method allreduce --num-workers 4 --steps 500

# ASGD (fully asynchronous)
python main.py --mode asgd --num-workers 4 --steps 500

# SSP (bounded staleness s=2)
python main.py --mode ssp --num-workers 4 --steps 500 --ssp-staleness 2

# LocalSGD (sync every 5 steps)
python main.py --mode localsgd --num-workers 4 --steps 500 --local-k 5
```

### With Heterogeneity
```bash
# Add straggler every 10 steps (worker 0 becomes 5x slower)
python main.py --mode ssgd --num-workers 4 --steps 100 --hetero-straggler-every 10

# Compare with ASGD (should handle straggler better)
python main.py --mode asgd --num-workers 4 --steps 100 --hetero-straggler-every 10
```

### GPU Usage
```bash
# Use CUDA (requires single GPU, all workers share it in this simulation)
python main.py --mode ssgd --num-workers 4 --steps 500 --device cuda
```

---

## Key Metrics and Output

### What to Look For

**SSGD output**:
```
[SSGD-Centralized] step=   5 loss=0.45123
[SSGD-Centralized] step=  10 loss=0.38742
...
```
- Should converge smoothly
- Stragglers will slow down ALL workers

**ASGD output**:
```
[ASGD] updates=498 final_loss=0.25431
  worker 0: steps=125 avg_staleness≈2.34
  worker 1: steps=125 avg_staleness≈1.87
  worker 2: steps=125 avg_staleness≈2.11
  worker 3: steps=125 avg_staleness≈1.93
```
- **updates**: total gradient applications on server
- **steps**: how many local iterations each worker did
- **avg_staleness**: average version difference (higher = more stale gradients)

**SSP output**:
```
[SSP s=2] updates=498 final_loss=0.24892
  worker 0: steps=125 avg_window_staleness≈1.12
  worker 1: steps=125 avg_window_staleness≈0.98
```
- **staleness < s**: SSP constraint is working
- Compare to ASGD: should have lower staleness

**LocalSGD output**:
```
[LocalSGD K=5] round=  0 steps≈   5 loss=0.52341
[LocalSGD K=5] round=  2 steps≈  15 loss=0.43210
```
- **round**: how many sync rounds
- **steps**: total gradient computations ≈ round * K * num_workers

---

## Communication Patterns Summary

| Mode | Sync Type | Comm Freq | Staleness | Straggler Handling |
|------|-----------|-----------|-----------|-------------------|
| **SSGD** | Synchronous | Every step | 0 | ❌ Blocks all |
| **ASGD** | Asynchronous | Every step | Unbounded | ✅ No blocking |
| **SSP** | Bounded-async | Every step | ≤ s | ⚠️ Partial block |
| **LocalSGD** | Synchronous | Every K steps | 0 | ⚠️ Blocks at sync |

**Communication volume** (per step):
- SSGD: N gradients to coordinator (or AllReduce)
- ASGD: 1 gradient per worker push (async)
- SSP: 1 gradient per worker push (async with gate)
- LocalSGD: N weight vectors every K steps

**Bandwidth comparison**:
- SSGD: `(gradient_size * N) / step_time`
- LocalSGD: `(gradient_size * N) / (K * step_time)` ← K× reduction!

---

## Design Decisions

### Why Ray?
- Simulates distributed system on single machine
- Actors = natural parameter server / worker abstraction
- No need for actual cluster

### Why XOR with polynomial features?
- Linear XOR is impossible (not linearly separable)
- Adding x₁², x₂², x₁·x₂ makes it learnable
- Non-convex but solvable → demonstrates convergence differences

### Why Adam optimizer?
- Converges faster than vanilla SGD on this problem
- Shows realistic training scenario
- Adaptive learning rates help with staleness

### Why CPU by default?
- No GPU required to run
- Educational focus (not performance)
- Can enable `--device cuda` for realism

---

## Extending the Code

### Add New Synchronization Strategy
Create new Ray actors + runner function:

```python
@ray.remote
class MyCustomWorker:
    def __init__(self, worker_id, ...):
        self.id = worker_id
        # your state

    def step(self):
        # your logic
        pass

def run_mycustom(cfg, shards, X_full, y_full):
    workers = [MyCustomWorker.remote(i, shards[i], cfg) for i in range(len(shards))]
    # your coordination logic
```

Add to main:
```python
elif args.mode == "mycustom":
    run_mycustom(cfg, shards, X, y)
```

### Add Gradient Compression
Modify worker gradient computation:

```python
g = grad_bce(X, w, y)
g_compressed = top_k(g, k=0.1)  # keep top 10%
# send g_compressed instead of g
```

### Add Different Loss Function
Replace `grad_bce` with your gradient function:

```python
def grad_mse(X, w, y):
    pred = X @ w
    return X.T @ (pred - y) / N
```

---

## Common Issues

### Issue: "Ray actors not starting"
**Solution**: Check Ray initialization: `ray.init(ignore_reinit_error=True)`

### Issue: "Loss not converging"
**Reasons**:
- ASGD with high staleness → reduce workers or use SSP
- LocalSGD with large K → reduce K
- Learning rate too high → reduce `--lr`

### Issue: "Heterogeneity not visible"
**Solution**: Enable with `--hetero-straggler-every 10 --hetero-base 0.01`

### Issue: "CUDA out of memory"
**Reason**: All workers share single GPU in this simulation
**Solution**: Use CPU or reduce `--batch` size

---

## Important Reminders

✅ **Standalone**: main.py has no dependencies on other local Python files
✅ **Educational**: Compare 4 communication patterns side-by-side
✅ **Configurable**: All hyperparameters via CLI args
✅ **Hackable**: ~580 lines, easy to modify
✅ **No cluster needed**: Ray simulates distribution on single machine
❌ **Not production**: Missing checkpointing, fault tolerance, monitoring
❌ **Single machine**: Doesn't show true network bottlenecks
- always use --active with uv with ddp package