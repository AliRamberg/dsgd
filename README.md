# Distributed Deep Learning - Communication Patterns

Minimal, hackable code for understanding communication in distributed training.

**Philosophy**: Simple and direct. No abstractions, just working code you can read and modify.

## Quick Start

```bash
# Install dependencies
uv sync

# Single GPU training
python main.py

# Multi-GPU training (2 GPUs)
torchrun --nproc_per_node=2 main.py --ddp

# Custom model size
python main.py --channels 64 128 256 --hidden 2048

# With synthetic data (faster, no download)
python main.py --synthetic
```

## Project Structure

```
ddp/
├── main.py          # CLI entry point (127 lines)
├── train.py         # Training logic + configs (398 lines)
├── synthetic.py     # Synthetic data + benchmarks (332 lines)
└── README.md

Total: ~857 lines of code
```

## What's Included

### 1. Configurable CNN Model
- Specify conv layers: `--channels 32 64 128`
- Specify hidden size: `--hidden 512`
- No preset "tiny/small/large" - fully customizable

### 2. Training Modes
- **Single GPU**: `python main.py`
- **Multi-GPU DDP**: `torchrun --nproc_per_node=N main.py --ddp`
- **Communication profiling**: Tracks compute vs communication time

### 3. Synthetic Data
- No I/O bottleneck (generated on-device)
- Deterministic (fixed seeds)
- Fast iteration for testing communication patterns

### 4. AllReduce Bandwidth Benchmark
```bash
# Test raw NCCL bandwidth
torchrun --nproc_per_node=4 -m synthetic --bench

# Sweep multiple tensor sizes
torchrun --nproc_per_node=4 -m synthetic --bench --sweep
```

## Examples

### Example 1: Baseline Single GPU
```bash
python main.py
```

Output shows:
- Model parameters
- Training progress (loss, accuracy per epoch)
- Total training time

### Example 2: Multi-GPU with Communication Profiling
```bash
torchrun --nproc_per_node=2 main.py --ddp
```

Additional output:
- Compute time vs Communication time
- Communication overhead percentage

### Example 3: Scale Model Size
```bash
# Tiny model (~1M params)
python main.py --channels 16 32 --hidden 128

# Medium model (~5M params)
python main.py --channels 32 64 128 --hidden 512

# Large model (~20M params)
python main.py --channels 64 128 256 --hidden 2048
```

Observe how communication overhead decreases with larger models.

### Example 4: AllReduce Bandwidth Test
```bash
torchrun --nproc_per_node=4 -m synthetic --bench --size-mb 100
```

Measures:
- Raw NCCL bandwidth (GB/s per GPU)
- Latency per all-reduce (ms)
- Validates interconnect performance

## Understanding Communication Overhead

### Formula
```
Communication time ≈ (gradient_bytes * 2 * (N-1) / N) / bandwidth
```

For Ring AllReduce algorithm with N GPUs.

### Typical Results

| Model Size | Params | Gradient Size | Comm Overhead (10 Gbps) |
|-----------|---------|---------------|------------------------|
| Tiny      | 1M      | 4 MB         | ~60% (comm-bound)      |
| Small     | 5M      | 20 MB        | ~40% (balanced)        |
| Medium    | 20M     | 80 MB        | ~20% (compute-bound)   |

**Key Insight**: Larger models amortize communication cost better.

## Code Organization

### `train.py`
- `ModelConfig`: Configure conv channels, hidden size
- `TrainingConfig`: Configure epochs, LR, batch size
- `SimpleCNN`: Flexible CNN builder
- `train_single_gpu()`: Single GPU training loop
- `train_ddp()`: Multi-GPU DDP with profiling

### `synthetic.py`
- `SyntheticDataset`: Random X, target Y = X·w* + noise
- `benchmark_allreduce()`: Measure NCCL bandwidth
- `analyze_comm_vs_compute()`: Print communication analysis

### `main.py`
- CLI argument parsing
- Data loading (MNIST or synthetic)
- Calls training functions

## Hardware Expectations

### Network Bandwidth
- **NVLink** (GPU-GPU): 300-600 GB/s
- **PCIe 4.0 x16**: 30-50 GB/s
- **10G Ethernet**: 1.2 GB/s
- **InfiniBand HDR**: 200+ GB/s

### Communication Overhead
- **Fast network + large model**: <20% overhead
- **Slow network + small model**: >60% overhead

## Extending the Code

All code is designed to be hackable. Examples:

### Add Gradient Compression
```python
# In train_ddp(), before all-reduce:
compressed = compress_top_k(param.grad, k=0.1)  # Keep top 10%
dist.all_reduce(compressed)
param.grad = decompress(compressed)
```

### Add Local SGD
```python
# Sync every K steps instead of every step
if step % local_steps == 0:
    dist.all_reduce(param.grad)
```

### Add Custom Model
```python
# Replace SimpleCNN with your own
class MyModel(nn.Module):
    def __init__(self, config):
        # Your architecture
        pass
```

## Debugging Tips

### Check Distributed Setup
```bash
# Test if NCCL works
torchrun --nproc_per_node=2 python -c "import torch.distributed as dist; dist.init_process_group('nccl'); print(f'Rank {dist.get_rank()}')"
```

### Profile Communication
Set environment variables:
```bash
export NCCL_DEBUG=INFO  # Verbose NCCL logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch distributed logs
```

### Measure Bandwidth
```bash
# Test your actual hardware
torchrun --nproc_per_node=4 -m synthetic --bench --sweep
```

## References

- **PyTorch DDP Tutorial**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **NCCL Documentation**: https://docs.nvidia.com/deeplearning/nccl/
- **Ring All-Reduce**: Baidu Research (2017)

## License

MIT - Use freely for education and research.
