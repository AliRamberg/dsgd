# Agent Guidelines for DDP Repository

This repository contains code for a distributed deep learning seminar, showcasing various distributed SGD strategies (SSGD, ASGD, SSP, LocalSGD) using Ray on Kubernetes.

## 1. Environment & Build

### Prerequisites
- **Python**: 3.13+
- **Dependency Manager**: `uv`
- **Container Engine**: Docker with `buildx`
- **Cloud**: AWS (ECR)
- **Orchestrator**: Kubernetes (KubeRay)

### Environment Setup
- Always use the `self` AWS profile:
  ```bash
  export AWS_PROFILE=self
  ```
- Use `uv` for local dependency management:
  ```bash
  uv sync
  ```

### Build & Publish
To build and push the Docker image to ECR with caching enabled, use the following command. **Do not use standard `docker build`.**

```bash
docker buildx build --platform linux/amd64 \
  -t 496105080108.dkr.ecr.us-east-2.amazonaws.com/ddp:latest \
  --cache-from type=registry,ref=496105080108.dkr.ecr.us-east-2.amazonaws.com/ddp:cache \
  --cache-to type=registry,ref=496105080108.dkr.ecr.us-east-2.amazonaws.com/ddp:cache,mode=max \
  --push .
```

### Deployment
Deploy the training jobs to the Kubernetes cluster using `kubectl`. Ensure you are in the correct context (`seminar` or similar).

```bash
# Example: Deploying LocalSGD comparison
kubectl apply -f ray-job-compare-localsgd.yaml
```

To restart a job, delete it first:
```bash
kubectl delete rayjob compare-localsgd -n ray-jobs
kubectl apply -f ray-job-compare-localsgd.yaml
```

## 2. Development & Testing

### Running Locally
You can run the training scripts locally using `python` (via `uv run`):

```bash
# Run SSGD locally
uv run python main.py --mode ssgd --num-workers 2 --steps 10
```

### Testing
There is no formal test suite (pytest) currently configured in `pyproject.toml`.
- **Manual Verification**: Run a short training loop locally (as above) to verify syntax and basic logic before deploying.
- **Cluster Verification**: Check logs of the Ray driver pod.

### Linting
Follow standard Python PEP 8 guidelines.
- **Imports**: Group imports: Standard Library, Third Party (Ray, Torch, Boto3), Local.
- **Formatting**: Keep lines reasonable (< 100 chars if possible, though not strictly enforced).

## 3. Code Style & Conventions

### Python
- **Type Hints**: Use type hints for function arguments and return values.
  ```python
  def train_step(self, weights: list[np.ndarray]) -> dict[str, float]: ...
  ```
- **Docstrings**: Add docstrings to classes and complex methods explaining their role in the distributed system.
- **Data Classes**: Use `dataclasses` for configuration objects (e.g., `TrainConfig`).

### Ray & Distributed Systems
- **Actors**:
  - Worker classes should be decorated with `@ray.remote`.
  - Ensure GPU actors explicitly call `.to(device)` on tensors, as the driver might run on CPU.
- **Resources**:
  - Specify `num_gpus` in `@ray.remote` if the actor needs GPU access.
  - Use `ray.get()` sparingly to avoid blocking the driver unnecessarily (except for SSGD).
- **Heterogeneity**:
  - Use `time.sleep()` to simulate stragglers as defined in `sleep_heterogeneity()`.

### Kubernetes Manifests
- **RayJob**:
  - Ensure `entrypoint` commands match the CLI arguments of `main.py`.
  - Define correct resource requests (CPU/GPU) for worker groups.
  - Use the correct image tag (`:latest` for development).

## 4. Key Files Structure
- `main.py`: Entry point and driver logic. Orchestrates the training.
- `train_*.py`: Implementation of specific strategies (SSGD, ASGD, SSP, LocalSGD).
- `data.py`: Dataset generation and sharding (Synthetic XOR).
- `config.py`: Configuration dataclasses.
- `ray-job-*.yaml`: K8s deployment manifests for different experiments.

## 5. Troubleshooting
- **"Found no NVIDIA driver"**: This usually means code running on the CPU head node is trying to initialize CUDA.
  - **Fix**: Create `DatasetShard` on CPU (`driver_device="cpu"`). Move to GPU *inside* the worker actor.
- **Staleness/Lag**: Check `metrics.py` and Grafana dashboards.

---
*Generated for AI Agents working on the DDP repository.*
