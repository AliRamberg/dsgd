FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

COPY pyproject.toml uv.lock .

RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim 

# For Kubernetes readiness/liveness probes (HTTP checks, etc.)
# Ray also requires 'procps' (ps) for metrics/monitoring.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    procps \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy uv + the already-synced environment.
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /opt/venv /opt/venv

COPY pyproject.toml uv.lock ./
COPY . .

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1

ENTRYPOINT ["python"]