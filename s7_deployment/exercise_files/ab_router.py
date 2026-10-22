"""A/B test router: sticky user bucketing in front of two model backends.

It also exposes admin endpoints (weights + metrics) so a canary controller can shift traffic gradually.
State is in memory, so run a SINGLE replica. In production you would use Cloud Monitoring / Prometheus.

Run with: uvicorn ab_router:app --port 8000
"""

import hashlib
import json
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

BACKENDS = {"a": os.getenv("BACKEND_A", "http://localhost:8001"), "b": os.getenv("BACKEND_B", "http://localhost:8002")}
EXPERIMENT = os.getenv("EXPERIMENT", "mnist-ab")
weights = {"a": 50.0, "b": 50.0}  # percentages


def assign_variant(user_id: str, experiment: str, weights: dict[str, float]) -> str:
    """Deterministically map a user to a variant (sticky assignment).

    We hash "experiment:user_id" with SHA-256 and turn it into a bucket in [0, 10000). Variants own consecutive
    slices of the buckets, proportional to their weights. Do NOT use Python's builtin `hash()` here: for strings it is
    salted per process (PYTHONHASHSEED), so every restart or replica would reassign users to other variants.
    Including the experiment name makes assignments independent between experiments.
    """
    digest = hashlib.sha256(f"{experiment}:{user_id}".encode()).hexdigest()
    bucket = int(digest, 16) % 10000
    total = sum(weights.values())
    cumulative = 0.0
    for variant, weight in weights.items():
        cumulative += weight / total * 10000
        if bucket < cumulative:
            return variant
    return list(weights)[-1]  # guard against floating point round-off


# Per-variant statistics since the last reset (bounded memory for the latencies)
stats = defaultdict(lambda: {"requests": 0, "errors": 0, "latencies": deque(maxlen=1000)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create one shared HTTP client for all requests."""
    app.state.client = httpx.AsyncClient(timeout=10)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: dict, x_user_id: str | None = Header(default=None)):
    """Route the request to a backend based on the user id."""
    # Without a user id we fall back to a random one, so assignment is NOT sticky for such requests.
    user_id = x_user_id or str(uuid.uuid4())
    variant = assign_variant(user_id, EXPERIMENT, weights)
    start = time.perf_counter()
    try:
        response = await app.state.client.post(f"{BACKENDS[variant]}/predict", json=request)
        status = response.status_code
    except httpx.HTTPError:
        status = 502
    latency_ms = (time.perf_counter() - start) * 1000

    s = stats[variant]
    s["requests"] += 1
    s["errors"] += status >= 500
    s["latencies"].append(latency_ms)
    # On Cloud Run, a JSON line on stdout becomes a structured log entry (`jsonPayload`) that can be queried.
    log = {"severity": "INFO", "message": "ab_assignment", "experiment": EXPERIMENT, "user_id": user_id}
    print(json.dumps({**log, "variant": variant, "status": status, "latency_ms": round(latency_ms, 1)}), flush=True)

    if status >= 300:
        raise HTTPException(status_code=status, detail=f"Backend {variant} returned {status}")
    return {**response.json(), "variant": variant}


class Weights(BaseModel):
    """Traffic split in percent."""

    a: float
    b: float


@app.get("/admin/weights")
def get_weights():
    """Current traffic split."""
    return weights


@app.put("/admin/weights")
def set_weights(new: Weights):
    """Change the traffic split (must sum to 100)."""
    if abs(new.a + new.b - 100) > 1e-6 or min(new.a, new.b) < 0:
        raise HTTPException(status_code=422, detail="Weights must be non-negative and sum to 100")
    weights.update(a=new.a, b=new.b)
    return weights


@app.get("/admin/metrics")
def get_metrics():
    """Per-variant request count, error rate and latency percentiles since the last reset."""
    out = {}
    for variant in BACKENDS:
        s = stats[variant]
        n = s["requests"]
        p50, p95 = np.percentile(s["latencies"], [50, 95]) if n else (0.0, 0.0)
        out[variant] = {
            "requests": n,
            "errors": s["errors"],
            "error_rate": s["errors"] / n if n else 0.0,
            "p50_latency_ms": float(p50),
            "p95_latency_ms": float(p95),
        }
    return out


@app.post("/admin/metrics/reset")
def reset_metrics():
    """Forget all collected statistics."""
    stats.clear()
    return {"status": "reset"}
