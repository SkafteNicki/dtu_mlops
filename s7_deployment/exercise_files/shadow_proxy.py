"""Shadow deployment proxy: users get the primary model's answer, the shadow model sees a copy of the traffic.

In `async` mode the shadow call happens in the background and cannot slow down or break the user's request.
In `sync` mode (the naive implementation) we wait for both models, so the shadow's latency is paid by the user.
Every shadow call produces one comparison log line (stdout, and a local JSONL file if SHADOW_LOG_FILE is set).

Run with: uvicorn shadow_proxy:app --port 8000
"""

import asyncio
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

PRIMARY_URL = os.getenv("PRIMARY_URL", "http://localhost:8001")
SHADOW_URL = os.getenv("SHADOW_URL", "http://localhost:8002")
SHADOW_TIMEOUT_S = float(os.getenv("SHADOW_TIMEOUT_S", "2.0"))
SHADOW_MODE = os.getenv("SHADOW_MODE", "async")  # "async" or "sync"
SHADOW_LOG_FILE = os.getenv("SHADOW_LOG_FILE", "shadow_log.jsonl")

# Keep strong references to background tasks: the event loop only keeps weak references, so a task nobody references
# may be garbage collected before it finishes (see the asyncio.create_task docs).
background_tasks = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create one shared HTTP client."""
    app.state.client = httpx.AsyncClient(timeout=10)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


async def call(url: str, payload: dict, timeout_s: float):
    """Call a model. Returns (response body or None, latency in ms, error string or None). Never raises."""
    start = time.perf_counter()
    try:
        response = await app.state.client.post(f"{url}/predict", json=payload, timeout=timeout_s)
        response.raise_for_status()
        body, error = response.json(), None
    except httpx.HTTPError as e:
        body, error = None, type(e).__name__
    return body, (time.perf_counter() - start) * 1000, error


def log_comparison(payload, primary, shadow):
    """Write one comparison line. We log a hash of the input, not the raw pixels (privacy + log volume)."""
    (p_body, p_ms, _), (s_body, s_ms, s_error) = primary, shadow
    p_pred = p_body["prediction"] if p_body else None
    s_pred = s_body["prediction"] if s_body else None
    line = {
        "severity": "INFO",
        "message": "shadow_comparison",
        "request_id": str(uuid.uuid4()),
        "input_hash": hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16],
        "primary_prediction": p_pred,
        "primary_latency_ms": round(p_ms, 1),
        "shadow_prediction": s_pred,
        "shadow_latency_ms": round(s_ms, 1),
        "shadow_error": s_error,
        "agree": s_error is None and p_pred == s_pred,
    }
    print(json.dumps(line), flush=True)  # on Cloud Run this becomes a jsonPayload in Cloud Logging
    if SHADOW_LOG_FILE:
        with open(SHADOW_LOG_FILE, "a") as f:
            f.write(json.dumps(line) + "\n")


async def shadow_and_log(payload, primary):
    """Background job: call the shadow model, then log the comparison."""
    log_comparison(payload, primary, await call(SHADOW_URL, payload, SHADOW_TIMEOUT_S))


@app.post("/predict")
async def predict(payload: dict):
    """Return the primary prediction; compare with the shadow prediction on the side."""
    if SHADOW_MODE == "sync":
        primary, shadow = await asyncio.gather(
            call(PRIMARY_URL, payload, 10), call(SHADOW_URL, payload, SHADOW_TIMEOUT_S)
        )
        log_comparison(payload, primary, shadow)
    else:
        primary = await call(PRIMARY_URL, payload, 10)
        if primary[0] is not None:
            task = asyncio.create_task(shadow_and_log(payload, primary))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
    if primary[0] is None:
        raise HTTPException(status_code=502, detail=f"Primary model failed: {primary[2]}")
    return primary[0]
