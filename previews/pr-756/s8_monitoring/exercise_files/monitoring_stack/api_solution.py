"""The classifier as an HTTP API, instrumented with Prometheus metrics.

    uv run uvicorn api_solution:app --port 8000      # directly, without Docker
    docker compose up -d --build                     # together with Prometheus and Grafana (see compose.yaml)

POST an image to /predict and get the class, the confidence and the model version back. Prometheus scrapes /metrics.

Environment: MODEL_CHECKPOINT, the checkpoint to serve (default: models/model.pt).
"""

import os
import time
from contextlib import asynccontextmanager

import torch
from dataset import CLASSES, BadImageError, decode, preprocess
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from model import load_checkpoint
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Histogram, generate_latest

# --- service layer metrics ---------------------------------------------------------------------------------------
REQUESTS_TOTAL = Counter(
    "predict_requests_total",
    "Answered /predict requests, by predicted class",
    ["predicted_class"],
)
LATENCY = Histogram(
    "predict_latency_seconds",
    "Time to answer a /predict request",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
ERRORS = Counter("predict_errors_total", "Rejected /predict requests, by reason", ["reason"])

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, when the service starts."""
    torch.set_num_threads(1)
    state["model"], state["version"] = load_checkpoint(os.environ.get("MODEL_CHECKPOINT", "models/model.pt"))
    print(f"serving model version {state['version']}", flush=True)
    yield
    state.clear()


app = FastAPI(title="Image classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness check that also says which model version is served."""
    return {"status": "ok", "model_version": state["version"]}


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict:
    """Classify one image."""
    start = time.perf_counter()
    try:
        img = decode(file.file.read())
    except BadImageError as err:
        ERRORS.labels(reason=err.reason).inc()
        raise HTTPException(status_code=400, detail=str(err)) from None

    with torch.no_grad():
        probs = state["model"](preprocess(img)[None]).softmax(1)[0]
    k = int(probs.argmax())

    REQUESTS_TOTAL.labels(predicted_class=CLASSES[k]).inc()
    LATENCY.observe(time.perf_counter() - start)
    return {"class": CLASSES[k], "confidence": round(float(probs[k]), 4), "model_version": state["version"]}


@app.get("/metrics")
def metrics() -> Response:
    """Every metric in the default registry, in Prometheus' text format."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
