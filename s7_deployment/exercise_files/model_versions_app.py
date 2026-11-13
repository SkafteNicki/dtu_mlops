"""Digit classifier service with two "model versions", used for the deployment testing exercises.

Choose the version with the env var MODEL_VERSION (v1 = weak model, v2 = better model). Faults can be injected with
FAILURE_RATE (probability of an HTTP 500) and EXTRA_LATENCY_MS (sleep added to every request) so that we can
practice canary rollbacks and shadow deployments.

Run with: MODEL_VERSION=v2 uvicorn model_versions_app:app --port 8002
"""

import asyncio
import os
import random

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
FAILURE_RATE = float(os.getenv("FAILURE_RATE", "0"))
EXTRA_LATENCY_MS = float(os.getenv("EXTRA_LATENCY_MS", "0"))
REVISION = os.getenv("K_REVISION", "local")  # set automatically by Cloud Run

# Train at startup (takes well below 2 seconds). Same split everywhere so all services agree on the test set.
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
if MODEL_VERSION == "v1":
    # weak model: few training samples + strong regularisation (test accuracy ~0.91)
    model = LogisticRegression(C=0.01, max_iter=200).fit(X_train[:150], y_train[:150])
elif MODEL_VERSION == "v2":
    # better model: SVM trained on all training data (test accuracy ~0.99)
    model = SVC(gamma=0.001).fit(X_train, y_train)
else:
    msg = f"Unknown MODEL_VERSION {MODEL_VERSION!r}, use 'v1' or 'v2'"
    raise ValueError(msg)

app = FastAPI()


class PredictRequest(BaseModel):
    """A flattened 8x8 digit image."""

    pixels: list[float] = Field(min_length=64, max_length=64)


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(request: PredictRequest):
    """Predict the digit in the image."""
    if EXTRA_LATENCY_MS > 0:
        await asyncio.sleep(EXTRA_LATENCY_MS / 1000)  # does not block other requests
    if random.random() < FAILURE_RATE:
        raise HTTPException(status_code=500, detail="Injected failure")
    prediction = int(model.predict([request.pixels])[0])
    return {"prediction": prediction, "model_version": MODEL_VERSION, "revision": REVISION}
