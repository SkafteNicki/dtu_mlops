"""Simulate users sending requests and log the (delayed) ground truth feedback.

Every response is written as one JSON line to the output file, together with the true label of the image. This mimics
a production system where the label arrives later (e.g. a user corrects the prediction) and is joined to the request.

Example: python traffic_generator.py --url http://localhost:8000 --n-requests 2000
"""

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
import typer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

app = typer.Typer()


async def send_one(client, semaphore, url, user_id, pixels, label):
    """Send one request and log the outcome."""
    async with semaphore:  # at most `concurrency` requests in flight
        start = time.perf_counter()
        record = {"user_id": user_id, "variant": None, "model_version": None, "prediction": None, "label": label}
        try:
            response = await client.post(f"{url}/predict", json={"pixels": pixels}, headers={"X-User-ID": user_id})
            record["status_code"] = response.status_code
            if response.status_code == 200:
                body = response.json()
                record.update(
                    variant=body.get("variant"), model_version=body["model_version"], prediction=body["prediction"]
                )
        except httpx.HTTPError:
            record["status_code"] = 0  # connection error / timeout
        record["correct"] = record["prediction"] == label
        record["latency_ms"] = (time.perf_counter() - start) * 1000
        record["timestamp"] = time.time()
        return record


async def run(url, n_requests, n_users, concurrency, seed):
    """Run all requests concurrently (bounded by the semaphore)."""
    _, images, _, labels = train_test_split(*load_digits(return_X_y=True), test_size=0.5, random_state=42)
    rng = np.random.default_rng(seed)
    image_ids = rng.integers(0, len(images), n_requests)
    user_ids = rng.integers(0, n_users, n_requests)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=10) as client:
        tasks = [
            send_one(client, semaphore, url, f"user-{u:04d}", images[i].tolist(), int(labels[i]))
            for u, i in zip(user_ids, image_ids)
        ]
        return await asyncio.gather(*tasks)


@app.command()
def main(
    url: str = "http://localhost:8000",
    n_requests: int = 2000,
    n_users: int = 500,
    concurrency: int = 20,
    output: str = "traffic_log.jsonl",
    seed: int = 0,
):
    """Send traffic to URL and print a summary per model version."""
    records = asyncio.run(run(url, n_requests, n_users, concurrency, seed))
    Path(output).write_text("".join(json.dumps(r) + "\n" for r in records))
    groups = defaultdict(list)
    for r in records:
        groups[r["model_version"] or "failed"].append(r)
    print(f"{'model':<8}{'requests':>10}{'accuracy':>10}{'error rate':>12}{'p50 ms':>9}{'p95 ms':>9}")
    for name, rs in sorted(groups.items()):
        ok = [r for r in rs if r["status_code"] == 200]
        accuracy = np.mean([r["correct"] for r in ok]) if ok else float("nan")
        errors = 1 - len(ok) / len(rs)
        p50, p95 = np.percentile([r["latency_ms"] for r in rs], [50, 95])
        print(f"{name:<8}{len(rs):>10}{accuracy:>10.3f}{errors:>12.3f}{p50:>9.1f}{p95:>9.1f}")
    print(f"Log written to {output}")


if __name__ == "__main__":
    app()
