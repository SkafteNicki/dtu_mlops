"""Automated canary rollout: shift traffic to the canary step by step and roll back if guardrails are violated.

Two backends are available:
  local     the traffic split is controlled through the admin API of ab_router.py (variant "b" = canary)
  cloudrun  the traffic split is controlled with `gcloud run services update-traffic` (canary revision tagged "canary")

Examples:
    python canary_rollout.py --plot figures/canary_rollout_success.png
    python canary_rollout.py --backend cloudrun --service my-service --region europe-west1
"""

import asyncio
import random
import subprocess
import time
from typing import Annotated, Protocol

import httpx
import matplotlib as mpl
import numpy as np
import typer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

app = typer.Typer()
_, X_TEST, _, _ = train_test_split(*load_digits(return_X_y=True), test_size=0.5, random_state=42)


async def send_requests(url: str, n: int, headers: dict | None = None) -> list[tuple[int, float]]:
    """Send n random digit images to url/predict. Returns (status code, latency in ms) per request."""
    semaphore = asyncio.Semaphore(20)

    async def one(client, i):
        async with semaphore:
            payload = {"pixels": random.choice(X_TEST).tolist()}
            start = time.perf_counter()
            try:
                status = (
                    await client.post(
                        f"{url}/predict", json=payload, headers={"X-User-ID": f"probe-{i}-{random.random()}"}
                    )
                ).status_code
            except httpx.HTTPError:
                status = 599
            return status, (time.perf_counter() - start) * 1000

    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        return await asyncio.gather(*[one(client, i) for i in range(n)])


def summarize(results: list[tuple[int, float]]) -> dict:
    """Client-side request count, error rate and p95 latency."""
    errors = sum(status >= 500 for status, _ in results)
    p95 = float(np.percentile([ms for _, ms in results], 95)) if results else 0.0
    return {"requests": len(results), "error_rate": errors / max(len(results), 1), "p95_latency_ms": p95}


class TrafficBackend(Protocol):
    """What the controller needs from the platform."""

    def set_canary_percent(self, percent: int) -> None:
        """Send `percent` of the traffic to the canary."""

    def measure(self, n_requests: int, percent: int) -> dict:
        """Generate probe traffic and return {"stable": {...}, "canary": {...}} metrics."""


class LocalRouter:
    """Use ab_router.py: variant a is stable, variant b is the canary."""

    def __init__(self, router_url: str):
        self.url = router_url

    def set_canary_percent(self, percent: int) -> None:
        """Update the router weights."""
        httpx.put(f"{self.url}/admin/weights", json={"a": 100 - percent, "b": percent}).raise_for_status()

    def measure(self, n_requests: int, percent: int) -> dict:
        """Reset router metrics, send traffic through the router and read the router's metrics."""
        httpx.post(f"{self.url}/admin/metrics/reset")
        asyncio.run(send_requests(self.url, n_requests))
        metrics = httpx.get(f"{self.url}/admin/metrics").json()
        return {"stable": metrics["a"], "canary": metrics["b"]}


class CloudRun:
    """Use Cloud Run revision tags. Metrics are measured by this script by calling the two URLs directly."""

    def __init__(self, service: str, region: str):
        self.service, self.region = service, region
        url = self.gcloud("describe", self.service, "--format=value(status.url)")
        self.stable_url = url
        self.canary_url = url.replace("https://", "https://canary---")  # URL of the revision tagged "canary"

    def gcloud(self, *args: str) -> str:
        """Run `gcloud run services ...` and return stdout."""
        cmd = ["gcloud", "run", "services", args[0], *args[1:], "--region", self.region]
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()

    def set_canary_percent(self, percent: int) -> None:
        """Assign `percent` of the traffic to the revision tagged "canary" (the rest stays on the other revision)."""
        self.gcloud("update-traffic", self.service, f"--to-tags=canary={percent}")

    def measure(self, n_requests: int, percent: int) -> dict:
        """Send the same number of probe requests to both URLs, so both have enough data for the comparison."""
        # Note: assumes the service allows unauthenticated access; otherwise add an identity token header.
        stable = asyncio.run(send_requests(self.stable_url, n_requests))
        canary = asyncio.run(send_requests(self.canary_url, n_requests))
        return {"stable": summarize(stable), "canary": summarize(canary)}


def check_guardrails(
    canary: dict, baseline: dict, max_error_increase: float, max_latency_ratio: float, latency_slack_ms: float
) -> str | None:
    """Return a reason string if a guardrail is violated, otherwise None."""
    if canary["error_rate"] > baseline["error_rate"] + max_error_increase:
        return f"error rate {canary['error_rate']:.3f} > stable {baseline['error_rate']:.3f} + {max_error_increase}"
    # The slack avoids rollbacks caused by measurement noise when latencies are only a few milliseconds.
    if canary["p95_latency_ms"] > baseline["p95_latency_ms"] * max_latency_ratio + latency_slack_ms:
        return (
            f"p95 {canary['p95_latency_ms']:.0f} ms > {max_latency_ratio} x stable {baseline['p95_latency_ms']:.0f} ms"
        )
    return None


def plot_history(history: list[dict], rolled_back: bool, path: str):
    """Canary traffic share, error rate and p95 latency per step. The rollback is marked in red."""
    steps = list(range(1, len(history) + 1))
    percents = [h["percent"] for h in history] + ([0] if rolled_back else [])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].step(range(1, len(percents) + 1), percents, where="post", color="#0072B2", lw=2)
    axes[0].set_ylabel("Traffic to canary (%)")
    axes[0].set_title("Canary traffic share")
    for ax, key, title, label in [
        (axes[1], "error_rate", "Error rate", "Error rate"),
        (axes[2], "p95_latency_ms", "p95 latency", "p95 latency (ms)"),
    ]:
        ax.plot(
            steps,
            [h["stable"][key] if h["stable"]["requests"] else np.nan for h in history],
            "o-",
            color="#009E73",
            label="stable",
        )
        ax.plot(
            steps,
            [h["canary"][key] if h["canary"]["requests"] else np.nan for h in history],
            "o-",
            color="#D55E00",
            label="canary",
        )
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.legend()
    labels = [f"{p}%" for p in percents]
    if rolled_back:
        labels[-1] = "rolled\nback"
    for ax in axes:
        ax.set_xlabel("Rollout step (canary traffic share)")
        ax.set_xticks(range(1, len(percents) + 1), labels)
        ax.set_xlim(0.7, len(percents) + 0.3)
        ax.grid(alpha=0.3)
        if rolled_back:
            ax.axvline(len(history), color="red", ls="--")
            ax.text(
                len(history) + 0.05,
                0.5,
                "guardrail violated",
                color="red",
                rotation=90,
                va="center",
                transform=ax.get_xaxis_transform(),
            )
    fig.suptitle("Canary rollout " + ("ROLLED BACK" if rolled_back else "PROMOTED"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)


@app.command()
def main(
    backend: str = "local",
    router_url: str = "http://localhost:8000",
    service: str = "model-service",
    region: str = "europe-west1",
    steps: Annotated[list[int], typer.Option()] = (5, 25, 50, 100),
    requests_per_step: int = 400,
    max_error_rate_increase: float = 0.01,
    max_latency_ratio: float = 1.5,
    latency_slack_ms: float = 50,
    plot: str = typer.Option(None, help="Save a figure of the rollout to this path"),
):
    """Roll out the canary step by step. Exit code 1 means it was rolled back."""
    platform = LocalRouter(router_url) if backend == "local" else CloudRun(service, region)
    history, baseline, failure = [], None, None
    for percent in steps:
        platform.set_canary_percent(percent)
        m = platform.measure(requests_per_step, percent)
        history.append({"percent": percent, **m})
        # At 100% the stable version gets (almost) no traffic, so keep comparing to the last good baseline.
        if m["stable"]["requests"] >= 30:
            baseline = m["stable"]
        for name in ("stable", "canary"):
            r = m[name]
            line = f"{percent:>3}% | {name:<6} n={r['requests']:<4} errors={r['error_rate']:.3f}"
            print(f"{line} p95={r['p95_latency_ms']:.0f} ms")
        failure = check_guardrails(
            m["canary"], baseline or m["stable"], max_error_rate_increase, max_latency_ratio, latency_slack_ms
        )
        if failure:
            break
    if failure:
        platform.set_canary_percent(0)
        print(f"ROLLBACK at {percent}%: {failure}")
    else:
        print("PROMOTED: canary passed all guardrails")
    if plot:
        plot_history(history, failure is not None, plot)
    raise typer.Exit(code=1 if failure else 0)


if __name__ == "__main__":
    app()
