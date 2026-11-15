"""Analyse shadow deployment logs: agreement, shadow errors, latency and a primary-vs-shadow disagreement matrix.

Input is either the local shadow_log.jsonl written by shadow_proxy.py, or a Cloud Logging export:
    gcloud logging read 'jsonPayload.message="shadow_comparison"' --format json > export.json
The format is detected automatically. No labels are used: in a real shadow test you usually don't have the ground
truth at request time, so we can only compare the two models with each other.

Run with: python shadow_analysis.py shadow_log.jsonl --plot figures/shadow_analysis.png
"""

import json
from pathlib import Path

import numpy as np
import typer

app = typer.Typer()


def load_entries(path: str) -> list[dict]:
    """Load comparison records from a JSONL file or a Cloud Logging JSON export."""
    text = Path(path).read_text().strip()
    if text.startswith("["):  # gcloud export: a JSON list of log entries with a `jsonPayload`
        return [
            e["jsonPayload"] for e in json.loads(text) if e.get("jsonPayload", {}).get("message") == "shadow_comparison"
        ]
    return [r for r in map(json.loads, text.splitlines()) if r.get("message") == "shadow_comparison"]


def disagreement_matrix(entries: list[dict]) -> np.ndarray:
    """10x10 counts, rows = primary prediction, columns = shadow prediction (failed shadow calls are skipped)."""
    matrix = np.zeros((10, 10), dtype=int)
    for e in entries:
        if e["shadow_prediction"] is not None and e["primary_prediction"] is not None:
            matrix[e["primary_prediction"], e["shadow_prediction"]] += 1
    return matrix


def plot(entries, matrix, path):
    """Heatmap of the matrix and latency ECDFs."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    image = ax1.imshow(matrix + 0.5, norm=LogNorm(), cmap="viridis")  # +0.5 so zeros are drawn on a log scale
    for i, j in np.ndindex(10, 10):
        if matrix[i, j]:
            ax1.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7, color="white")
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xlabel("Shadow prediction")
    ax1.set_ylabel("Primary prediction")
    ax1.set_title("Prediction counts (log colour scale)")
    fig.colorbar(image, ax=ax1, label="count")
    for key, label, color in [("primary", "primary", "#0072B2"), ("shadow", "shadow", "#D55E00")]:
        latencies = np.sort([e[f"{key}_latency_ms"] for e in entries])
        ax2.plot(latencies, np.arange(1, len(latencies) + 1) / len(latencies), label=label, color=color, lw=2)
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Fraction of requests (ECDF)")
    ax2.set_title("Latency of primary vs shadow")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


@app.command()
def main(log: str = typer.Argument("shadow_log.jsonl"), plot_path: str = typer.Option(None, "--plot")):
    """Print the shadow report."""
    entries = load_entries(log)
    n = len(entries)
    errors = sum(e["shadow_error"] is not None for e in entries)
    print(f"Requests compared:   {n}")
    print(f"Shadow error rate:   {errors / n:.3f}")
    print(f"Agreement rate:      {np.mean([e['agree'] for e in entries]):.3f}")
    for key in ["primary", "shadow"]:
        p50, p95 = np.percentile([e[f"{key}_latency_ms"] for e in entries], [50, 95])
        print(f"{key:<8} latency:    p50 {p50:.1f} ms, p95 {p95:.1f} ms")
    matrix = disagreement_matrix(entries)
    print("\nPrimary (rows) vs shadow (columns) predictions:")
    print(matrix)
    if plot_path:
        plot(entries, matrix, plot_path)
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    app()
