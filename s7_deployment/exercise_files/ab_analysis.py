"""Statistical analysis of an A/B test: sample ratio check, two-proportion z-test and sample size planning.

Examples:
    python ab_analysis.py traffic_log.jsonl      # analyse a log from traffic_generator.py
    python ab_analysis.py --example              # textbook example: 92.0% vs 94.0% accuracy
    python ab_analysis.py --plot                 # power curve figure
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import typer
from scipy.stats import chisquare, norm

app = typer.Typer()


def two_proportion_ztest(successes_a: int, n_a: int, successes_b: int, n_b: int) -> tuple[float, float, tuple]:
    """Two-sided z-test for p_b != p_a. Returns z, p-value and the 95% confidence interval of p_b - p_a."""
    p_a, p_b = successes_a / n_a, successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)  # under H0 both groups share one rate
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool
    p_value = 2 * norm.sf(abs(z))
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)  # unpooled SE for the interval
    margin = norm.ppf(0.975) * se_diff
    return float(z), float(p_value), (float(p_b - p_a - margin), float(p_b - p_a + margin))


def required_sample_size(p_baseline: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Samples needed PER GROUP to detect an absolute improvement `mde` over `p_baseline`."""
    p1, p2 = p_baseline, p_baseline + mde
    z_alpha, z_beta = norm.ppf(1 - alpha / 2), norm.ppf(power)
    variance = p1 * (1 - p1) + p2 * (1 - p2)
    return int(np.ceil((z_alpha + z_beta) ** 2 * variance / mde**2))


def sample_ratio_mismatch(n_a: int, n_b: int, expected: tuple[float, float] = (0.5, 0.5)) -> float:
    """Chi-square test that the observed split matches the intended split. A tiny p-value means a bug."""
    total = n_a + n_b
    return float(chisquare([n_a, n_b], [total * expected[0], total * expected[1]]).pvalue)


def report(successes_a: int, n_a: int, successes_b: int, n_b: int, names=("A", "B"), units=None):
    """Print SRM check followed by the z-test. `units` = (users in A, users in B), the randomisation unit."""
    units = units or (n_a, n_b)
    p_srm = sample_ratio_mismatch(*units)
    print(f"Assigned units: {names[0]}={units[0]}, {names[1]}={units[1]}  (SRM p-value {p_srm:.4f})")
    print(f"Samples: {names[0]}={n_a}, {names[1]}={n_b}")
    if p_srm < 0.001:  # common convention (e.g. Microsoft ExP, Kohavi et al.): SRM alarms at p < 0.001
        print("WARNING: sample ratio mismatch! The assignment is broken, do not trust the results below.")
    z, p, (lo, hi) = two_proportion_ztest(successes_a, n_a, successes_b, n_b)
    print(f"Accuracy {names[0]}: {successes_a / n_a:.4f}   {names[1]}: {successes_b / n_b:.4f}")
    print(f"Difference: {successes_b / n_b - successes_a / n_a:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"z = {z:.3f}, p-value = {p:.5f} ->", "significant at 5%" if p < 0.05 else "NOT significant at 5%")


def plot_power_curve(path: str = "figures/ab_power_curve.png"):
    """Plot required sample size against the minimum detectable effect."""
    import matplotlib.pyplot as plt

    mdes = np.linspace(0.005, 0.05, 50)
    fig, ax = plt.subplots(figsize=(6, 4))
    for power, color in [(0.8, "#0072B2"), (0.9, "#D55E00")]:
        ax.plot(mdes * 100, [required_sample_size(0.92, m, power=power) for m in mdes], color=color, lw=2)
        ax.lines[-1].set_label(f"power = {power:.0%}")
    ax.set_yscale("log")
    ax.set_xlabel("Minimum detectable effect (percentage points)")
    ax.set_ylabel("Required samples per group")
    ax.set_title("Sample size for baseline accuracy 92% (alpha = 0.05)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)


@app.command()
def main(log: str = typer.Argument("traffic_log.jsonl"), example: bool = False, plot: bool = False):
    """Analyse a traffic log (or the textbook example)."""
    if plot:
        plot_power_curve()
        print("Saved figures/ab_power_curve.png")
    if example:
        report(9200, 10000, 9400, 10000)
        print(f"Needed per group for a 1pp effect at 92%: {required_sample_size(0.92, 0.01)}")
    if example or plot:
        return
    n, correct, users = Counter(), Counter(), {}
    for line in Path(log).read_text().splitlines():
        r = json.loads(line)
        if r["status_code"] == 200:  # only successful requests have a prediction to judge
            key = r["variant"] or r["model_version"]
            n[key] += 1
            correct[key] += r["correct"]
            users[r["user_id"]] = key  # SRM is checked on users: they, not requests, are randomised
    if len(n) != 2:
        msg = f"Expected exactly 2 variants in the log, found {sorted(n)}"
        raise typer.BadParameter(msg)
    a, b = sorted(n)
    report(correct[a], n[a], correct[b], n[b], names=(a, b), units=tuple(Counter(users.values())[v] for v in (a, b)))


if __name__ == "__main__":
    app()
