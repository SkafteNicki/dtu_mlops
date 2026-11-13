"""Why you should not stop an A/B test as soon as p < 0.05 ("peeking").

We run many A/A tests (both variants have the SAME true accuracy), so every "significant" result is a false positive.
A single test at the end gives ~5% false positives, stopping at the first significant look gives far more.

Run with: python ab_peeking_simulation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
N_EXPERIMENTS, MAX_N, STEP, P_TRUE = 2000, 10000, 100, 0.92

# Cumulative successes of both arms at every look (100, 200, ..., 10000 samples per arm)
looks = np.arange(STEP, MAX_N + 1, STEP)
succ_a = rng.binomial(STEP, P_TRUE, (N_EXPERIMENTS, len(looks))).cumsum(axis=1)
succ_b = rng.binomial(STEP, P_TRUE, (N_EXPERIMENTS, len(looks))).cumsum(axis=1)

# Vectorised two-proportion z-test at every look of every experiment
p_a, p_b = succ_a / looks, succ_b / looks
p_pool = (succ_a + succ_b) / (2 * looks)
z = (p_b - p_a) / np.sqrt(2 * p_pool * (1 - p_pool) / looks + 1e-12)
p_values = 2 * norm.sf(np.abs(z))
significant = p_values < 0.05

fixed = significant[:, -1].mean()  # (a) only look once, at the end
peeking_curve = np.maximum.accumulate(significant, axis=1).mean(axis=0)  # (b) stop at first significant look
fixed_curve = significant.mean(axis=0)  # a single test with that sample size
print(f"False positive rate, single test at the end: {fixed:.3f}")
print(f"False positive rate, peeking every {STEP} samples: {peeking_curve[-1]:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for i in range(5):
    ax1.plot(looks, p_values[i], lw=1, alpha=0.8)
ax1.axhline(0.05, color="black", ls="--", label="p = 0.05")
ax1.set_yscale("log")
ax1.set_xlabel("Samples per variant")
ax1.set_ylabel("p-value")
ax1.set_title("p-values of 5 A/A tests (no real difference)")
ax1.legend()
ax2.plot(looks, peeking_curve, color="#D55E00", lw=2, label="peeking (stop at first p < 0.05)")
ax2.plot(looks, fixed_curve, color="#0072B2", lw=2, label="fixed horizon (single test)")
ax2.axhline(0.05, color="black", ls="--", lw=1)
ax2.set_xlabel("Samples per variant")
ax2.set_ylabel("False positive rate")
ax2.set_title(f"False positives in {N_EXPERIMENTS} A/A tests")
ax2.legend()
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/ab_peeking.png", dpi=120)
