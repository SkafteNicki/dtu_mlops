"""Compare fixed A/B/n split, epsilon-greedy and Thompson sampling on three model variants.

The bandits shift traffic to the better variant while the experiment is running, which lowers the "regret" (expected
successes lost compared to always using the best variant).

Run with: python bandit_simulation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
TRUE_RATES = np.array([0.90, 0.92, 0.95])
N_REQUESTS, N_RUNS, EPSILON = 10000, 200, 0.1
K = len(TRUE_RATES)


def simulate(policy: str):
    """Run N_RUNS independent experiments in parallel. Returns cumulative regret and best-arm share per request."""
    wins = np.zeros((N_RUNS, K))
    pulls = np.zeros((N_RUNS, K))
    regret = np.zeros((N_RUNS, N_REQUESTS))
    best_share = np.zeros((N_RUNS, N_REQUESTS))
    runs = np.arange(N_RUNS)
    for t in range(N_REQUESTS):
        if policy == "fixed":
            arm = rng.integers(0, K, N_RUNS)  # equal split
        elif policy == "epsilon-greedy":
            means = wins / np.maximum(pulls, 1)
            explore = rng.random(N_RUNS) < EPSILON
            arm = np.where(explore, rng.integers(0, K, N_RUNS), means.argmax(axis=1))
        else:  # Thompson sampling: sample a plausible rate per arm from its Beta posterior, play the highest
            arm = rng.beta(wins + 1, pulls - wins + 1).argmax(axis=1)
        reward = rng.random(N_RUNS) < TRUE_RATES[arm]
        wins[runs, arm] += reward
        pulls[runs, arm] += 1
        regret[:, t] = TRUE_RATES.max() - TRUE_RATES[arm]  # expected loss of this choice
        best_share[:, t] = arm == TRUE_RATES.argmax()
    return regret.cumsum(axis=1).mean(axis=0), best_share.mean(axis=0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
colors = {"fixed": "#999999", "epsilon-greedy": "#0072B2", "thompson": "#D55E00"}
for policy, color in colors.items():
    cum_regret, share = simulate(policy)
    print(f"{policy:<15} final cumulative regret: {cum_regret[-1]:7.1f}")
    ax1.plot(cum_regret, color=color, lw=2, label=policy)
    smooth = np.convolve(share, np.ones(100) / 100, mode="valid")  # moving average for readability
    ax2.plot(smooth, color=color, lw=2, label=policy)
ax1.set_xlabel("Requests")
ax1.set_ylabel("Cumulative regret (lost successes)")
ax1.set_title("Regret (lower is better)")
ax2.set_xlabel("Requests")
ax2.set_ylabel("Share of traffic to best variant")
ax2.set_title("Traffic to the best variant (100-request average)")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/bandit_regret.png", dpi=150)
