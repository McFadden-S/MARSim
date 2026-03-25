"""
Generate performance plots from evaluation metrics JSON.

Usage::

    python scripts/generate_plots.py results/eval_metrics.json --output docs/performance.png
"""

import argparse
import json
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)

import numpy as np


def plot_metrics(metrics_path: str, output_path: str):
    with open(metrics_path) as f:
        data = json.load(f)

    episodes = data["episodes"]
    n = len(episodes["steps"])
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MARSim Evaluation Results", fontsize=16, fontweight="bold")

    # 1) Team rewards
    ax = axes[0, 0]
    ax.plot(x, episodes["friendly_reward"], label="Friendly", color="dodgerblue", alpha=0.7)
    ax.plot(x, episodes["enemy_reward"], label="Enemy", color="firebrick", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Team Reward")
    ax.set_title("Team Rewards per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Survival rates
    ax = axes[0, 1]
    ax.plot(x, episodes["friendly_alive"], label="Friendly Alive", color="dodgerblue", alpha=0.7)
    ax.plot(x, episodes["enemy_alive"], label="Enemy Alive", color="firebrick", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Surviving Agents")
    ax.set_title("Agent Survival at Episode End")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Episode length
    ax = axes[1, 0]
    ax.bar(x, episodes["steps"], color="mediumseagreen", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.grid(True, alpha=0.3)

    # 4) UGV outcomes
    ax = axes[1, 1]
    success = [int(v) for v in episodes["ugv_reached_goal"]]
    destroyed = [int(v) for v in episodes["ugv_destroyed"]]
    timeout = [1 - s - d for s, d in zip(success, destroyed)]

    # Cumulative rates
    cum_success = np.cumsum(success) / x
    cum_destroyed = np.cumsum(destroyed) / x

    ax.plot(x, cum_success, label="Success Rate", color="dodgerblue", linewidth=2)
    ax.plot(x, cum_destroyed, label="Destruction Rate", color="firebrick", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Rate")
    ax.set_title("UGV Outcome Rates")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance plots from eval metrics")
    parser.add_argument("metrics", help="Path to evaluation metrics JSON file")
    parser.add_argument("--output", default="docs/performance.png", help="Output image path")
    args = parser.parse_args()
    plot_metrics(args.metrics, args.output)
