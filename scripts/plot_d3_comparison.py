#!/usr/bin/env python3
"""Plot D3 seed-42 comparison across all grammar variants."""

import json
from pathlib import Path
import sys

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_doors_derivation_suite import (
    ALL_MODES,
    _COLORS,
    _DIR_LABELS,
    _LABELS,
    _aggregate_across_seeds,
    _load_iteration_logs,
)

SUITE_DIR = Path(__file__).resolve().parent.parent / (
    "experiments/doors_grammar_suite/20260304_154856_rooms3_5"
)
D = 3
SEEDS = [42]

METRICS = [
    ("best_solve_rate", "Best Solve Rate", (-0.05, 1.1)),
    ("best_avg_reward", "Best Avg Reward", None),
    ("unique_programs", "Unique Programs", None),
]


def _load_suite_hp(suite_dir):
    """Load hyperparameters from suite_config.json if available."""
    cfg_path = suite_dir / "suite_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = _load_iteration_logs(SUITE_DIR, ALL_MODES, [D], SEEDS)
    if not logs:
        print("No logs found.")
        return

    suite_hp = _load_suite_hp(SUITE_DIR)
    sims = suite_hp.get("sims_by_d", {}).get(str(D), "?")
    n_games = suite_hp.get("n_games", "?")
    n_iters = suite_hp.get("n_iters", "?")
    hp_str = f"sims={sims} games={n_games} iters={n_iters}"

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5))
    fig.suptitle(f"D{D} Grammar Comparison (seed {SEEDS[0]})\n{hp_str}",
                 fontsize=14, fontweight="bold")

    for ax, (metric, ylabel, ylim) in zip(axes, METRICS):
        agg = _aggregate_across_seeds(logs, ALL_MODES, D, SEEDS, metric)
        for mode, (iters, means, _stds) in agg.items():
            ax.plot(iters, means, color=_COLORS.get(mode, "#333"),
                    linewidth=2, label=_LABELS.get(mode, mode))
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = SUITE_DIR / "d3_seed42_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
