#!/usr/bin/env python3
"""Plot ablation sweep results.

Reads eval_summary.csv files from experiment directories and produces:
  1. Learning curves: solve rate vs iteration, one line per mode, subplots per D
  2. Final solve rate bar chart: modes grouped by D
  3. Ablation delta heatmap: solve_rate(mode) - solve_rate(full)

Usage:
    python scripts/plot_ablation_sweep.py
    python scripts/plot_ablation_sweep.py --exp-dir experiments/doors_direct
    python scripts/plot_ablation_sweep.py --exp-dir experiments/doors_direct --D 15 20
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODE_ORDER = [
    "full", "policy-only", "value-only-1step",
    "uniform-prior", "zero-value", "unguided-search", "random-net",
]

MODE_COLORS = {
    "full": "#2196F3",
    "policy-only": "#FF9800",
    "value-only-1step": "#4CAF50",
    "uniform-prior": "#9C27B0",
    "zero-value": "#F44336",
    "unguided-search": "#795548",
    "random-net": "#607D8B",
}

MODE_SHORT = {
    "full": "full",
    "policy-only": "policy",
    "value-only-1step": "val-1step",
    "uniform-prior": "uni-prior",
    "zero-value": "zero-val",
    "unguided-search": "unguided",
    "random-net": "random",
}


def _discover_train_stats(exp_dir: Path):
    """Find train_stats.jsonl files and parse D/mode from directory names.

    Returns list of dicts with keys: D, mode, jsonl_path, records.
    Each record has: train_loss_policy, train_loss_value, avg_reward, num_examples.
    """
    runs = []
    for jsonl_path in sorted(exp_dir.rglob("train_stats.jsonl")):
        dirname = jsonl_path.parent.name
        m_d = re.search(r"_D(\d+)_", dirname)
        if not m_d:
            continue
        D = int(m_d.group(1))
        m_abl = re.search(r"_abl-([a-z0-9-]+)", dirname)
        mode = m_abl.group(1) if m_abl else "full"

        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            continue
        runs.append({"D": D, "mode": mode, "jsonl_path": jsonl_path,
                     "records": records})
    return runs


def plot_loss_curves(train_runs, D_values, output_path):
    """2xN panel: policy loss (top) and value loss (bottom) vs iteration."""
    n_d = len(D_values)
    fig, axes = plt.subplots(2, n_d, figsize=(6 * n_d, 8), squeeze=False)

    for col, D in enumerate(D_values):
        d_runs = [r for r in train_runs if r["D"] == D]

        for mode in MODE_ORDER:
            mode_runs = [r for r in d_runs if r["mode"] == mode]
            if not mode_runs:
                continue

            # Aggregate per-iteration across seeds
            p_vals = defaultdict(list)
            v_vals = defaultdict(list)
            for run in mode_runs:
                for i, rec in enumerate(run["records"]):
                    it = i + 1
                    p_vals[it].append(rec["train_loss_policy"])
                    v_vals[it].append(rec["train_loss_value"])

            iters = sorted(p_vals.keys())
            color = MODE_COLORS.get(mode, "gray")
            label = MODE_SHORT[mode]

            for row, vals, ylabel in [
                (0, p_vals, "Policy Loss (CE)"),
                (1, v_vals, "Value Loss (MSE)"),
            ]:
                ax = axes[row][col]
                means = [np.mean(vals[i]) for i in iters]
                stds = [np.std(vals[i]) for i in iters]
                ax.plot(iters, means, label=label, color=color, linewidth=2)
                ax.fill_between(iters,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.15, color=color)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)

        for row in range(2):
            axes[row][col].set_xlabel("Iteration")
            axes[row][col].legend(fontsize=8, loc="best")
        axes[0][col].set_title(f"D = {D}", fontsize=14)

    fig.suptitle("Training Loss by Ablation Mode\n"
                 "(all modes train identically with full MCTS; "
                 "differences are seed variance)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Loss curves saved to {output_path}")


def _discover_runs(exp_dir: Path):
    """Find eval_summary.csv files and parse D/mode from directory names.

    Returns list of dicts with keys: D, mode, seed, csv_path, rows.
    """
    runs = []
    for csv_path in sorted(exp_dir.rglob("eval_summary.csv")):
        dirname = csv_path.parent.name
        # Parse D from dirname like ..._D15_...
        m_d = re.search(r"_D(\d+)_", dirname)
        if not m_d:
            continue
        D = int(m_d.group(1))

        # Parse mode from _abl-{mode} or from CSV content
        m_abl = re.search(r"_abl-([a-z0-9-]+)", dirname)

        # Read CSV rows
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        # Get mode from dirname or CSV
        if m_abl:
            mode = m_abl.group(1)
        else:
            mode = rows[0].get("eval_ablation", "full")

        # Get seed from CSV
        seed = int(float(rows[0].get("seed", 0)))

        # Convert numeric fields
        for row in rows:
            for key in ("iteration", "solve_rate", "avg_reward",
                         "abl_solve_rate", "abl_avg_reward", "wall_clock_s"):
                if key in row and row[key] != "":
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass

        runs.append({
            "D": D, "mode": mode, "seed": seed,
            "csv_path": csv_path, "rows": rows,
        })

    return runs


def _get_solve_rate_key(mode):
    """Which CSV column holds the solve rate for this mode."""
    if mode == "full":
        return "solve_rate"
    return "abl_solve_rate"


def plot_learning_curves(runs, D_values, output_path):
    """Panel 1: solve rate vs iteration, one subplot per D."""
    n_d = len(D_values)
    fig, axes = plt.subplots(1, n_d, figsize=(6 * n_d, 5), squeeze=False)
    axes = axes[0]

    for ax, D in zip(axes, D_values):
        d_runs = [r for r in runs if r["D"] == D]

        # Group by mode, then aggregate across seeds
        for mode in MODE_ORDER:
            mode_runs = [r for r in d_runs if r["mode"] == mode]
            if not mode_runs:
                continue

            # For "full" mode, use solve_rate column
            # For ablation modes, use abl_solve_rate column
            sr_key = _get_solve_rate_key(mode)

            # Collect per-iteration values across seeds
            iter_values = defaultdict(list)
            for run in mode_runs:
                for row in run["rows"]:
                    it = int(float(row["iteration"]))
                    val = row.get(sr_key)
                    if val is not None and val != "":
                        iter_values[it].append(float(val))

            if not iter_values:
                continue

            iters = sorted(iter_values.keys())
            means = [np.mean(iter_values[i]) for i in iters]
            stds = [np.std(iter_values[i]) for i in iters]

            color = MODE_COLORS.get(mode, "gray")
            ax.plot(iters, means, label=MODE_SHORT[mode],
                    color=color, linewidth=2)
            ax.fill_between(iters,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.15, color=color)

        ax.set_title(f"D = {D}", fontsize=14)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Solve Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ablation Sweep: Solve Rate vs Training Iteration",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Learning curves saved to {output_path}")


def plot_final_bars(runs, D_values, output_path):
    """Panel 2: final solve rate bar chart, modes grouped by D."""
    fig, ax = plt.subplots(figsize=(12, 5))

    n_modes = len(MODE_ORDER)
    n_d = len(D_values)
    bar_width = 0.8 / n_d
    x = np.arange(n_modes)

    for d_idx, D in enumerate(D_values):
        d_runs = [r for r in runs if r["D"] == D]
        means = []
        stds = []

        for mode in MODE_ORDER:
            mode_runs = [r for r in d_runs if r["mode"] == mode]
            sr_key = _get_solve_rate_key(mode)

            # Get final iteration solve rate per seed
            final_vals = []
            for run in mode_runs:
                if run["rows"]:
                    val = run["rows"][-1].get(sr_key)
                    if val is not None and val != "":
                        final_vals.append(float(val))

            if final_vals:
                means.append(np.mean(final_vals))
                stds.append(np.std(final_vals))
            else:
                means.append(0)
                stds.append(0)

        offset = (d_idx - (n_d - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=f"D={D}", capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_SHORT[m] for m in MODE_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Final Solve Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Final Solve Rate by Ablation Mode and D", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Final bars saved to {output_path}")


def plot_delta_heatmap(runs, D_values, output_path):
    """Panel 3: heatmap of solve_rate(mode) - solve_rate(full)."""
    ablation_modes = [m for m in MODE_ORDER if m != "full"]
    n_modes = len(ablation_modes)
    n_d = len(D_values)

    # Build matrix
    delta = np.full((n_modes, n_d), np.nan)

    for d_idx, D in enumerate(D_values):
        d_runs = [r for r in runs if r["D"] == D]

        # Get full baseline
        full_runs = [r for r in d_runs if r["mode"] == "full"]
        full_vals = []
        for run in full_runs:
            if run["rows"]:
                val = run["rows"][-1].get("solve_rate")
                if val is not None and val != "":
                    full_vals.append(float(val))
        full_mean = np.mean(full_vals) if full_vals else 0.0

        for m_idx, mode in enumerate(ablation_modes):
            mode_runs = [r for r in d_runs if r["mode"] == mode]
            sr_key = _get_solve_rate_key(mode)
            mode_vals = []
            for run in mode_runs:
                if run["rows"]:
                    val = run["rows"][-1].get(sr_key)
                    if val is not None and val != "":
                        mode_vals.append(float(val))
            if mode_vals:
                delta[m_idx, d_idx] = np.mean(mode_vals) - full_mean

    fig, ax = plt.subplots(figsize=(max(6, n_d * 2), max(4, n_modes * 0.6)))
    im = ax.imshow(delta, cmap="RdYlGn", vmin=-1, vmax=0.2, aspect="auto")

    ax.set_xticks(range(n_d))
    ax.set_xticklabels([f"D={d}" for d in D_values])
    ax.set_yticks(range(n_modes))
    ax.set_yticklabels([MODE_SHORT[m] for m in ablation_modes])

    # Annotate cells
    for i in range(n_modes):
        for j in range(n_d):
            val = delta[i, j]
            if not np.isnan(val):
                color = "white" if val < -0.5 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    ax.set_title("Ablation Delta: solve_rate(mode) - solve_rate(full)",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Delta heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ablation sweep results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp-dir", type=Path,
                        default=Path("experiments/doors_direct"),
                        help="Directory containing experiment subdirs")
    parser.add_argument("--D", type=int, nargs="*", default=None,
                        help="Filter to specific D values (default: all found)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots (default: exp-dir)")

    args = parser.parse_args()
    output_dir = args.output_dir or args.exp_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(args.exp_dir)
    if not runs:
        print(f"No eval_summary.csv files found under {args.exp_dir}",
              file=sys.stderr)
        sys.exit(1)

    # Determine D values
    all_D = sorted(set(r["D"] for r in runs))
    if args.D:
        D_values = [d for d in all_D if d in args.D]
    else:
        D_values = all_D

    # Summary
    modes_found = sorted(set(r["mode"] for r in runs))
    seeds_found = sorted(set(r["seed"] for r in runs))
    print(f"Found {len(runs)} runs: D={D_values}, "
          f"modes={modes_found}, seeds={seeds_found}\n")

    plot_learning_curves(runs, D_values,
                         output_dir / "ablation_learning_curves.png")
    plot_final_bars(runs, D_values,
                    output_dir / "ablation_final_bars.png")
    plot_delta_heatmap(runs, D_values,
                       output_dir / "ablation_delta_heatmap.png")

    # Training loss curves
    train_runs = _discover_train_stats(args.exp_dir)
    if train_runs:
        train_D = [d for d in D_values
                   if any(r["D"] == d for r in train_runs)]
        if train_D:
            plot_loss_curves(train_runs, train_D,
                             output_dir / "ablation_loss_curves.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
